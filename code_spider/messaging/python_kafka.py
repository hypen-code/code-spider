"""Kafka producer + consumer detection for Python sources.

Targets (heuristic, MVP):
    - ``kafka-python``        — ``KafkaProducer.send('topic', ...)``,
                                ``KafkaConsumer('topic', ...)``,
                                ``consumer.subscribe(['topic'])``
    - ``confluent-kafka``     — ``Producer().produce('topic', ...)``,
                                ``Consumer().subscribe(['topic'])``
    - ``aiokafka``            — ``AIOKafkaProducer.send_and_wait('topic', ...)``,
                                ``AIOKafkaConsumer('topic', ...)``
    - ``faust``               — ``@app.agent(topic)`` (topic is a variable;
                                we capture the bare name)

The result is best-effort: topic names are only extracted when they appear as
string literals. Variable-resolved topics will appear as
``KafkaTopic{name='<dynamic>'}`` and never participate in a KAFKA_FLOW edge.
"""

from __future__ import annotations

from tree_sitter import Node

from code_spider.symbols.fqn import qualify
from code_spider.symbols.model import KafkaConsumer, KafkaProducer, Span

_PRODUCER_METHODS: frozenset[str] = frozenset(
    {"send", "send_and_wait", "produce"}
)
_CONSUMER_METHODS: frozenset[str] = frozenset({"subscribe", "assign"})

# Constructors whose first positional string arg is a topic name.
_CONSUMER_CONSTRUCTORS: frozenset[str] = frozenset(
    {"KafkaConsumer", "AIOKafkaConsumer"}
)
_PRODUCER_CONSTRUCTORS: frozenset[str] = frozenset(
    {"KafkaProducer", "AIOKafkaProducer", "Producer"}
)


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _span(node: Node) -> Span:
    sr, sc = node.start_point
    er, ec = node.end_point
    return Span(start_line=sr + 1, start_col=sc, end_line=er + 1, end_col=ec)


def _string_value(node: Node, source: bytes) -> str | None:
    if node.type != "string":
        return None
    raw = _text(node, source)
    s = raw
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    s = s[i:]
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q):
            return s[len(q) : -len(q)]
    return None


def _attr_chain(node: Node, source: bytes) -> list[str]:
    parts: list[str] = []
    current: Node | None = node
    while current is not None:
        if current.type == "identifier":
            parts.append(_text(current, source))
            break
        if current.type == "attribute":
            attr = current.child_by_field_name("attribute")
            obj = current.child_by_field_name("object")
            if attr is not None:
                parts.append(_text(attr, source))
            current = obj
            continue
        break
    parts.reverse()
    return parts


def _first_positional_string(args: Node, source: bytes) -> str | None:
    for arg in args.named_children:
        if arg.type == "string":
            return _string_value(arg, source)
        if arg.type == "keyword_argument":
            continue
        if arg.type == "list":
            # subscribe(['topic']) — return first string in list.
            for elem in arg.named_children:
                v = _string_value(elem, source)
                if v is not None:
                    return v
            return None
        return None
    return None


def _collect_string_list_arg(args: Node, source: bytes) -> list[str]:
    """For ``subscribe(['a', 'b'])`` return [a, b]."""
    for arg in args.named_children:
        if arg.type in {"list", "tuple"}:
            return [v for v in (_string_value(c, source) for c in arg.named_children) if v]
        if arg.type == "string":
            v = _string_value(arg, source)
            return [v] if v else []
        if arg.type == "keyword_argument":
            continue
        return []
    return []


def extract_python_kafka(
    *, file_path: str, source: bytes, root: Node, module_fqn: str
) -> tuple[list[KafkaProducer], list[KafkaConsumer]]:
    producers: list[KafkaProducer] = []
    consumers: list[KafkaConsumer] = []
    _walk(
        node=root,
        source=source,
        file_path=file_path,
        producers=producers,
        consumers=consumers,
        caller_fqn=module_fqn,
    )
    return producers, consumers


def _walk(
    *,
    node: Node,
    source: bytes,
    file_path: str,
    producers: list[KafkaProducer],
    consumers: list[KafkaConsumer],
    caller_fqn: str,
) -> None:
    for child in node.named_children:
        if child.type in {"function_definition", "class_definition"}:
            name_node = child.child_by_field_name("name")
            inner = (
                qualify(caller_fqn, _text(name_node, source))
                if name_node is not None
                else caller_fqn
            )
            body = child.child_by_field_name("body")
            if body is not None:
                _walk(
                    node=body,
                    source=source,
                    file_path=file_path,
                    producers=producers,
                    consumers=consumers,
                    caller_fqn=inner,
                )
            continue

        if child.type == "decorated_definition":
            inner_def = child.child_by_field_name("definition")
            if inner_def is not None:
                name_node = inner_def.child_by_field_name("name")
                inner = (
                    qualify(caller_fqn, _text(name_node, source))
                    if name_node is not None
                    else caller_fqn
                )
                # Faust: @app.agent(topic) decorator emits a consumer.
                _maybe_faust_agent(child, source, file_path, consumers, inner)
                body = inner_def.child_by_field_name("body")
                if body is not None:
                    _walk(
                        node=body,
                        source=source,
                        file_path=file_path,
                        producers=producers,
                        consumers=consumers,
                        caller_fqn=inner,
                    )
            continue

        if child.type == "call":
            _maybe_kafka_call(
                call=child,
                source=source,
                file_path=file_path,
                producers=producers,
                consumers=consumers,
                caller_fqn=caller_fqn,
            )

        _walk(
            node=child,
            source=source,
            file_path=file_path,
            producers=producers,
            consumers=consumers,
            caller_fqn=caller_fqn,
        )


def _maybe_kafka_call(
    *,
    call: Node,
    source: bytes,
    file_path: str,
    producers: list[KafkaProducer],
    consumers: list[KafkaConsumer],
    caller_fqn: str,
) -> None:
    func = call.child_by_field_name("function")
    args = call.child_by_field_name("arguments")
    if func is None or args is None:
        return

    # Constructor call: ``KafkaConsumer('topic', ...)`` / ``AIOKafkaConsumer('topic')``.
    if func.type == "identifier":
        name = _text(func, source)
        if name in _CONSUMER_CONSTRUCTORS:
            topic = _first_positional_string(args, source) or "<dynamic>"
            consumers.append(
                KafkaConsumer(
                    caller_fqn=caller_fqn,
                    topic_name=topic,
                    client_lib=_lib_for_constructor(name),
                    group_id=None,
                    file_path=file_path,
                    span=_span(call),
                )
            )
            return
        if name in _PRODUCER_CONSTRUCTORS:
            # Constructor alone does not produce; wait for .send/.produce/.send_and_wait.
            return

    if func.type != "attribute":
        return

    chain = _attr_chain(func, source)
    if not chain:
        return
    method = chain[-1]

    if method in _PRODUCER_METHODS:
        topic = _first_positional_string(args, source) or "<dynamic>"
        producers.append(
            KafkaProducer(
                caller_fqn=caller_fqn,
                topic_name=topic,
                client_lib=_lib_for_method(chain),
                file_path=file_path,
                span=_span(call),
            )
        )
        return

    if method in _CONSUMER_METHODS:
        topics = _collect_string_list_arg(args, source)
        if not topics:
            topics = ["<dynamic>"]
        for t in topics:
            consumers.append(
                KafkaConsumer(
                    caller_fqn=caller_fqn,
                    topic_name=t,
                    client_lib=_lib_for_method(chain),
                    group_id=None,
                    file_path=file_path,
                    span=_span(call),
                )
            )


def _lib_for_constructor(ctor: str) -> str:
    if ctor.startswith("AIO"):
        return "aiokafka"
    if ctor in {"Producer", "Consumer"}:
        return "confluent-kafka"
    return "kafka-python"


def _lib_for_method(chain: list[str]) -> str:
    receiver = chain[-2] if len(chain) >= 2 else ""
    if "aio" in receiver.lower():
        return "aiokafka"
    if receiver in {"producer", "consumer"}:
        return "kafka-python"
    return "kafka-python"


def _maybe_faust_agent(
    decorated: Node,
    source: bytes,
    file_path: str,
    consumers: list[KafkaConsumer],
    handler_fqn: str,
) -> None:
    """Detect ``@app.agent(topic)`` — Faust consumer pattern."""
    for dec in (c for c in decorated.named_children if c.type == "decorator"):
        if not dec.named_children:
            continue
        expr = dec.named_children[0]
        if expr.type != "call":
            continue
        func = expr.child_by_field_name("function")
        args = expr.child_by_field_name("arguments")
        if func is None or args is None or func.type != "attribute":
            continue
        chain = _attr_chain(func, source)
        if len(chain) < 2 or chain[-1] != "agent":
            continue
        # First arg is a topic variable or function call — we cannot resolve it
        # statically, so emit a placeholder name. Phase 2 may resolve via the
        # workspace symbol index.
        topic_name = "<dynamic>"
        for arg in args.named_children:
            v = _string_value(arg, source) if arg.type == "string" else None
            if v:
                topic_name = v
                break
        consumers.append(
            KafkaConsumer(
                caller_fqn=handler_fqn,
                topic_name=topic_name,
                client_lib="faust",
                group_id=None,
                file_path=file_path,
                span=_span(dec),
            )
        )
