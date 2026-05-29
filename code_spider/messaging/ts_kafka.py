"""Kafka producer + consumer detection for TypeScript / JavaScript sources.

Targets (heuristic, MVP):
    - ``kafkajs``      — ``kafka.producer().send({topic, ...})``,
                         ``consumer.subscribe({topic, ...})``
    - ``node-rdkafka`` — ``producer.produce(topic, ...)``,
                         ``consumer.subscribe([topics])``

Topic names are only captured when they appear as string literals; dynamic
topics produce a ``<dynamic>`` placeholder and never participate in a
KAFKA_FLOW edge.
"""

from __future__ import annotations

from tree_sitter import Node

from code_spider.symbols.fqn import qualify
from code_spider.symbols.model import KafkaConsumer, KafkaProducer, Span

_PRODUCER_METHODS: frozenset[str] = frozenset({"send", "produce", "sendBatch"})
_CONSUMER_METHODS: frozenset[str] = frozenset({"subscribe", "assign"})


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _span(node: Node) -> Span:
    sr, sc = node.start_point
    er, ec = node.end_point
    return Span(start_line=sr + 1, start_col=sc, end_line=er + 1, end_col=ec)


def _string_value(node: Node, source: bytes) -> str | None:
    if node.type == "string":
        raw = _text(node, source)
        if len(raw) >= 2 and raw[0] in ("'", '"', "`") and raw[-1] == raw[0]:
            return raw[1:-1]
        return raw
    if node.type == "template_string":
        raw = _text(node, source)
        inner = raw[1:-1] if raw.startswith("`") and raw.endswith("`") else raw
        if "${" in inner:
            return None
        return inner
    return None


def _attr_chain(node: Node, source: bytes) -> list[str]:
    parts: list[str] = []
    current: Node | None = node
    while current is not None:
        if current.type in {"identifier", "property_identifier"}:
            parts.append(_text(current, source))
            break
        if current.type == "member_expression":
            prop = current.child_by_field_name("property")
            obj = current.child_by_field_name("object")
            if prop is not None:
                parts.append(_text(prop, source))
            current = obj
            continue
        break
    parts.reverse()
    return parts


def _topic_from_object(obj_node: Node, source: bytes) -> str | None:
    """Find ``{ topic: 'name' }`` inside an object literal."""
    for pair in obj_node.named_children:
        if pair.type != "pair":
            continue
        key = pair.child_by_field_name("key")
        value = pair.child_by_field_name("value")
        if key is None or value is None:
            continue
        key_text = _text(key, source).strip("'\"")
        if key_text not in {"topic", "topics"}:
            continue
        if value.type == "array":
            for el in value.named_children:
                v = _string_value(el, source)
                if v:
                    return v
            return None
        return _string_value(value, source)
    return None


def _first_positional(args: Node) -> Node | None:
    for arg in args.named_children:
        if arg.type == "comment":
            continue
        return arg
    return None


def extract_ts_kafka(
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
        if child.type in {
            "function_declaration",
            "method_definition",
            "class_declaration",
            "abstract_class_declaration",
        }:
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

        if child.type in {"lexical_declaration", "variable_declaration"}:
            for decl in child.named_children:
                if decl.type != "variable_declarator":
                    continue
                name_node = decl.child_by_field_name("name")
                value = decl.child_by_field_name("value")
                if (
                    value is not None
                    and value.type in {"arrow_function", "function_expression"}
                    and name_node is not None
                    and name_node.type == "identifier"
                ):
                    inner = qualify(caller_fqn, _text(name_node, source))
                    body = value.child_by_field_name("body")
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
                _walk(
                    node=decl,
                    source=source,
                    file_path=file_path,
                    producers=producers,
                    consumers=consumers,
                    caller_fqn=caller_fqn,
                )
            continue

        if child.type == "call_expression":
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
    if func is None or args is None or func.type != "member_expression":
        return

    chain = _attr_chain(func, source)
    if not chain:
        return
    method = chain[-1]

    if method in _PRODUCER_METHODS:
        topic = _topic_from_call_args(args, source)
        producers.append(
            KafkaProducer(
                caller_fqn=caller_fqn,
                topic_name=topic or "<dynamic>",
                client_lib="kafkajs",
                file_path=file_path,
                span=_span(call),
            )
        )
        return

    if method in _CONSUMER_METHODS:
        topic = _topic_from_call_args(args, source)
        if topic:
            consumers.append(
                KafkaConsumer(
                    caller_fqn=caller_fqn,
                    topic_name=topic,
                    client_lib="kafkajs",
                    group_id=None,
                    file_path=file_path,
                    span=_span(call),
                )
            )
        else:
            consumers.append(
                KafkaConsumer(
                    caller_fqn=caller_fqn,
                    topic_name="<dynamic>",
                    client_lib="kafkajs",
                    group_id=None,
                    file_path=file_path,
                    span=_span(call),
                )
            )


def _topic_from_call_args(args: Node, source: bytes) -> str | None:
    first = _first_positional(args)
    if first is None:
        return None
    if first.type in {"string", "template_string"}:
        return _string_value(first, source)
    if first.type == "object":
        return _topic_from_object(first, source)
    if first.type == "array":
        for el in first.named_children:
            v = _string_value(el, source)
            if v:
                return v
    return None
