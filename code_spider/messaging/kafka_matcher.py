"""Materialise KAFKA_FLOW edges from collected producers + consumers.

For each topic name present on both a :class:`KafkaProducer` and a
:class:`KafkaConsumer`, the matcher emits one :class:`KafkaFlowEdge` per
(producer, consumer) pair. Dynamic topics (``<dynamic>``) are excluded —
they can never form a deterministic edge.
"""

from __future__ import annotations

from collections import defaultdict

from code_spider.logging_setup import get_logger
from code_spider.symbols.model import (
    KafkaConsumer,
    KafkaFlowEdge,
    KafkaProducer,
    ParseResult,
    WorkspaceParseBundle,
)

_log = get_logger(__name__)


#: Placeholder topic name used when the producer/consumer call's first arg is
#: not a string literal (e.g. ``producer.send(topic_var, ...)``). The flow
#: matcher and per-repo counters both exclude this sentinel because it can
#: never form a deterministic edge — and because the vast majority of
#: heuristic false positives (``requests.Session().send(req)``,
#: ``observable.subscribe(handler)`` etc.) land here.
DYNAMIC_TOPIC = "<dynamic>"

# Backwards-compatible alias for any in-tree callers still using the
# underscore-prefixed name. New code should import ``DYNAMIC_TOPIC``.
_DYNAMIC_TOPIC = DYNAMIC_TOPIC


def match_kafka_flows(bundle: WorkspaceParseBundle) -> list[KafkaFlowEdge]:
    producers_by_topic: dict[str, list[tuple[str, KafkaProducer]]] = defaultdict(list)
    consumers_by_topic: dict[str, list[tuple[str, KafkaConsumer]]] = defaultdict(list)

    for pr in bundle.repos:
        for f in pr.files:
            for p in f.kafka_producers:
                if p.topic_name == DYNAMIC_TOPIC:
                    continue
                producers_by_topic[p.topic_name].append((pr.repo_name, p))
            for c in f.kafka_consumers:
                if c.topic_name == DYNAMIC_TOPIC:
                    continue
                consumers_by_topic[c.topic_name].append((pr.repo_name, c))

    edges: list[KafkaFlowEdge] = []
    for topic, producers in producers_by_topic.items():
        consumers = consumers_by_topic.get(topic, [])
        for prod_repo, prod in producers:
            for cons_repo, cons in consumers:
                edges.append(
                    KafkaFlowEdge(
                        producer_caller_fqn=prod.caller_fqn,
                        producer_repo=prod_repo,
                        consumer_caller_fqn=cons.caller_fqn,
                        consumer_repo=cons_repo,
                        topic_name=topic,
                    )
                )

    _log.info(
        "kafka_flow edges materialised",
        topics=len(producers_by_topic | consumers_by_topic),
        edges=len(edges),
    )
    return edges


def count_repo_topics(pr: ParseResult) -> int:
    """Count distinct non-dynamic Kafka topics this repo touches.

    A "real" Kafka usage in a service shows up as at least one producer or
    consumer with a **string-literal** topic name. The Kafka extractor's
    method-name heuristic (matches ``.send``, ``.subscribe``, ``.produce``,
    ``.assign``) over-counts on common Python idioms — ``requests.Session
    .send(req)``, Django signals, observer patterns, ``multiprocessing.Pipe
    .send`` — but those all have non-string first args and so end up as
    :data:`DYNAMIC_TOPIC`. Filtering by ``topic_name != DYNAMIC_TOPIC`` is
    exactly the same gate :func:`match_kafka_flows` already applies; using
    it for the per-repo counter keeps the CLI honest:

        ``kafka`` column = "topics this repo actually touches"

    not "how many call-sites happen to use a method named ``.send``".

    Returns ``0`` when there's no string-literal evidence — which for
    services that don't use Kafka is virtually always the right answer.
    """
    topics: set[str] = set()
    for f in pr.files:
        for p in f.kafka_producers:
            if p.topic_name != DYNAMIC_TOPIC:
                topics.add(p.topic_name)
        for c in f.kafka_consumers:
            if c.topic_name != DYNAMIC_TOPIC:
                topics.add(c.topic_name)
    return len(topics)
