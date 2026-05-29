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
    WorkspaceParseBundle,
)

_log = get_logger(__name__)


_DYNAMIC_TOPIC = "<dynamic>"


def match_kafka_flows(bundle: WorkspaceParseBundle) -> list[KafkaFlowEdge]:
    producers_by_topic: dict[str, list[tuple[str, KafkaProducer]]] = defaultdict(list)
    consumers_by_topic: dict[str, list[tuple[str, KafkaConsumer]]] = defaultdict(list)

    for pr in bundle.repos:
        for f in pr.files:
            for p in f.kafka_producers:
                if p.topic_name == _DYNAMIC_TOPIC:
                    continue
                producers_by_topic[p.topic_name].append((pr.repo_name, p))
            for c in f.kafka_consumers:
                if c.topic_name == _DYNAMIC_TOPIC:
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
