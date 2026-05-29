"""HTTP_FLOW + KAFKA_FLOW matcher tests."""

from __future__ import annotations

from textwrap import dedent

from code_spider.messaging.kafka_matcher import match_kafka_flows
from code_spider.parser import get_adapter
from code_spider.routes.matcher import match_http_flows
from code_spider.symbols.model import ParseResult, WorkspaceParseBundle


def _parse(source: str, path: str, lang: str = "python"):
    return get_adapter(lang).parse_file(path, dedent(source).encode("utf-8"))


def _bundle(*files_per_repo: tuple[str, list]) -> WorkspaceParseBundle:
    """Build a WorkspaceParseBundle from ``(repo_name, [FileRecord, ...])`` pairs."""
    bundle = WorkspaceParseBundle(
        workspace_id="ws", workspace_name="Ws", manifest_sha="x"
    )
    for repo_name, files in files_per_repo:
        bundle.repos.append(
            ParseResult(
                workspace_id="ws",
                repo_name=repo_name,
                commit_sha="sha",
                files=list(files),
            )
        )
    return bundle


def test_http_flow_matcher_links_client_to_route_in_different_repo() -> None:
    server = _parse(
        """
        from fastapi import FastAPI
        app = FastAPI()

        @app.get('/users/{id}')
        def get_user(id: int):
            return id
        """,
        "server/api.py",
    )
    client = _parse(
        """
        import requests

        def fetch_user(uid: int):
            return requests.get(f'/users/{uid}')
        """,
        "client/svc.py",
    )
    bundle = _bundle(("server", [server]), ("client", [client]))
    edges = match_http_flows(bundle)
    assert edges, "expected at least one HTTP_FLOW edge"
    assert any(
        e.method == "GET"
        and e.path_template == "/users/{}"
        and e.route_repo == "server"
        and e.client_repo == "client"
        for e in edges
    )


def test_http_flow_methods_must_align() -> None:
    server = _parse(
        """
        from fastapi import FastAPI
        app = FastAPI()

        @app.get('/things')
        def get_things(): return []
        """,
        "server/api.py",
    )
    client = _parse(
        """
        import requests
        def post_thing(): return requests.post('/things')
        """,
        "client/svc.py",
    )
    bundle = _bundle(("server", [server]), ("client", [client]))
    edges = match_http_flows(bundle)
    assert not edges, "GET route should not match a POST client"


def test_kafka_flow_matcher_links_producers_to_consumers_via_topic() -> None:
    producer = _parse(
        """
        from kafka import KafkaProducer

        def publish():
            producer = KafkaProducer()
            producer.send('orders', b'payload')
        """,
        "svc_a/producer.py",
    )
    consumer = _parse(
        """
        from kafka import KafkaConsumer

        def listen():
            consumer = KafkaConsumer('orders')
            for msg in consumer:
                pass
        """,
        "svc_b/consumer.py",
    )
    bundle = _bundle(("svc_a", [producer]), ("svc_b", [consumer]))
    edges = match_kafka_flows(bundle)
    assert edges, f"expected one KAFKA_FLOW edge, got {edges}"
    edge = edges[0]
    assert edge.topic_name == "orders"
    assert edge.producer_repo == "svc_a"
    assert edge.consumer_repo == "svc_b"


def test_kafka_flow_skips_dynamic_topics() -> None:
    producer = _parse(
        """
        from kafka import KafkaProducer

        def publish(topic):
            producer = KafkaProducer()
            producer.send(topic, b'payload')
        """,
        "svc/p.py",
    )
    consumer = _parse(
        """
        from kafka import KafkaConsumer

        def listen(topic):
            consumer = KafkaConsumer(topic)
        """,
        "svc/c.py",
    )
    bundle = _bundle(("svc", [producer, consumer]))
    edges = match_kafka_flows(bundle)
    assert not edges, "dynamic topics should never form a KAFKA_FLOW edge"
