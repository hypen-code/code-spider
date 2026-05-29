"""HTTP_FLOW + KAFKA_FLOW matcher tests."""

from __future__ import annotations

from textwrap import dedent

from code_spider.messaging.kafka_matcher import count_repo_topics, match_kafka_flows
from code_spider.parser import get_adapter
from code_spider.routes.matcher import match_http_flows
from code_spider.symbols.model import ParseResult, WorkspaceParseBundle


def _parse(source: str, path: str, lang: str = "python"):
    return get_adapter(lang).parse_file(path, dedent(source).encode("utf-8"))


def _bundle(*files_per_repo: tuple[str, list]) -> WorkspaceParseBundle:
    """Build a WorkspaceParseBundle from ``(repo_name, [FileRecord, ...])`` pairs."""
    bundle = WorkspaceParseBundle(workspace_id="ws", workspace_name="Ws", manifest_sha="x")
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


# --------------------------------------------------------------------------- #
# count_repo_topics — the developer-facing "does this repo use Kafka?" signal #
# --------------------------------------------------------------------------- #


def _pr(repo_name: str, *files) -> ParseResult:
    return ParseResult(workspace_id="ws", repo_name=repo_name, commit_sha="sha", files=list(files))


def test_count_repo_topics_returns_zero_for_pure_false_positives() -> None:
    """The whole point of the column change: a service that never imports
    Kafka but happens to call ``.send`` / ``.subscribe`` on HTTP clients
    and observer patterns must report **zero** Kafka topics.

    The chiron repo in the user's screenshots had 38 such call-sites; with
    the new logic that column collapses to 0 because every topic argument
    is a variable (``<dynamic>``).
    """
    fr = _parse(
        """
        import requests

        def fetch(req):
            session = requests.Session()
            session.send(req)         # heuristic matches .send → DYNAMIC

        def watch(observable, handler):
            observable.subscribe(handler)  # matches .subscribe → DYNAMIC

        def push(pipe, message):
            pipe.send(message)        # multiprocessing.Pipe → DYNAMIC
        """,
        "svc/noise.py",
    )
    # The heuristic must still have matched call-sites (sanity-check our
    # premise) — but every one of them is a dynamic topic.
    assert fr.kafka_producers or fr.kafka_consumers, (
        "test invariant: the heuristic should fire on these idioms; if it "
        "stops firing, this regression test loses its meaning."
    )
    assert all(p.topic_name == "<dynamic>" for p in fr.kafka_producers)
    assert all(c.topic_name == "<dynamic>" for c in fr.kafka_consumers)

    assert count_repo_topics(_pr("noisy", fr)) == 0


def test_count_repo_topics_counts_distinct_literal_topics() -> None:
    """Real Kafka usage with string-literal topics is counted correctly."""
    fr = _parse(
        """
        from kafka import KafkaProducer, KafkaConsumer

        def publish():
            p = KafkaProducer()
            p.send('orders', b'')
            p.send('payments', b'')

        def listen():
            c = KafkaConsumer('orders')   # overlaps with producer topic
            c.subscribe(['audit-log'])    # new topic via subscribe(list)
        """,
        "svc/kafka_io.py",
    )
    # Expected distinct non-dynamic topics: {orders, payments, audit-log} = 3
    # (orders counted once even though it appears as both producer + consumer).
    assert count_repo_topics(_pr("real", fr)) == 3


def test_count_repo_topics_ignores_dynamic_among_real() -> None:
    """A repo with a mix of real and dynamic topics counts only the reals."""
    fr = _parse(
        """
        from kafka import KafkaProducer

        def emit(topic_from_config):
            p = KafkaProducer()
            p.send('telemetry', b'')          # real
            p.send(topic_from_config, b'')    # dynamic — ignored
        """,
        "svc/kafka_emit.py",
    )
    assert count_repo_topics(_pr("mixed", fr)) == 1


def test_count_repo_topics_zero_for_empty_repo() -> None:
    """No files → 0 (no crash, no divide-by-zero in callers)."""
    assert count_repo_topics(_pr("empty")) == 0
