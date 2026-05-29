"""Python Kafka extractor tests."""

from __future__ import annotations

from textwrap import dedent

from code_spider.parser import get_adapter


def _parse(source: str, path: str = "svc/kafka_io.py"):
    return get_adapter("python").parse_file(path, dedent(source).encode("utf-8"))


def test_kafka_python_producer_send() -> None:
    fr = _parse(
        """
        from kafka import KafkaProducer

        def publish():
            producer = KafkaProducer()
            producer.send('events', b'payload')
        """
    )
    topics = {p.topic_name for p in fr.kafka_producers}
    assert "events" in topics
    callers = {p.caller_fqn for p in fr.kafka_producers}
    assert "svc.kafka_io.publish" in callers


def test_kafka_python_consumer_constructor() -> None:
    fr = _parse(
        """
        from kafka import KafkaConsumer

        def listen():
            c = KafkaConsumer('orders')
            for msg in c:
                pass
        """
    )
    topics = {c.topic_name for c in fr.kafka_consumers}
    assert "orders" in topics


def test_kafka_python_consumer_subscribe_list() -> None:
    fr = _parse(
        """
        from kafka import KafkaConsumer

        def listen():
            c = KafkaConsumer()
            c.subscribe(['orders', 'payments'])
        """
    )
    topics = {c.topic_name for c in fr.kafka_consumers}
    assert "orders" in topics
    assert "payments" in topics


def test_confluent_kafka_producer_produce() -> None:
    fr = _parse(
        """
        from confluent_kafka import Producer

        def publish():
            producer = Producer({'bootstrap.servers': 'broker:9092'})
            producer.produce('telemetry', b'value')
        """
    )
    libs = {p.client_lib for p in fr.kafka_producers}
    assert any(p.topic_name == "telemetry" for p in fr.kafka_producers)
    assert libs  # at least one client lib reported


def test_aiokafka_constructor_detection() -> None:
    fr = _parse(
        """
        from aiokafka import AIOKafkaConsumer

        async def listen():
            consumer = AIOKafkaConsumer('async-topic')
        """
    )
    topics = {c.topic_name for c in fr.kafka_consumers}
    assert "async-topic" in topics
    libs = {c.client_lib for c in fr.kafka_consumers}
    assert "aiokafka" in libs


def test_dynamic_topic_yields_placeholder() -> None:
    fr = _parse(
        """
        from kafka import KafkaProducer

        def publish(name):
            producer = KafkaProducer()
            producer.send(name, b'payload')
        """
    )
    topics = {p.topic_name for p in fr.kafka_producers}
    assert "<dynamic>" in topics
