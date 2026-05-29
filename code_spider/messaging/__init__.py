"""Kafka producer/consumer extraction (Phase 1).

Will produce ``:KafkaTopic``, ``:KafkaProducer``, ``:KafkaConsumer``, and
``:KAFKA_FLOW`` graph elements. Detects ``kafka-python``, ``confluent-kafka``,
``aiokafka``, ``faust`` (Python) and ``kafkajs``, ``node-rdkafka`` (TS/JS).
"""

__all__: list[str] = []
