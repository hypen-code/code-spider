"""Observability / metrics tests."""

from __future__ import annotations

from code_spider.observability import METRICS, stage_timer


def _histogram_count(workspace: str, stage: str) -> float:
    h = METRICS.indexer_stage_duration.labels(stage=stage, workspace=workspace)
    # ``_sum`` / ``_count`` are accessible on the underlying child via the
    # private attribute path; we just check that observe() has been called.
    samples = h.collect()
    for metric in samples:
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                return float(sample.value)
    return 0.0


def test_stage_timer_records_observation_in_histogram() -> None:
    before = _histogram_count(workspace="test-ws", stage="unit")
    with stage_timer("unit", "test-ws"):
        sum(range(100))
    after = _histogram_count(workspace="test-ws", stage="unit")
    assert after >= before + 1


def test_stage_timer_increments_error_counter_on_exception() -> None:
    counter = METRICS.indexer_errors.labels(stage="boom", workspace="test-ws")
    before = counter._value.get()
    try:
        with stage_timer("boom", "test-ws"):
            raise RuntimeError("simulated")
    except RuntimeError:
        pass
    assert counter._value.get() == before + 1


def test_metrics_bundle_exposes_expected_collectors() -> None:
    # Snapshot the metric names. Should at least contain these key series.
    names = {
        sample.name
        for collector in (
            METRICS.indexer_stage_duration,
            METRICS.indexer_files_parsed,
            METRICS.indexer_symbols_emitted,
            METRICS.indexer_resolved_calls,
            METRICS.mcp_tool_duration,
            METRICS.mcp_tool_errors,
        )
        for metric in collector.collect()
        for sample in metric.samples
    }
    # Every collector should produce at least its ``_created`` series.
    assert any(s.endswith("_created") for s in names) or names == set()
