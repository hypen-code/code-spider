"""Incremental diff tests (pure logic, no Neo4j)."""

from __future__ import annotations

from code_spider.incremental import compute_diff


def test_compute_diff_classifies_changed_unchanged_deleted() -> None:
    existing = {
        "a.py": "aaaaaaaa",  # unchanged
        "b.py": "bbbbbbbb",  # will change
        "c.py": "cccccccc",  # deleted on disk
    }
    on_disk = [
        ("a.py", b"AAA"),  # hash will differ from 'aaaaaaaa', so changed
        ("b.py", b"different content"),  # changed
        ("d.py", b"new file"),  # added
    ]
    # Real BLAKE3 hashes:
    import blake3

    # Force "a.py" to be unchanged by feeding the hashed bytes.
    a_bytes = b"matching content"
    a_hash = blake3.blake3(a_bytes).hexdigest()
    existing = {
        "a.py": a_hash,
        "b.py": "bbbbbbbb",
        "c.py": "cccccccc",
    }
    on_disk = [
        ("a.py", a_bytes),
        ("b.py", b"different content"),
        ("d.py", b"new file"),
    ]
    diff = compute_diff(
        workspace_id="ws",
        repo_name="repo",
        existing_hashes=existing,
        on_disk_files=on_disk,
    )
    assert diff.unchanged_paths == frozenset({"a.py"})
    assert diff.changed_paths == frozenset({"b.py", "d.py"})
    assert diff.deleted_paths == frozenset({"c.py"})
    assert set(diff.new_hashes.keys()) == {"b.py", "d.py"}
    assert diff.has_work
    assert diff.total == 4


def test_compute_diff_no_existing_marks_everything_as_changed() -> None:
    diff = compute_diff(
        workspace_id="ws",
        repo_name="repo",
        existing_hashes={},
        on_disk_files=[("x.py", b"hello"), ("y.py", b"world")],
    )
    assert diff.changed_paths == frozenset({"x.py", "y.py"})
    assert not diff.unchanged_paths
    assert not diff.deleted_paths


def test_compute_diff_no_disk_files_marks_everything_deleted() -> None:
    diff = compute_diff(
        workspace_id="ws",
        repo_name="repo",
        existing_hashes={"old.py": "deadbeef"},
        on_disk_files=[],
    )
    assert diff.deleted_paths == frozenset({"old.py"})
    assert not diff.changed_paths
    assert not diff.unchanged_paths


def test_compute_diff_idempotent_when_no_changes() -> None:
    import blake3

    src = b"matching content"
    h = blake3.blake3(src).hexdigest()
    diff = compute_diff(
        workspace_id="ws",
        repo_name="repo",
        existing_hashes={"only.py": h},
        on_disk_files=[("only.py", src)],
    )
    assert not diff.has_work
    assert diff.unchanged_paths == frozenset({"only.py"})
