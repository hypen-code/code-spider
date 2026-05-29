"""AST-aware chunker tests."""

from __future__ import annotations

from textwrap import dedent

from code_spider.chunker import chunk_file
from code_spider.parser import get_adapter


def _parse(source: str, path: str = "pkg/sample.py"):
    src = dedent(source).encode("utf-8")
    fr = get_adapter("python").parse_file(path, src)
    return src, fr


def test_chunks_one_per_top_level_function() -> None:
    src, fr = _parse(
        """
        def first():
            return 1

        def second():
            return 2
        """
    )
    chunks = chunk_file(workspace_id="ws", repo_name="repo", file_record=fr, source=src)
    # Two top-level functions = at least two chunks.
    assert len(chunks) >= 2
    fqns_in_text = [c.text for c in chunks]
    assert any("def first" in t for t in fqns_in_text)
    assert any("def second" in t for t in fqns_in_text)


def test_chunk_ids_are_stable_across_calls() -> None:
    src, fr = _parse(
        """
        def stable():
            return 1
        """
    )
    a = chunk_file(workspace_id="ws", repo_name="repo", file_record=fr, source=src)
    b = chunk_file(workspace_id="ws", repo_name="repo", file_record=fr, source=src)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunk_ids_include_workspace_for_multitenant_safety() -> None:
    src, fr = _parse(
        """
        def f():
            pass
        """
    )
    a = chunk_file(workspace_id="ws1", repo_name="repo", file_record=fr, source=src)
    b = chunk_file(workspace_id="ws2", repo_name="repo", file_record=fr, source=src)
    assert {c.chunk_id for c in a} & {c.chunk_id for c in b} == set()


def test_large_function_is_split_with_overlap() -> None:
    import itertools

    body = "\n".join(f"    x{i} = {i}" for i in range(0, 250))
    src = f"def big():\n{body}\n".encode()
    fr = get_adapter("python").parse_file("pkg/big.py", src)
    chunks = chunk_file(
        workspace_id="ws",
        repo_name="repo",
        file_record=fr,
        source=src,
        max_lines=80,
        overlap_lines=4,
    )
    assert len(chunks) > 1, "large body must split into multiple chunks"
    sorted_chunks = sorted(chunks, key=lambda c: c.span.start_line)
    for prev, nxt in itertools.pairwise(sorted_chunks):
        assert nxt.span.start_line <= prev.span.end_line + 1


def test_module_scope_gaps_are_covered() -> None:
    src, fr = _parse(
        """
        \"\"\"Module docstring.\"\"\"

        CONSTANT = 42
        print('hi')

        def used():
            return CONSTANT
        """
    )
    chunks = chunk_file(workspace_id="ws", repo_name="repo", file_record=fr, source=src)
    # ``CONSTANT = 42`` and the print are not inside ``used`` — they need a
    # module-scope chunk.
    texts = "\n---\n".join(c.text for c in chunks)
    assert "CONSTANT" in texts
    assert "def used" in texts
