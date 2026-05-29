"""AST-aware chunker for hybrid search.

Algorithm (per file):
    1. Group top-level :class:`Symbol`s (those whose ``parent_fqn`` equals the
       module FQN). These are functions, classes, type aliases, interfaces,
       and top-level constants.
    2. Each becomes a single chunk spanning ``[start_line, end_line]``.
    3. If any chunk would exceed ``max_lines``, split it at line boundaries
       with a ``overlap_lines`` overlap (token counts are approximated by
       whitespace word count).
    4. Cover any *unchunked* span of the file with file-level chunks so
       module-scope code is searchable too.

Chunk IDs are stable: ``blake2b(repo:file:start-end)``, so reindexing a
file at the same coordinates reuses the same node and avoids node bloat.
"""

from __future__ import annotations

import hashlib

from code_spider.symbols.model import Chunk, FileRecord, Span

#: Target maximum lines per chunk before splitting kicks in.
DEFAULT_MAX_LINES = 200
#: Overlap (in lines) between consecutive split chunks of one definition.
DEFAULT_OVERLAP_LINES = 5


def chunk_file(
    *,
    workspace_id: str,
    repo_name: str,
    file_record: FileRecord,
    source: bytes,
    max_lines: int = DEFAULT_MAX_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[Chunk]:
    """Generate Chunks for one parsed file. Empty list for tiny/empty files."""
    if file_record.size_bytes == 0:
        return []

    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return []

    top_level = _top_level_symbols(file_record)
    chunks: list[Chunk] = []
    covered: set[int] = set()

    for sym in top_level:
        for chunk in _split_to_chunks(
            workspace_id=workspace_id,
            repo_name=repo_name,
            file_path=file_record.repo_relative_path,
            start_line=sym.span.start_line,
            end_line=min(sym.span.end_line, len(lines)),
            lines=lines,
            max_lines=max_lines,
            overlap_lines=overlap_lines,
        ):
            chunks.append(chunk)
            covered.update(
                range(chunk.span.start_line, chunk.span.end_line + 1)
            )

    # Cover module-scope gaps with file-level chunks.
    chunks.extend(
        _cover_gaps(
            workspace_id=workspace_id,
            repo_name=repo_name,
            file_path=file_record.repo_relative_path,
            lines=lines,
            covered=covered,
            max_lines=max_lines,
        )
    )
    return chunks


def _top_level_symbols(fr: FileRecord) -> list:
    if fr.module is None:
        return []
    module_fqn = fr.module.fqn
    return sorted(
        (s for s in fr.symbols if s.parent_fqn == module_fqn),
        key=lambda s: s.span.start_line,
    )


def _split_to_chunks(
    *,
    workspace_id: str,
    repo_name: str,
    file_path: str,
    start_line: int,
    end_line: int,
    lines: list[str],
    max_lines: int,
    overlap_lines: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    cur_start = start_line
    while cur_start <= end_line:
        cur_end = min(cur_start + max_lines - 1, end_line)
        text = "\n".join(lines[cur_start - 1 : cur_end])
        out.append(
            Chunk(
                chunk_id=_chunk_id(workspace_id, repo_name, file_path, cur_start, cur_end),
                file_path=file_path,
                span=Span(
                    start_line=cur_start,
                    start_col=0,
                    end_line=cur_end,
                    end_col=len(lines[cur_end - 1]) if cur_end - 1 < len(lines) else 0,
                ),
                text=text,
            )
        )
        if cur_end >= end_line:
            break
        cur_start = cur_end - overlap_lines + 1
        if cur_start <= 0:
            cur_start = 1
    return out


def _cover_gaps(
    *,
    workspace_id: str,
    repo_name: str,
    file_path: str,
    lines: list[str],
    covered: set[int],
    max_lines: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    total = len(lines)
    if total == 0:
        return out

    i = 1
    while i <= total:
        if i in covered:
            i += 1
            continue
        gap_start = i
        while i <= total and i not in covered:
            i += 1
        gap_end = i - 1
        # Skip whitespace-only gaps.
        if all(not lines[j - 1].strip() for j in range(gap_start, gap_end + 1)):
            continue
        out.extend(
            _split_to_chunks(
                workspace_id=workspace_id,
                repo_name=repo_name,
                file_path=file_path,
                start_line=gap_start,
                end_line=gap_end,
                lines=lines,
                max_lines=max_lines,
                overlap_lines=0,
            )
        )
    return out


def _chunk_id(
    workspace_id: str, repo_name: str, file_path: str, start: int, end: int
) -> str:
    key = f"{workspace_id}:{repo_name}:{file_path}:{start}-{end}".encode()
    return "ck_" + hashlib.blake2b(key, digest_size=12).hexdigest()
