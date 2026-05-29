"""AST-aware code chunker for hybrid search."""

from code_spider.chunker.ast_chunker import (
    DEFAULT_MAX_LINES,
    DEFAULT_OVERLAP_LINES,
    chunk_file,
)

__all__ = ["DEFAULT_MAX_LINES", "DEFAULT_OVERLAP_LINES", "chunk_file"]
