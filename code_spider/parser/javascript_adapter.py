"""JavaScript adapter — Tree-sitter parser + symbol/route/HTTP/Kafka extractor."""

from __future__ import annotations

from typing import Final

import tree_sitter_javascript as ts_js
from tree_sitter import Language

from code_spider.messaging.ts_kafka import extract_ts_kafka
from code_spider.parser._ts_js_base import TsJsAdapterBase
from code_spider.routes.ts_routes import (
    extract_ts_http_clients,
    extract_ts_routes,
)
from code_spider.symbols.fqn import file_to_module_fqn
from code_spider.symbols.model import FileRecord

_JS_LANGUAGE: Final = Language(ts_js.language())


class JavaScriptAdapter(TsJsAdapterBase):
    """Parser for JavaScript / JSX sources."""

    lang: str = "javascript"
    extensions: tuple[str, ...] = (".js", ".jsx", ".mjs", ".cjs")

    def __init__(self) -> None:
        super().__init__(_JS_LANGUAGE)

    def parse_file(self, repo_relative_path: str, source: bytes) -> FileRecord:
        record = super().parse_file(repo_relative_path, source)
        tree = self._parser.parse(source)
        module_fqn = file_to_module_fqn(repo_relative_path, self.lang)
        routes = extract_ts_routes(
            file_path=repo_relative_path,
            source=source,
            root=tree.root_node,
            module_fqn=module_fqn,
        )
        http_clients = extract_ts_http_clients(
            file_path=repo_relative_path,
            source=source,
            root=tree.root_node,
            module_fqn=module_fqn,
        )
        kafka_producers, kafka_consumers = extract_ts_kafka(
            file_path=repo_relative_path,
            source=source,
            root=tree.root_node,
            module_fqn=module_fqn,
        )
        record.routes.extend(routes)
        record.http_clients.extend(http_clients)
        record.kafka_producers.extend(kafka_producers)
        record.kafka_consumers.extend(kafka_consumers)
        return record
