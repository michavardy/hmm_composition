from __future__ import annotations

import sys
import tomllib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.setup_logger import get_logger
from apps.experiment_table import build_table_html

logger = get_logger("DASHBOARD")
CONFIG_PATH = ROOT_DIR / "utils" / "config.toml"
TEMPLATE_PATH = ROOT_DIR / "apps" / "templates" / "index.html"
TABLE_TEMPLATE_PATH = ROOT_DIR / "apps" / "templates" / "table.html"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as config_file:
        return tomllib.load(config_file)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def build_page() -> bytes:
    config = load_config()
    dashboard_config = config["dashboard"]
    experiments_dir = resolve_path(dashboard_config["experiments_dir"])
    table_html = build_table_html(experiments_dir, TABLE_TEMPLATE_PATH)
    html_page = TEMPLATE_PATH.read_text(encoding="utf-8")
    html_page = html_page.replace("{{TITLE}}", dashboard_config["title"])
    html_page = html_page.replace("{{TABLE_VIEW}}", table_html)
    return html_page.encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not Found")
            return

        self._send(HTTPStatus.OK, "text/html; charset=utf-8", build_page())

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    config = load_config()
    dashboard_config = config["dashboard"]
    host = dashboard_config["host"]
    port = int(dashboard_config["port"])

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    logger.info("Serving dashboard template on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()