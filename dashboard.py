"""Lightweight dashboard server for the bio-experiment agent.

No external dependencies — uses only the Python standard library.

Usage:
    python dashboard.py          # serves on http://localhost:8050
    python dashboard.py --port 9000
"""

from __future__ import annotations

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent
STATE_FILE = ROOT / "_dashboard_state.json"
CMD_FILE = ROOT / "_dashboard_cmd.json"
DASHBOARD_HTML = ROOT / "dashboard.html"


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(DASHBOARD_HTML, "text/html")
        elif self.path == "/api/state":
            self._serve_state()
        elif self.path == "/api/scenarios":
            self._serve_scenarios()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/restart":
            self._handle_command({"action": "restart"})
        elif self.path == "/api/run":
            body = self._read_body()
            if body is None:
                return
            body["action"] = "restart"
            self._handle_command(body)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
            return None

    def _handle_command(self, cmd: dict):
        CMD_FILE.write_text(json.dumps(cmd), encoding="utf-8")
        self._json_response(200, {"ok": True, "command": cmd.get("action")})

    def _serve_state(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            data = STATE_FILE.read_bytes()
        except FileNotFoundError:
            data = b'{"error": "No state file yet. Run run_agent.py to start an episode."}'
        self.wfile.write(data)

    def _serve_scenarios(self):
        try:
            from server.tasks.scenarios import SCENARIO_LIBRARY
            names = [s.name for s in SCENARIO_LIBRARY]
        except Exception:
            names = []
        self._json_response(200, {"scenarios": names})

    def _serve_file(self, path: Path, content_type: str):
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"{path.name} not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Bio-experiment dashboard server")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at  http://localhost:{args.port}")
    print("Waiting for agent state from run_agent.py ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
