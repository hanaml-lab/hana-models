"""A simulated OpenAI-compatible streaming server for exercising serve-bench offline.

Latency follows a simple model: time to first token = base + per-prompt-token cost (prefill),
then one token every ``itl_ms``. The profiles below are simulation settings chosen to show the
shape of the trade-off (prefix caching cuts prefill; speculative decoding cuts per-token time);
they are not measurements of any server.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class Profile:
    name: str
    ttft_base_ms: float
    prefill_ms_per_1k_tokens: float
    itl_ms: float
    cached_prefix_share: float = 0.0  # share of prompt tokens served from the prefix cache


PROFILES = {
    "bf16": Profile("bf16", 30, 40, 6.0),
    "fp8-eagle-prefix": Profile("fp8-eagle-prefix", 20, 25, 2.5, cached_prefix_share=0.6),
}


def serve(profile: Profile, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            chars = sum(len(m.get("content", "")) for m in body.get("messages", []))
            prompt_tokens = chars / 4 * (1 - profile.cached_prefix_share)
            n = int(body.get("max_tokens", 64))
            time.sleep((profile.ttft_base_ms + profile.prefill_ms_per_1k_tokens * prompt_tokens / 1000) / 1000)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(n):
                chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(profile.itl_ms / 1000)
            usage = {"choices": [], "usage": {"prompt_tokens": int(chars / 4), "completion_tokens": n}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}/v1"
