import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORDERS = {
    "55231": {
        "order_id": "55231",
        "status": "shipped",
        "carrier": "BlueDart",
        "eta": "2025-01-17",
        "customer_email": "priya.sharma@example.co.in",
        "ship_to": "12 MG Road, Bengaluru",
    },
    "55232": {
        "order_id": "55232",
        "status": "processing",
        "carrier": None,
        "eta": None,
        "customer_email": "rahul@example.com",
        "ship_to": "7 Park Street, Kolkata",
    },
    "55234": {
        "order_id": "55234",
        "status": "delivered",
        "carrier": "Delhivery",
        "eta": None,
        "customer_email": "a@example.com",
        "ship_to": "1 Main St",
    },
    "55235": {
        "order_id": "55235",
        "status": "shipped",
        "carrier": "DTDC",
        "eta": "2025-01-18",
        "customer_email": "b@example.com",
        "ship_to": "2 Main St",
    },
}
HITS: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/__reset":
            HITS.clear()
            return self._send(200, {"ok": True})
        if self.path.startswith("/__hits/"):
            return self._send(200, {"hits": HITS.get(self.path.rsplit("/", 1)[-1], 0)})
        if self.headers.get("X-Api-Key") != "dev-key":
            return self._send(401, {"error": "unauthorized"})
        order_id = self.path.rstrip("/").rsplit("/", 1)[-1]
        HITS[order_id] = HITS.get(order_id, 0) + 1
        if order_id == "55233":
            return self._send(500, {"error": "internal"})
        if order_id == "55234" and HITS[order_id] <= 2:
            return self._send(429, {"error": "rate_limited"}, {"Retry-After": "1"})
        if order_id == "55235":
            time.sleep(10)
        if order_id == "55236":
            return self._send(200, b"<html>bad gateway</html>")
        if order_id in ORDERS:
            return self._send(200, ORDERS[order_id])
        return self._send(404, {"error": "not_found"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8099), Handler).serve_forever()
