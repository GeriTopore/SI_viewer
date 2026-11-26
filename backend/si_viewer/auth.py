"""Token authentication, as Jupyter's: every request needs the server's token.

The server runs as its user and can read their files, so on a shared
machine (any user can reach a ``127.0.0.1`` port) or a cluster node, nobody
but the person given the token may use it. The token arrives once in the
URL (``?token=…``); the server answers with a cookie and sends the browser
to the same address without it, so later requests (fetches, images,
downloads) carry the cookie. Scripts send ``Authorization: token …``.

The cookie's name comes from the token, so servers on one host (tunnels to
several machines, all ``localhost``) keep their own; ``SameSite=Strict``
keeps other sites' pages from using it.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, urlencode

QUERY = "token"
# Served without the token: the tab's icon (browsers fetch it without the cookie).
OPEN_PATHS = {"/favicon.ico"}


def new_token() -> str:
    return secrets.token_urlsafe(24)


def cookie_name(token: str) -> str:
    return "si-viewer-" + hashlib.sha256(token.encode()).hexdigest()[:12]


DENIED = (
    "<!doctype html><meta charset=utf-8><title>SI Viewer</title>"
    "<body style='font:15px system-ui;margin:3em;color:#ccc;background:#1c1c1e'>"
    "<h3>SI Viewer: token needed</h3>"
    "<p>Open the address the server printed when it started (it ends in <code>?token=…</code>).</p>"
)


class TokenAuth:
    """ASGI middleware: lets a request through if it has the token (cookie,
    ``Authorization`` header, or ``?token=``); a page opened with
    ``?token=`` is answered with the cookie and a redirect without it."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token
        self.cookie = cookie_name(token)

    def _ok(self, value: str | None) -> bool:
        return bool(value) and hmac.compare_digest(value.encode(), self.token.encode())

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope["path"] in OPEN_PATHS:
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        jar = SimpleCookie()
        try:
            jar.load(headers.get("cookie", ""))
        except Exception:  # noqa: BLE001 - a malformed cookie header is just no cookie
            jar = SimpleCookie()
        if self.cookie in jar and self._ok(jar[self.cookie].value):
            return await self.app(scope, receive, send)
        auth = headers.get("authorization", "")
        if auth.lower().startswith("token ") and self._ok(auth[6:].strip()):
            return await self.app(scope, receive, send)
        query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        given = next((v for k, v in query if k == QUERY), None)
        if self._ok(given):
            # A page opened in a browser (not a script's API call, which keeps no cookie).
            page = "text/html" in headers.get("accept", "") and not scope["path"].startswith("/api/")
            if scope["type"] == "http" and scope["method"] in ("GET", "HEAD") and page:
                # Set the cookie and drop the token from the address (and history).
                rest = urlencode([(k, v) for k, v in query if k != QUERY])
                location = scope.get("root_path", "") + scope["path"] + (f"?{rest}" if rest else "")
                cookie = f"{self.cookie}={self.token}; Path=/; HttpOnly; SameSite=Strict"
                await send({"type": "http.response.start", "status": 303, "headers": [
                    (b"location", location.encode("latin-1")), (b"set-cookie", cookie.encode("latin-1")),
                    (b"cache-control", b"no-store"),
                ]})
                return await send({"type": "http.response.body", "body": b""})
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        html = "text/html" in headers.get("accept", "") and not scope["path"].startswith("/api/")
        body = DENIED.encode() if html else b'{"detail":"Not authorised: the server\'s token is needed"}'
        await send({"type": "http.response.start", "status": 403, "headers": [
            (b"content-type", b"text/html; charset=utf-8" if html else b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]})
        await send({"type": "http.response.body", "body": body})
