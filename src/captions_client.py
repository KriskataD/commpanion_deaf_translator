from __future__ import annotations

import json
import socket
import textwrap
from dataclasses import dataclass

DEFAULT_PORT = 37777


@dataclass
class CaptionMsg:
    text: str = ""
    clear: bool = False
    ttl_ms: int | None = None  # auto-clear after this many ms (overlay supports it below)


class CaptionsClient:
    """
    UDP client that sends caption updates to captions_overlay.py
    """
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _format(self, text: str, *, max_chars: int = 220, max_lines: int = 3) -> str:
        text = " ".join(text.split())  # normalize whitespace
        if not text:
            return ""
        lines = textwrap.wrap(text, width=max(1, max_chars // max(1, max_lines)))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            # small ellipsis hint
            if lines[-1] and not lines[-1].endswith("…"):
                lines[-1] = lines[-1][: max(0, len(lines[-1]) - 1)] + "…"
        return "\n".join(lines)

    def send(self, text: str = "", *, clear: bool = False, ttl_ms: int | None = 9000) -> None:
        msg = CaptionMsg(
            text="" if clear else self._format(text),
            clear=bool(clear),
            ttl_ms=None if clear else ttl_ms,
        )
        payload = json.dumps(msg.__dict__, ensure_ascii=False).encode("utf-8")
        # UDP send
        self.sock.sendto(payload, self.addr)

    def clear(self) -> None:
        self.send(clear=True)