from __future__ import annotations
import json
import socket
from typing import Optional

class CaptionClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 37777) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def set_text(self, text: str) -> None:
        self._send({"text": text, "clear": False})

    def clear(self) -> None:
        self._send({"text": "", "clear": True})

    def _send(self, payload: dict) -> None:
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.sock.sendto(data, self.addr)
        except Exception:
            # Never let captions break the pipeline.
            pass