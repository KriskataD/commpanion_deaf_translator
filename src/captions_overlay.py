# src/captions_overlay.py
from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import tkinter as tk

DEFAULT_PORT = 37777


class CaptionOverlayApp:
    """
    Simple always-on-top caption overlay window.

    Receives UDP JSON messages on 127.0.0.1:<port> with payload:
      {"text": "...", "clear": false, "ttl_ms": 9000}

    - "clear": true clears immediately
    - "ttl_ms": optional auto-clear timer for the displayed text
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        font_size: int = 36,
        width: int = 1400,
        height: int = 140,
        opacity: float | None = None,  # e.g. 0.90 or None
        host: str = "127.0.0.1",
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Commpanion Captions")

        # State for moving / framing / auto-clear
        self._framed = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._win_start_x = 0
        self._win_start_y = 0
        self._clear_job: str | None = None

        # Borderless + always-on-top
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Optional transparency
        if opacity is not None:
            try:
                self.root.attributes("-alpha", float(opacity))
            except Exception:
                pass

        # Styling
        self.root.configure(bg="black")

        self.label = tk.Label(
            self.root,
            text="",
            fg="white",
            bg="black",
            font=("Segoe UI", font_size, "bold"),
            justify="center",
            anchor="center",
            wraplength=max(200, width - 60),
        )
        self.label.pack(fill="both", expand=True, padx=24, pady=18)

        # --- Dragging: bind to BOTH root + label (works even if no text)
        self.root.bind("<ButtonPress-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._do_move)
        self.label.bind("<ButtonPress-1>", self._start_move)
        self.label.bind("<B1-Motion>", self._do_move)

        # Hotkeys
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<F2>", self._toggle_frame)
        self.root.bind("<F4>", lambda e: self._set_text("", ttl_ms=None))  # quick clear

        # Message queue from UDP thread -> Tk thread
        self.msg_q: queue.Queue[dict] = queue.Queue()
        self._start_udp_listener(host=host, port=port)

        # Initial position: bottom center-ish on *primary* screen (drag to AR display after)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, (sw - width) // 2)
        y = max(0, sh - height - 80)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Keep wraplength consistent if you later change geometry
        self.root.bind("<Configure>", self._on_configure)

        # Poll queue
        self._poll()

    def _on_configure(self, event) -> None:
        try:
            self.label.configure(wraplength=max(200, int(event.width) - 60))
        except Exception:
            pass

    # -----------------------
    # Dragging
    # -----------------------
    def _start_move(self, event) -> None:
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        self._win_start_x = self.root.winfo_x()
        self._win_start_y = self.root.winfo_y()

    def _do_move(self, event) -> None:
        dx = event.x_root - self._drag_start_x
        dy = event.y_root - self._drag_start_y
        self.root.geometry(f"+{self._win_start_x + dx}+{self._win_start_y + dy}")

    # -----------------------
    # Window mode
    # -----------------------
    def _toggle_frame(self, event=None) -> None:
        """
        F2: toggle framed window (title bar) vs borderless overlay.
        Useful if Windows is being weird about dragging a borderless window.
        """
        self._framed = not self._framed
        self.root.overrideredirect(not self._framed)
        self.root.attributes("-topmost", True)
        self.root.update_idletasks()

    # -----------------------
    # Text + TTL
    # -----------------------
    def _cancel_pending_clear(self) -> None:
        if self._clear_job is None:
            return
        try:
            self.root.after_cancel(self._clear_job)
        except Exception:
            pass
        finally:
            self._clear_job = None

    def _clear_if_unchanged(self, expected_text: str) -> None:
        # Only clear if nothing newer has replaced it.
        if self.label.cget("text") == expected_text:
            self.label.config(text="")
        self._clear_job = None

    def _set_text(self, text: str, ttl_ms: int | None) -> None:
        self.label.config(text=text)

        # cancel previous auto-clear, then schedule a new one if requested
        self._cancel_pending_clear()
        if ttl_ms is not None and text:
            try:
                ttl_ms_i = int(ttl_ms)
                if ttl_ms_i > 0:
                    self._clear_job = self.root.after(
                        ttl_ms_i, lambda t=text: self._clear_if_unchanged(t)
                    )
            except Exception:
                self._clear_job = None

    # -----------------------
    # UDP Listener
    # -----------------------
    def _start_udp_listener(self, host: str, port: int) -> None:
        def worker():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((host, int(port)))
            while True:
                try:
                    data, _ = sock.recvfrom(65535)
                    payload = json.loads(data.decode("utf-8", errors="ignore"))
                    if isinstance(payload, dict):
                        self.msg_q.put(payload)
                except Exception:
                    # Keep running even if a message is malformed
                    continue

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # -----------------------
    # Tk polling loop
    # -----------------------
    def _poll(self) -> None:
        try:
            while True:
                payload = self.msg_q.get_nowait()
                text = str(payload.get("text", ""))
                clear = bool(payload.get("clear", False))
                ttl_ms = payload.get("ttl_ms", None)

                if clear:
                    self._set_text("", ttl_ms=None)
                else:
                    self._set_text(text, ttl_ms=ttl_ms)
        except queue.Empty:
            pass

        self.root.after(30, self._poll)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Commpanion captions overlay (UDP -> Tk window).")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--font-size", type=int, default=36)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=140)
    ap.add_argument("--opacity", type=float, default=None, help="e.g. 0.9 (optional)")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    args = ap.parse_args()

    CaptionOverlayApp(
        port=args.port,
        font_size=args.font_size,
        width=args.width,
        height=args.height,
        opacity=args.opacity,
        host=args.host,
    ).run()


if __name__ == "__main__":
    main()