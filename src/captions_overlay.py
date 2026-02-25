# src/captions_overlay.py
from __future__ import annotations

import json
import queue
import socket
import threading
import tkinter as tk

DEFAULT_PORT = 37777


class CaptionOverlayApp:
    def __init__(
        self,
        port: int = DEFAULT_PORT,
        font_size: int = 36,
        width: int = 1400,
        height: int = 140,
        opacity: float | None = None,  # e.g. 0.90 or None
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Commpanion Captions")

        # State
        self._framed = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._win_start_x = 0
        self._win_start_y = 0

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
            wraplength=width - 60,
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

        # Message queue from UDP thread -> Tk thread
        self.msg_q: queue.Queue[dict] = queue.Queue()
        self._start_udp_listener(port)

        # Poll queue
        self._poll()

        # Initial position: bottom center-ish on *primary* screen (drag to AR display after)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, (sw - width) // 2)
        y = max(0, sh - height - 80)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Keep wraplength consistent if you later change geometry
        self.root.bind("<Configure>", self._on_configure)

    def _on_configure(self, event) -> None:
        # event.width includes padding; keep wraplength a bit smaller than window width
        try:
            self.label.configure(wraplength=max(200, event.width - 60))
        except Exception:
            pass

    def _start_move(self, event) -> None:
        # Store mouse position in SCREEN coords
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        # Store window position at start of drag
        self._win_start_x = self.root.winfo_x()
        self._win_start_y = self.root.winfo_y()

    def _do_move(self, event) -> None:
        dx = event.x_root - self._drag_start_x
        dy = event.y_root - self._drag_start_y
        self.root.geometry(f"+{self._win_start_x + dx}+{self._win_start_y + dy}")

    def _toggle_frame(self, event=None) -> None:
        """
        F2: toggle framed window (title bar) vs borderless overlay.
        Useful if Windows is being weird about dragging a borderless window.
        """
        self._framed = not self._framed
        # When framed=True, we want overrideredirect(False)
        self.root.overrideredirect(not self._framed)
        self.root.attributes("-topmost", True)
        self.root.update_idletasks()

    def _start_udp_listener(self, port: int) -> None:
        def worker():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", port))
            while True:
                try:
                    data, _ = sock.recvfrom(65535)
                    payload = json.loads(data.decode("utf-8", errors="ignore"))
                    self.msg_q.put(payload)
                except Exception:
                    # Keep running even if a message is malformed
                    continue

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _poll(self) -> None:
        try:
            while True:
                payload = self.msg_q.get_nowait()
                text = str(payload.get("text", ""))
                clear = bool(payload.get("clear", False))
                self.label.config(text="" if clear else text)
        except queue.Empty:
            pass

        self.root.after(30, self._poll)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    CaptionOverlayApp().run()