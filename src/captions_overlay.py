# src/captions_overlay.py
from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import tkinter as tk

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

DEFAULT_PORT = 37777

MONITORINFOF_PRIMARY = 1


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


@dataclass
class MonitorInfo:
    index: int
    device: str
    left: int
    top: int
    right: int
    bottom: int
    work_left: int
    work_top: int
    work_right: int
    work_bottom: int
    is_primary: bool

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def work_width(self) -> int:
        return self.work_right - self.work_left

    @property
    def work_height(self) -> int:
        return self.work_bottom - self.work_top


def list_windows_monitors() -> list[MonitorInfo]:
    if not hasattr(ctypes, "windll"):
        return []

    user32 = ctypes.windll.user32
    raw: list[tuple[str, int, int, int, int, int, int, int, int, bool]] = []

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )

    def _callback(hmonitor, hdc, lprc, lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        ok = user32.GetMonitorInfoW(hmonitor, ctypes.byref(info))
        if ok:
            raw.append(
                (
                    info.szDevice,
                    info.rcMonitor.left,
                    info.rcMonitor.top,
                    info.rcMonitor.right,
                    info.rcMonitor.bottom,
                    info.rcWork.left,
                    info.rcWork.top,
                    info.rcWork.right,
                    info.rcWork.bottom,
                    bool(info.dwFlags & MONITORINFOF_PRIMARY),
                )
            )
        return True

    user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(_callback), 0)

    raw.sort(key=lambda m: (m[1], m[2]))  # left-to-right, then top-to-bottom

    return [
        MonitorInfo(
            index=i,
            device=item[0],
            left=item[1],
            top=item[2],
            right=item[3],
            bottom=item[4],
            work_left=item[5],
            work_top=item[6],
            work_right=item[7],
            work_bottom=item[8],
            is_primary=item[9],
        )
        for i, item in enumerate(raw)
    ]

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
        opacity: float | None = None,
        host: str = "127.0.0.1",
        monitor_index: int | None = None,
        prefer_non_primary: bool = False,
        x_offset: int = 0,
        y_offset: int = 0,
        bottom_margin: int = 80,
    ) -> None:
        self.width = width
        self.height = height
        self.monitor_index = monitor_index
        self.prefer_non_primary = prefer_non_primary
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.bottom_margin = bottom_margin

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

        self._place_initial_window()
        self.root.after(1500, self._place_initial_window)  # retry in case VDM finishes a bit later
        self.root.after(4000, self._place_initial_window)  # second retry

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

    def _select_monitor(self) -> MonitorInfo | None:
        monitors = list_windows_monitors()
        if not monitors:
            return None

        if self.monitor_index is not None:
            for mon in monitors:
                if mon.index == self.monitor_index:
                    return mon

        if self.prefer_non_primary:
            for mon in monitors:
                if not mon.is_primary:
                    return mon

        for mon in monitors:
            if mon.is_primary:
                return mon

        return monitors[0]

    def _place_initial_window(self) -> None:
        mon = self._select_monitor()

        if mon is None:
            # fallback to original behavior
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            x = max(0, (sw - self.width) // 2) + self.x_offset
            y = max(0, sh - self.height - self.bottom_margin) + self.y_offset
        else:
            x = mon.work_left + max(0, (mon.work_width - self.width) // 2) + self.x_offset
            y = mon.work_bottom - self.height - self.bottom_margin + self.y_offset

        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self.root.update_idletasks()
        self.root.lift()

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

    ap.add_argument("--monitor-index", type=int, default=None, help="Windows monitor index from --list-monitors.")
    ap.add_argument("--prefer-non-primary", action="store_true", help="Place on first non-primary monitor.")
    ap.add_argument("--x-offset", type=int, default=0)
    ap.add_argument("--y-offset", type=int, default=0)
    ap.add_argument("--bottom-margin", type=int, default=80)
    ap.add_argument("--list-monitors", action="store_true")

    args = ap.parse_args()

    if args.list_monitors:
        monitors = list_windows_monitors()
        if not monitors:
            print("No Windows monitor info available.")
        else:
            for m in monitors:
                print(
                    f"[{m.index}] {m.device} "
                    f"monitor=({m.left},{m.top})-({m.right},{m.bottom}) "
                    f"work=({m.work_left},{m.work_top})-({m.work_right},{m.work_bottom}) "
                    f"primary={m.is_primary}"
                )
        return

    CaptionOverlayApp(
        port=args.port,
        font_size=args.font_size,
        width=args.width,
        height=args.height,
        opacity=args.opacity,
        host=args.host,
        monitor_index=args.monitor_index,
        prefer_non_primary=args.prefer_non_primary,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        bottom_margin=args.bottom_margin,
    ).run()


if __name__ == "__main__":
    main()