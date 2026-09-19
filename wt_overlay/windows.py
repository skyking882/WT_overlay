"""Small Win32 adapter for global hotkeys and the game's visible client area.

Only queries window metadata; no hooks into the game or third-party Win32 package.
"""

from dataclasses import dataclass
import ctypes
from ctypes import wintypes as w
import sys
import time


@dataclass(frozen=True)
class GameWindow:
    foreground: bool
    minimized: bool
    rect: tuple[int, int, int, int]
    monitor_origin: tuple[int, int]
    monitor_name: str


class MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", w.DWORD), ("rcMonitor", w.RECT),
                ("rcWork", w.RECT), ("dwFlags", w.DWORD), ("szDevice", w.WCHAR * 32)]


class WindowsDesktop:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("WindowsDesktop requires Windows")
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._enum_type = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        signatures = {
            "RegisterHotKey": ([w.HWND, ctypes.c_int, w.UINT, w.UINT], w.BOOL),
            "UnregisterHotKey": ([w.HWND, ctypes.c_int], w.BOOL),
            "GetForegroundWindow": ([], w.HWND),
            "IsWindow": ([w.HWND], w.BOOL),
            "IsWindowVisible": ([w.HWND], w.BOOL),
            "IsIconic": ([w.HWND], w.BOOL),
            "EnumWindows": ([self._enum_type, w.LPARAM], w.BOOL),
            "GetWindowTextW": ([w.HWND, w.LPWSTR, ctypes.c_int], ctypes.c_int),
            "GetClassNameW": ([w.HWND, w.LPWSTR, ctypes.c_int], ctypes.c_int),
            "GetWindowThreadProcessId": ([w.HWND, ctypes.POINTER(w.DWORD)], w.DWORD),
            "GetClientRect": ([w.HWND, ctypes.POINTER(w.RECT)], w.BOOL),
            "ClientToScreen": ([w.HWND, ctypes.POINTER(w.POINT)], w.BOOL),
            "MonitorFromWindow": ([w.HWND, w.DWORD], w.HANDLE),
            "GetMonitorInfoW": ([w.HANDLE, ctypes.POINTER(MonitorInfo)], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.user, name)
            function.argtypes, function.restype = args, result
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]
        self.kernel.QueryFullProcessImageNameW.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self._registered: set[int] = set()
        self._hwnd = None
        self._next_search = 0.0

    def register_hotkey(self, identifier: int, letter: str) -> bool:
        ok = bool(self.user.RegisterHotKey(None, identifier, 0x4003, ord(letter)))
        if ok:
            self._registered.add(identifier)
        return ok

    def close(self):
        for identifier in self._registered:
            self.user.UnregisterHotKey(None, identifier)
        self._registered.clear()

    def _is_game(self, hwnd) -> bool:
        title, window_class = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
        self.user.GetWindowTextW(hwnd, title, len(title))
        self.user.GetClassNameW(hwnd, window_class, len(window_class))
        # Restrict candidates before opening a process query handle.
        caption = title.value.casefold().replace(" ", "")
        if not (window_class.value.startswith("Dagor") or caption in ("warthunder", "战争雷霆")):
            return False
        pid = w.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process = self.kernel.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if process:
            try:
                path, length = ctypes.create_unicode_buffer(32768), w.DWORD(32768)
                if self.kernel.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(length)):
                    return path.value.replace("/", "\\").rsplit("\\", 1)[-1].casefold() == "aces.exe"
            finally:
                self.kernel.CloseHandle(process)
        return window_class.value.startswith("Dagor") and caption in ("warthunder", "战争雷霆")

    def game_window(self) -> GameWindow | None:
        if self._hwnd and not self.user.IsWindow(self._hwnd):
            self._hwnd = None
        if not self._hwnd and time.monotonic() >= self._next_search:
            self._next_search = time.monotonic() + 2.0

            @self._enum_type
            def visit(hwnd, _):
                if self.user.IsWindowVisible(hwnd) and self._is_game(hwnd):
                    self._hwnd = hwnd
                    return False
                return True

            self.user.EnumWindows(visit, 0)
        if not self._hwnd:
            return None
        hwnd = self._hwnd
        rect, origin = w.RECT(), w.POINT()
        monitor = MonitorInfo()
        monitor.cbSize = ctypes.sizeof(MonitorInfo)
        if not (self.user.GetClientRect(hwnd, ctypes.byref(rect))
                and self.user.ClientToScreen(hwnd, ctypes.byref(origin))
                and self.user.GetMonitorInfoW(self.user.MonitorFromWindow(hwnd, 2), ctypes.byref(monitor))):
            return None
        return GameWindow(
            self.user.GetForegroundWindow() == hwnd,
            bool(self.user.IsIconic(hwnd) or not self.user.IsWindowVisible(hwnd)),
            (origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top),
            (monitor.rcMonitor.left, monitor.rcMonitor.top), monitor.szDevice)
