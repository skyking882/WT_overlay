"""Console-free Windows entry point with visible startup errors."""

from contextlib import redirect_stderr, redirect_stdout
import ctypes
import os
from pathlib import Path
import sys
import traceback


def show_error(message: str) -> None:
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.MessageBoxW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    user.MessageBoxW.restype = ctypes.c_int
    user.MessageBoxW(None, message, "WT Energy · 启动失败", 0x10 | 0x10000)


def main(argv: list[str] | None = None) -> int:
    try:
        local = Path(os.environ.get("LOCALAPPDATA") or Path.home()/"AppData"/"Local")
        log_path = local/"WT Energy"/"startup.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    from .__main__ import main as run
                    code = run(argv)
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
                    if isinstance(exc.code, str):
                        print(exc.code, file=sys.stderr)
                except Exception:
                    traceback.print_exc()
                    code = 2
        if code:
            detail = log_path.read_text(encoding="utf-8")[-2500:].strip()
            show_error(f"{detail}\n\n错误记录：{log_path}")
        return code
    except OSError as exc:
        show_error(f"无法启动 WT Energy：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
