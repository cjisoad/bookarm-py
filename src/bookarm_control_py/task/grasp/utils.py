"""Small utilities shared by grasp task modules."""

from __future__ import annotations

import sys
import time
from typing import Sequence

import numpy as np


def format_array(values: np.ndarray | Sequence[float]) -> str:
    return np.array2string(np.asarray(values), precision=6, suppress_small=True)


def array_to_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def wait(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def flush_pending_terminal_input() -> None:
    try:
        if sys.platform.startswith("win"):
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getwch()
            return

        import select

        if not sys.stdin.isatty():
            return
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                break
            sys.stdin.read(1)
    except (ImportError, OSError):
        return


def ask_yes_no(prompt: str, *, default: bool = False) -> bool:
    flush_pending_terminal_input()
    suffix = " [Y/n]: " if default else " [y/N]: "
    full_prompt = prompt.rstrip()
    display_prompt = full_prompt + " " if full_prompt.endswith(("[y/N]:", "[Y/n]:")) else full_prompt + suffix

    if not sys.stdin.isatty():
        print(f"{display_prompt}{'yes' if default else 'no'}")
        print("当前不是交互式终端，已使用默认选择。")
        return default

    if sys.platform.startswith("win"):
        try:
            import msvcrt

            print(display_prompt, end="", flush=True)
            while True:
                key = msvcrt.getwch().lower()
                if key in {"\x00", "\xe0"}:
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                if key in {"\r", "\n"}:
                    print()
                    return default
                if key in {"y", "n"}:
                    print(key)
                    return key == "y"
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == "\x1b":
                    print()
                    return False
        except OSError:
            print()

    try:
        answer = input(display_prompt).strip().lower()
    except EOFError:
        print()
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}
