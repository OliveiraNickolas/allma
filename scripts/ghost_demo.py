#!/usr/bin/env python3
"""Watch the loading ghost animation on its own — Ctrl+C to exit."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.ghost_art import render_rows, ROWS  # noqa: E402


def main():
    tick = 0
    print("\033[?25l", end="")          # hide cursor
    try:
        print("\n" * ROWS, end="")
        while True:
            sys.stdout.write(f"\033[{ROWS}F")
            for line in render_rows(tick, colored=True):
                sys.stdout.write("\033[2K" + line + "\n")
            sys.stdout.flush()
            time.sleep(0.08)
            tick += 1
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h", end="")      # cursor back


if __name__ == "__main__":
    main()
