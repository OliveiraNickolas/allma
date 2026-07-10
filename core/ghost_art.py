"""Shared loading animation: a little ghost drifting right, leaving a
nyan-cat-style rainbow trail in the allma palette, with twinkling stars.

Single source of truth for both spinners (allma_cli and core.loader).
`render_rows(tick)` returns three ANSI-styled strings, each exactly WIN
visible columns wide. Colors are skipped when stdout isn't a TTY.
"""
import random
import sys

WIN = 36          # visible columns of the animation window
_GHOST_X = 22     # ghost anchor (trail fills everything to its left)

# ── allma palette rainbow (6 stripes → 3 rows via half-blocks) ───────────────
_STRIPES = [
    (229, 37, 41),    # red     #e52529
    (247, 148, 29),   # orange  #f7941d
    (255, 215, 95),   # yellow  #ffd75f
    (67, 176, 71),    # green   #43b047
    (0, 157, 220),    # blue    #009ddc
    (0, 120, 120),    # teal    #007878
]
_GHOST_RGB = (240, 232, 208)   # cream — reads white on dark terminals
_STAR_RGB = (232, 223, 200)

_RESET = "\033[0m"


def _fg(rgb):
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _bg(rgb):
    return f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


# ── ghost sprite (v4) ─────────────────────────────────────────────────────────
_DOME = " ▄██▄ "
_FACE = "███▟▟█"
_HEMS = ["█▀██▀█", "██▀█▀█"]
_SPEED = 7


class _Stars:
    """Sparse twinkles ahead of the ghost: born, shine, fade, gone."""
    _GLYPHS = ("·", "✧", "✦", "✧", "·")

    def __init__(self):
        self._rng = random.Random()
        self._stars = []          # [x, row, age]

    def tick(self, lo: int, hi: int):
        self._stars = [[x, r, a + 1] for x, r, a in self._stars
                       if a + 1 < len(self._GLYPHS)]
        if hi - lo > 2 and len(self._stars) < 4 and self._rng.random() < 0.25:
            self._stars.append(
                [self._rng.randrange(lo, hi), self._rng.randrange(0, 3), 0])

    def blit(self, rows, colored: bool):
        for x, r, age in self._stars:
            g = self._GLYPHS[age]
            rows[r][x] = (_fg(_STAR_RGB) + g + _RESET) if colored else g


_stars = _Stars()


def render_rows(tick: int, colored=None):
    """Three WIN-column rows: rainbow trail + ghost + twinkling stars."""
    if colored is None:
        colored = sys.stdout.isatty()

    sway = (0, 1, 1, 0)[(tick // (_SPEED * 3)) % 4]
    gx = _GHOST_X + sway
    rows = [[" "] * WIN for _ in range(3)]

    # rainbow trail: 2 stripes per row via half-blocks; the ▀/▄ + fg/bg swap
    # travels along x so the trail waves like the nyan cat's.
    for x in range(gx - 1):
        wave = ((tick // 4) + (x // 3)) % 2
        for r in range(3):
            top, bot = _STRIPES[2 * r], _STRIPES[2 * r + 1]
            if colored:
                if wave:
                    rows[r][x] = f"{_fg(bot)}{_bg(top)}▄{_RESET}"
                else:
                    rows[r][x] = f"{_fg(top)}{_bg(bot)}▀{_RESET}"
            else:
                rows[r][x] = "▀" if not wave else "▄"

    # stars ahead of the ghost
    _stars.tick(gx + 8, WIN)
    _stars.blit(rows, colored)

    # ghost on top
    hem = _HEMS[(tick // _SPEED) % 2]
    paint = (_fg(_GHOST_RGB)) if colored else ""
    reset = _RESET if colored else ""
    for r, sprite in enumerate((_DOME, _FACE, hem)):
        for k, ch in enumerate(sprite):
            p = gx + k
            if p < WIN:
                rows[r][p] = f"{paint}{ch}{reset}" if ch != " " else " "

    return ["".join(r) for r in rows]
