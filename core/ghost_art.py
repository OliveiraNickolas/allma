"""Shared loading animation: a little ghost bobbing up and down, leaving a
nyan-cat-style undulating rainbow trail in the allma palette, plus stars.

Single source of truth for both spinners (allma_cli and core.loader).
`render_rows(tick)` returns FOUR ANSI-styled strings, each exactly WIN
visible columns wide. Colors are skipped when stdout isn't a TTY.

How the wiggle works: the canvas is 4 rows. The trail is a continuous
six-stripe ribbon that shifts down by HALF a cell in an asymmetric
rhythm — up segments 2 columns wide, down segments 3 — scrolling with
time. The ghost is 3 rows tall (dome / eyes / hem) and rides the same
wave at its own column, bobbing between rows 0-2 and 1-3.
"""
import random
import sys

WIN = 36          # visible columns of the animation window
ROWS = 4          # canvas height
_GHOST_X = 24     # ghost anchor (trail fills everything to its left)

# ── allma palette rainbow (6 stripes, nyan-style continuous ribbon) ──────────
_STRIPES = [
    (229, 37, 41),    # red     #e52529
    (247, 148, 29),   # orange  #f7941d
    (255, 215, 95),   # yellow  #ffd75f
    (67, 176, 71),    # green   #43b047
    (0, 157, 220),    # blue    #009ddc
    (0, 136, 136),    # teal    #008888
]
_GHOST_RGB = (240, 232, 208)   # cream — reads white on dark terminals
_STAR_RGB = (168, 152, 120)

_RESET = "\033[0m"


def _fg(rgb):
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _bg(rgb):
    return f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


# ── ghost sprite: 4 wide × 3 rows (cell aspect makes it slightly tall) ───────
_GHOST_DOME = "▄███▄"
_GHOST_FACE = "███▙▙"            # tiny notch-eyes, pupils to the RIGHT
_GHOST_HEMS = ["█▀ ▀█","█▀█▀█"]  # scalloped hem, fluttering
_HEM_SPEED = 5


def _wave_pos(tick: int, x: int) -> int:
    """Trail ribbon state at column x: 0 = up, 1 = shifted down HALF a cell.
    Asymmetric rhythm: up segments are 2 columns wide, down segments 3
    (period of 5), scrolling with time so the wave travels."""
    return 0 if ((tick // 5) + x) % 5 < 2 else 1


class _Stars:
    """Sparse twinkles in the empty space: born, shine, fade, gone."""
    _GLYPHS = ("·", "✧", "✦", "✧", "·")

    def __init__(self):
        self._rng = random.Random()
        self._stars = []          # [x, row, age]

    def tick(self, lo: int, hi: int):
        self._stars = [[x, r, a + 1] for x, r, a in self._stars
                       if a + 1 < len(self._GLYPHS)]
        if hi - lo > 2 and len(self._stars) < 5 and self._rng.random() < 0.25:
            self._stars.append(
                [self._rng.randrange(lo, hi), self._rng.randrange(0, ROWS), 0])

    def blit(self, rows, colored: bool):
        for x, r, age in self._stars:
            if rows[r][x] != " ":
                continue
            g = self._GLYPHS[age]
            rows[r][x] = (_fg(_STAR_RGB) + g + _RESET) if colored else g


_stars = _Stars()


def render_rows(tick: int, colored=None):
    """FOUR WIN-column rows: undulating rainbow trail + bobbing ghost."""
    if colored is None:
        colored = sys.stdout.isatty()

    gx = _GHOST_X
    rows = [[" "] * WIN for _ in range(ROWS)]

    # ── trail: continuous 6-stripe ribbon, undulating by HALF a cell ────
    # up state:   rows 0-2 hold the six stripes as ▀(top,bottom) pairs.
    # down state: everything slides half a cell — ▄ paints the first
    # half-stripe at the top edge, ▀ without bg paints the last at row 3.
    s = _STRIPES
    for x in range(gx - 1):
        if _wave_pos(tick, x) == 0:   # up
            if colored:
                rows[0][x] = f"{_fg(s[0])}{_bg(s[1])}▀{_RESET}"
                rows[1][x] = f"{_fg(s[2])}{_bg(s[3])}▀{_RESET}"
                rows[2][x] = f"{_fg(s[4])}{_bg(s[5])}▀{_RESET}"
            else:
                rows[0][x] = rows[1][x] = rows[2][x] = "█"
        else:                          # down (half-cell shift)
            if colored:
                rows[0][x] = f"{_fg(s[0])}▄{_RESET}"
                rows[1][x] = f"{_fg(s[1])}{_bg(s[2])}▀{_RESET}"
                rows[2][x] = f"{_fg(s[3])}{_bg(s[4])}▀{_RESET}"
                rows[3][x] = f"{_fg(s[5])}▀{_RESET}"
            else:
                rows[0][x] = "▄"
                rows[1][x] = rows[2][x] = "█"
                rows[3][x] = "▀"

    # ── ghost rides the wave at its own column ──────────────────────────
    bob = _wave_pos(tick, gx)                     # rides the ribbon's shift
    hem = _GHOST_HEMS[(tick // _HEM_SPEED) % 2]
    paint = _fg(_GHOST_RGB) if colored else ""
    reset = _RESET if colored else ""
    for r, sprite in ((bob, _GHOST_DOME), (bob + 1, _GHOST_FACE), (bob + 2, hem)):
        for k, ch in enumerate(sprite):
            p = gx + k
            if p < WIN:
                rows[r][p] = f"{paint}{ch}{reset}" if ch != " " else " "

    # ── stars in the leftover empty space ────────────────────────────────
    _stars.tick(gx + 6, WIN)
    _stars.blit(rows, colored)

    return ["".join(r) for r in rows]
