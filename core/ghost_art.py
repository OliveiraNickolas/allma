"""Shared loading animation: a little ghost bobbing up and down, leaving a
nyan-cat-style undulating rainbow trail in the allma palette, plus stars.

Single source of truth for both spinners (allma_cli and core.loader).
`render_rows(tick)` returns FOUR ANSI-styled strings, each exactly WIN
visible columns wide. Colors are skipped when stdout isn't a TTY.

How the wiggle works: the canvas is 4 rows. The trail is a 2-row band
whose top row follows a triangle wave (0 → 1 → 2 → 1) along x, so it
snakes as it scrolls. The ghost is 3 rows tall (dome / eyes / hem) and
rides the same wave at its own column, bobbing between rows 0-2 and 1-3.
"""
import random
import sys

WIN = 36          # visible columns of the animation window
ROWS = 4          # canvas height
_GHOST_X = 24     # ghost anchor (trail fills everything to its left)

# ── allma palette rainbow (4 stripes → 2 rows via half-blocks) ───────────────
_STRIPES = [
    (229, 37, 41),    # red     #e52529
    (255, 215, 95),   # yellow  #ffd75f
    (67, 176, 71),    # green   #43b047
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
_GHOST_DOME = "▄██▄"
_GHOST_FACE = "█▟▟█"            # two tiny notch-eyes
_GHOST_HEMS = ["█▀█▀", "▀█▀█"]  # scalloped hem, fluttering
_HEM_SPEED = 7


def _wave_pos(tick: int, x: int) -> int:
    """Top row of the 2-row trail band at column x: triangle wave 0→1→2→1."""
    return (0, 1, 2, 1)[((tick // 4) + (x // 3)) % 4]


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

    # ── trail: 2-row band snaking through the 4-row canvas ──────────────
    for x in range(gx - 1):
        top = _wave_pos(tick, x)
        if colored:
            rows[top][x] = f"{_fg(_STRIPES[0])}{_bg(_STRIPES[1])}▀{_RESET}"
            rows[top + 1][x] = f"{_fg(_STRIPES[2])}{_bg(_STRIPES[3])}▀{_RESET}"
        else:
            rows[top][x] = "▀"
            rows[top + 1][x] = "▄"

    # ── ghost rides the wave at its own column ──────────────────────────
    bob = 1 if _wave_pos(tick, gx) == 2 else 0    # mostly high, dips on the low crest
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
