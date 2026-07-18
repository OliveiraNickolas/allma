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

WIN = 40          # visible columns of the animation window
ROWS = 8          # canvas height — fits the full 7-row mascot + bob
_GHOST_X = 20     # ghost anchor (trail fills everything to its left)

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


# ── the mascot: solid ghost with hollow square eyes ──────────────────────────
# Canonical big version (README, `allma status`, TUI idle panel). The tiny
# spinner sprite below is the same identity scaled to 3 rows.
BIG_GHOST = [
    "  ▄██████▄",
    " ██████████",
    " █████ ██ █",
    " ██████████",
    " ▀█ ▀██▀ █▀",
    "▄   ▄  ▄   "
]


# Hand-drawn 3-row version for tight spots (TUI topbar).
MINI_GHOST = [
    "▄████▄",
    "██▄█▄█",
    "█▀██▀█",
]


def big_ghost_lines(colored=None) -> list[str]:
    """The big mascot as printable lines (cream on dark TTYs, plain otherwise)."""
    if colored is None:
        colored = sys.stdout.isatty()
    if not colored:
        return list(BIG_GHOST)
    return [f"{_fg(_GHOST_RGB)}{line}{_RESET}" for line in BIG_GHOST]


# ── ghost sprite: the full BIG_GHOST rides the trail (11 wide × 7 rows) ──────
_GHOST_SPRITE = [line.ljust(11) for line in BIG_GHOST]


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
    """ROWS WIN-column rows: undulating rainbow trail + bobbing ghost."""
    if colored is None:
        colored = sys.stdout.isatty()

    gx = _GHOST_X
    rows = [[" "] * WIN for _ in range(ROWS)]

    # ── trail: continuous 6-stripe ribbon (one row per stripe), undulating
    # by HALF a cell. up state: rows 1-6 are full stripe cells. down state:
    # everything slides half a cell — ▄ opens the ribbon at row 1, fg/bg ▀
    # pairs carry the seams, a bare ▀ closes it at row 7.
    s = _STRIPES
    for x in range(gx - 1):
        if _wave_pos(tick, x) == 0:   # up
            for i in range(6):
                rows[1 + i][x] = f"{_fg(s[i])}█{_RESET}" if colored else "█"
        else:                          # down (half-cell shift)
            if colored:
                rows[1][x] = f"{_fg(s[0])}▄{_RESET}"
                for i in range(5):
                    rows[2 + i][x] = f"{_fg(s[i])}{_bg(s[i + 1])}▀{_RESET}"
                rows[7][x] = f"{_fg(s[5])}▀{_RESET}"
            else:
                rows[1][x] = "▄"
                for i in range(5):
                    rows[2 + i][x] = "█"
                rows[7][x] = "▀"

    # ── ghost rides the wave at its own column ──────────────────────────
    bob = _wave_pos(tick, gx)                     # rides the ribbon's shift
    paint = _fg(_GHOST_RGB) if colored else ""
    reset = _RESET if colored else ""
    # sprite is 6 rows on an 8-row canvas — +1 keeps it centered on the ribbon
    for r, line in enumerate(_GHOST_SPRITE):
        for k, ch in enumerate(line):
            p = gx + k
            if p < WIN and ch != " ":
                rows[bob + 1 + r][p] = f"{paint}{ch}{reset}"

    # ── stars in the leftover empty space ────────────────────────────────
    _stars.tick(gx + 12, WIN)
    _stars.blit(rows, colored)

    return ["".join(r) for r in rows]
