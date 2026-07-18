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


# ── palette + geometry shared by both variants ───────────────────────────────
# `cream` is the classic ghost: body in warm cream, eyes are hollow (they
# show the terminal background). `brown` swaps the body for a warm C64
# keycap tone and lights up the eyes as bright white squares — feels more
# like a physical mascot in decorative slots.
BODY_CREAM_RGB = (240, 232, 208)   # #f0e8d0
BODY_BROWN_RGB = (83, 71, 66)      # #534742
EYE_WHITE_RGB  = (255, 255, 255)

BODY_CREAM_HEX = "#f0e8d0"
BODY_BROWN_HEX = "#534742"
EYE_WHITE_HEX  = "#ffffff"

# Eye pixels in BIG_GHOST — the two spaces on row 2 that read as eyes.
_EYE_POSITIONS = {(2, 6), (2, 9)}


def _variant_palette(variant: str) -> tuple[tuple, bool]:
    """Return (body_rgb, fill_eyes) for a variant name."""
    if variant == "brown":
        return BODY_BROWN_RGB, True
    return BODY_CREAM_RGB, False


def big_ghost_lines(variant: str = "cream", colored=None) -> list[str]:
    """The big mascot as printable ANSI lines.

    variant='cream'  — solid cream body, eyes are ' ' (show terminal bg).
    variant='brown'  — warm-brown body, eyes are white-filled squares.
    """
    if colored is None:
        colored = sys.stdout.isatty()
    if not colored:
        return list(BIG_GHOST)

    body_rgb, fill_eyes = _variant_palette(variant)
    body_fg = _fg(body_rgb)
    eye_bg = _bg(EYE_WHITE_RGB)
    out = []
    for r_ix, row in enumerate(BIG_GHOST):
        line = []
        for c_ix, ch in enumerate(row):
            is_eye = (r_ix, c_ix) in _EYE_POSITIONS
            if ch == " ":
                if fill_eyes and is_eye:
                    line.append(f"{eye_bg} {_RESET}")
                else:
                    line.append(" ")
            else:
                line.append(f"{body_fg}{ch}{_RESET}")
        out.append("".join(line))
    return out


def big_ghost_rich(variant: str = "cream"):
    """Same mascot as a `rich.text.Text` for the Rich-rendered surfaces
    (show_banner, allma top). Import Rich lazily so the module stays
    stdlib-only for the many call sites that don't need it."""
    from rich.text import Text
    body_hex = BODY_BROWN_HEX if variant == "brown" else BODY_CREAM_HEX
    fill_eyes = variant == "brown"
    t = Text(justify="center")
    for r_ix, row in enumerate(BIG_GHOST):
        for c_ix, ch in enumerate(row):
            is_eye = (r_ix, c_ix) in _EYE_POSITIONS
            if ch == " ":
                if fill_eyes and is_eye:
                    t.append(" ", style=f"on {EYE_WHITE_HEX}")
                else:
                    t.append(" ")
            else:
                t.append(ch, style=f"bold {body_hex}")
        if r_ix < len(BIG_GHOST) - 1:
            t.append("\n")
    return t


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
