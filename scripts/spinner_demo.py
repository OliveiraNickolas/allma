#!/usr/bin/env python3
"""Quick demo of spinner style options — Ctrl+C to skip to next."""
import sys, time, threading, itertools

PHRASE = "Alpaca warming up the tensor before the next token."
DUR = 4.0  # seconds per demo


def run(fn, label):
    stop = threading.Event()
    t = threading.Thread(target=fn, args=(stop,), daemon=True)
    print(f"\n  ── Option {label} ──")
    t.start()
    try:
        time.sleep(DUR)
    except KeyboardInterrupt:
        pass
    stop.set()
    t.join()
    print()


# ── 1. Current (bouncing bar) ─────────────────────────────────────────────────
def opt1(stop):
    track = 12
    pos_seq = list(range(track)) + list(range(track - 2, 0, -1))
    peaks = ["▲", "△", "▲", "▲", "△", "▲", "△", "▲"]
    i = 0
    last = 0
    start = time.time()
    while not stop.is_set():
        pos = pos_seq[i % len(pos_seq)]
        bar = "░" * pos + "▓" + "░" * (track - pos - 1)
        pk = peaks[i % len(peaks)]
        line = f"  {pk} [{bar}]  {PHRASE}  {time.time()-start:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.083)
        i += 1
    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 2. Classic DOS marquee ────────────────────────────────────────────────────
def opt2(stop):
    arrows = ["»»»", "»» ", "»  ", "   ", "  «", " ««", "«««", " ««", "  «", "   ", "»  ", "»» "]
    i = 0
    last = 0
    while not stop.is_set():
        a = arrows[i % len(arrows)]
        line = f"  {a}  {PHRASE}"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.12)
        i += 1
    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 3. Rotating pipe (classic unix) ──────────────────────────────────────────
def opt3(stop):
    frames = ["|", "/", "─", "\\"]
    i = 0
    last = 0
    start = time.time()
    while not stop.is_set():
        f = frames[i % len(frames)]
        line = f"  [{f}]  {PHRASE}  {time.time()-start:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.1)
        i += 1
    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 4. Dot pulse (ellipsis breathing) ────────────────────────────────────────
def opt4(stop):
    states = ["   ", ".  ", ".. ", "...", " ..", "  .", "   "]
    i = 0
    last = 0
    start = time.time()
    while not stop.is_set():
        d = states[i % len(states)]
        line = f"  ▲ {d}  {PHRASE}  {time.time()-start:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.18)
        i += 1
    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 5. Block fill (left-to-right then clear) ─────────────────────────────────
def opt5(stop):
    width = 10
    i = 0
    last = 0
    start = time.time()
    while not stop.is_set():
        filled = i % (width * 2)
        if filled <= width:
            bar = "█" * filled + "▒" * (width - filled)
        else:
            n = width * 2 - filled
            bar = "▒" * n + " " * (width - n)
        line = f"  [{bar}]  {PHRASE}  {time.time()-start:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.1)
        i += 1
    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 6. Braille spinner + scrolling text ──────────────────────────────────────
def opt6(stop):
    frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    i = 0
    last = 0
    start = time.time()
    while not stop.is_set():
        f = frames[i % len(frames)]
        line = f"  {f}  {PHRASE}  {time.time()-start:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.083)
        i += 1
    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 7. Scrolling text marquee (retro LCD) ────────────────────────────────────
def opt7(stop):
    window = 38
    text = "  >>>  " + PHRASE + "  <<<  "
    i = 0
    last = 0
    while not stop.is_set():
        offset = i % len(text)
        chunk = (text * 2)[offset: offset + window]
        line = f"  ▲ [{chunk}]"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.07)
        i += 1
    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 8. Scrolling mountain range ───────────────────────────────────────────────
def opt8(stop):
    # Terrain made of block elements — different peak heights, natural valleys
    terrain = (
        "▁▁▁▂▃▄▅▆▇███▇▆▅▄▃▂▁▁"
        "▁▁▂▃▄▃▂▁▁"
        "▁▂▃▅▇████▇▅▃▂▁▁"
        "▁▁▁▂▄▆█████▆▄▂▁▁"
        "▂▃▄▅▄▃▂▁▁"
        "▁▂▄▆███▆▄▂▁▁"
        "▁▁▁▂▃▅▇██▇▅▃▂▁▁"
        "▁▂▃▄▅▆▅▄▃▂▁▁"
        "▁▁▂▄▇█████▇▄▂▁▁"
        "▁▁▂▃▄▃▂▁▁"
        "▁▁▂▄▆████▆▄▂▁"
    ) * 3  # repeat so scroll never runs out

    window = 32
    i = 0
    last = 0
    start = time.time()
    tick = 0

    while not stop.is_set():
        offset = i % len(terrain)
        view = (terrain * 2)[offset: offset + window]
        elapsed = time.time() - start
        line = f"  {view}  {PHRASE}  {elapsed:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.06)
        tick += 1
        # advance terrain every 2 ticks (~8 chars/sec) — feel of running
        if tick % 2 == 0:
            i += 1

    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 8b. Same but with phrase below and mountains on their own line ─────────────
def opt8b(stop):
    terrain = (
        "▁▁▁▂▃▄▅▆▇███▇▆▅▄▃▂▁▁"
        "▁▁▂▃▄▃▂▁▁"
        "▁▂▃▅▇████▇▅▃▂▁▁"
        "▁▁▁▂▄▆█████▆▄▂▁▁"
        "▂▃▄▅▄▃▂▁▁"
        "▁▂▄▆███▆▄▂▁▁"
        "▁▁▁▂▃▅▇██▇▅▃▂▁▁"
        "▁▂▃▄▅▆▅▄▃▂▁▁"
        "▁▁▂▄▇█████▇▄▂▁▁"
        "▁▁▂▃▄▃▂▁▁"
        "▁▁▂▄▆████▆▄▂▁"
    ) * 3

    window = 48
    i = 0
    last = 0
    start = time.time()
    tick = 0

    # Print phrase on a fixed line above
    sys.stdout.write(f"\n  {PHRASE}\n")
    sys.stdout.flush()

    while not stop.is_set():
        offset = i % len(terrain)
        view = (terrain * 2)[offset: offset + window]
        elapsed = time.time() - start
        line = f"  {view}  {elapsed:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.06)
        tick += 1
        if tick % 2 == 0:
            i += 1

    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()
    # clear the phrase line too
    sys.stdout.write("\033[1A\r" + " " * (len(PHRASE) + 4) + "\r")
    sys.stdout.flush()


# ── 9. Andes sharp peaks (single line) ────────────────────────────────────────
def opt9(stop):
    # Block chars for slopes + /\ only at the very tip = sharp Andes silhouette
    # small, medium, large, twin peaks, varied valleys
    terrain = (
        "▁▁▁▂▃/\\▃▂▁▁"
        "▁▁▂/\\▂▁▁▁"
        "▁▂▃▄/\\▄▃▂▁▁"
        "▁▁▂▃/\\/\\▃▂▁▁"
        "▁▂/\\▂▁▁▁▁"
        "▁▁▁▂▃▄/\\▄▃▂▁▁"
        "▁▁▂▃/\\▃▂▁▁"
        "▁▂▃/\\/\\▃▂▁▁"
        "▁▁▁▂/\\▂▁▁▁"
        "▁▂▃▄▄/\\▄▄▃▂▁▁"
        "▁▁▂▃▄/\\▄▃▂▁▁"
        "▁▁▁▂▃/\\▃▂▁▁"
    ) * 3

    window = 36
    i = 0
    last = 0
    start = time.time()
    tick = 0

    while not stop.is_set():
        offset = i % len(terrain)
        view = (terrain * 2)[offset: offset + window]
        elapsed = time.time() - start
        line = f"  {view}  {PHRASE}  {elapsed:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.06)
        tick += 1
        if tick % 2 == 0:
            i += 1

    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 9b. Andes — 2 rows: peaks line + base line ────────────────────────────────
def opt9b(stop):
    # Each mountain defined as (base_chars, peak_char_pair)
    # We build two parallel strips that scroll in sync

    # base strip  — block chars only, no slashes
    base = (
        "▁▁▁▂▃▄▃▂▁▁"
        "▁▁▂▃▂▁▁▁"
        "▁▂▃▄▅▄▃▂▁▁"
        "▁▁▂▃▄▃▄▃▂▁▁"
        "▁▂▃▂▁▁▁▁"
        "▁▁▂▃▄▅▄▃▂▁▁"
        "▁▁▂▃▄▃▂▁▁"
        "▁▂▃▄▃▄▃▂▁▁"
        "▁▁▁▂▃▂▁▁▁"
        "▁▂▃▄▅▅▄▃▂▁▁"
    ) * 3

    # peaks strip — spaces + /\ at tip positions (same offsets as above)
    peaks = (
        "   ▂▃/\\▃▂   "   # aligns with base above — tip at position 5-6
        "  ▂/\\▂  "
        " ▂▃▄/\\▄▃▂  "
        "  ▂▃/\\/\\▃▂  "
        " ▂/\\▂    "
        "  ▂▃▄/\\▄▃▂  "
        "  ▂▃/\\▃▂  "
        " ▂▃/\\/\\▃▂  "
        "   ▂/\\▂   "
        " ▂▃▄▅/\\▅▄▃▂  "
    ) * 3

    window = 36
    i = 0
    last_p = 0
    last_b = 0
    start = time.time()
    tick = 0

    sys.stdout.write("\n")  # make room for 2 lines

    while not stop.is_set():
        offset = i % len(base)
        bview = (base  * 2)[offset: offset + window]
        pview = (peaks * 2)[offset: offset + window]
        elapsed = time.time() - start

        phrase_line = f"  {pview}  {PHRASE}  {elapsed:.1f}s"
        base_line   = f"  {bview}"

        sys.stdout.write(f"\033[1A\r{' '*last_p}\r{phrase_line}\n")
        sys.stdout.write(f"\r{' '*last_b}\r{base_line}")
        sys.stdout.flush()
        last_p = len(phrase_line)
        last_b = len(base_line)
        time.sleep(0.06)
        tick += 1
        if tick % 2 == 0:
            i += 1

    sys.stdout.write(f"\r{' '*last_b}\r")
    sys.stdout.write(f"\033[1A\r{' '*last_p}\r")
    sys.stdout.flush()


# ── 10. Andes — pure block chars, single █ tip ────────────────────────────────
def opt10(stop):
    terrain = (
        "▁▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁"    # tall peak
        "▁▁▁▁▂▃▄█▄▃▂▁▁▁"          # medium sharp
        "▁▁▂▄▆█▆▄▂▁▁"             # steep sides
        "▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁"      # tall again
        "▁▁▁▂▃█▃▂▁▁▁"             # small pointy
        "▁▁▂▃▅▇█▇▅▃▂▁▁"           # medium-tall
        "▁▁▁▂▄▆█▆▄▂▁▁▁"           # steep narrow
        "▁▁▂▃▄▅▇█▇▅▄▃▂▁▁"        # broad base, sharp tip
        "▁▁▁▁▂▄█▄▂▁▁▁▁"           # small
        "▁▁▂▃▄▆█▆▄▃▂▁▁"           # medium
    ) * 3

    window = 36
    i = 0
    last = 0
    start = time.time()
    tick = 0

    while not stop.is_set():
        offset = i % len(terrain)
        view = (terrain * 2)[offset: offset + window]
        elapsed = time.time() - start
        line = f"  {view}  {PHRASE}  {elapsed:.1f}s"
        sys.stdout.write(f"\r{' '*last}\r{line}")
        sys.stdout.flush()
        last = len(line)
        time.sleep(0.06)
        tick += 1
        if tick % 2 == 0:
            i += 1

    sys.stdout.write(f"\r{' '*last}\r")
    sys.stdout.flush()


# ── 11. Andes 2-row parallax (far + near mountain ranges) ─────────────────────
def opt11(stop):
    # Background (far) — smaller, dimmer peaks, scrolls slower
    far = (
        "▁▁▁▂▃▄▅▄▃▂▁▁"
        "▁▁▂▃▄▅▄▃▂▁▁"
        "▁▁▁▂▃▄▃▂▁▁▁"
        "▁▁▂▄▅▄▂▁▁"
        "▁▁▁▁▂▃▄▅▄▃▂▁▁▁"
        "▁▁▂▃▄▃▂▁▁"
        "▁▁▁▂▃▄▅▄▃▂▁▁"
        "▁▁▁▂▄▅▄▂▁▁▁"
    ) * 4

    # Foreground (near) — taller, sharper peaks, scrolls faster
    near = (
        "▁▁▁▂▄▆█▆▄▂▁▁"
        "▁▁▁▂▃▅▇█▇▅▃▂▁▁"
        "▁▁▂▄▆█▆▄▂▁▁"
        "▁▁▁▁▂▄▆█▆▄▂▁▁▁"
        "▁▁▂▃▄▆█▆▄▃▂▁▁"
        "▁▁▁▂▅▇█▇▅▂▁▁"
        "▁▁▂▃▄▅▇█▇▅▄▃▂▁▁"
        "▁▁▁▂▄▅▇█▇▅▄▂▁▁"
    ) * 4

    window = 36
    fi = 0   # far index
    ni = 0   # near index
    last_f = 0
    last_n = 0
    start = time.time()
    tick = 0

    sys.stdout.write("\n")  # room for 2 lines

    while not stop.is_set():
        elapsed = time.time() - start

        fview = (far  * 2)[fi % len(far):  fi % len(far)  + window]
        nview = (near * 2)[ni % len(near): ni % len(near) + window]

        far_line  = f"  {fview}"
        near_line = f"  {nview}  {PHRASE}  {elapsed:.1f}s"

        sys.stdout.write(f"\033[1A\r{' '*last_f}\r{far_line}\n")
        sys.stdout.write(f"\r{' '*last_n}\r{near_line}")
        sys.stdout.flush()

        last_f = len(far_line)
        last_n = len(near_line)

        time.sleep(0.06)
        tick += 1
        if tick % 3 == 0:   # far scrolls at 1/3 speed
            fi += 1
        if tick % 2 == 0:   # near scrolls at 1/2 speed (faster than far)
            ni += 1

    sys.stdout.write(f"\r{' '*last_n}\r")
    sys.stdout.write(f"\033[1A\r{' '*last_f}\r")
    sys.stdout.flush()


# ── 11a. Parallax — far layer = intentional clouds ────────────────────────────
def opt11a(stop):
    # Clouds: small rounded bumps, irregular clusters, lots of flat sky
    clouds = (
        "     ▁▂▃▂▁▁         "
        "  ▁▂▃▄▃▂▁▁          "
        "       ▁▁▂▃▃▂▁▁     "
        "   ▁▂▂▃▂▂▁▁         "
        "           ▁▂▃▄▃▂▁▁ "
        "     ▁▂▃▂▁▂▃▂▁      "
        "  ▁▁▂▃▂▁▁           "
        "        ▁▂▃▃▂▁▁     "
    ) * 4

    near = (
        "▁▁▁▂▄▆█▆▄▂▁▁"
        "▁▁▁▂▃▅▇█▇▅▃▂▁▁"
        "▁▁▂▄▆█▆▄▂▁▁"
        "▁▁▁▁▂▄▆█▆▄▂▁▁▁"
        "▁▁▂▃▄▆█▆▄▃▂▁▁"
        "▁▁▁▂▅▇█▇▅▂▁▁"
        "▁▁▂▃▄▅▇█▇▅▄▃▂▁▁"
        "▁▁▁▂▄▅▇█▇▅▄▂▁▁"
    ) * 4

    window = 36
    ci = 0
    ni = 0
    last_c = 0
    last_n = 0
    start = time.time()
    tick = 0

    sys.stdout.write("\n")

    while not stop.is_set():
        elapsed = time.time() - start
        cview = (clouds * 2)[ci % len(clouds): ci % len(clouds) + window]
        nview = (near   * 2)[ni % len(near):   ni % len(near)   + window]

        cloud_line = f"  {cview}"
        near_line  = f"  {nview}  {PHRASE}  {elapsed:.1f}s"

        sys.stdout.write(f"\033[1A\r{' '*last_c}\r{cloud_line}\n")
        sys.stdout.write(f"\r{' '*last_n}\r{near_line}")
        sys.stdout.flush()
        last_c = len(cloud_line)
        last_n = len(near_line)

        time.sleep(0.06)
        tick += 1
        if tick % 4 == 0:
            ci += 1
        if tick % 2 == 0:
            ni += 1

    sys.stdout.write(f"\r{' '*last_n}\r")
    sys.stdout.write(f"\033[1A\r{' '*last_c}\r")
    sys.stdout.flush()


# ── 11b. Parallax — far layer = ░▒▓ hazy distant mountains ───────────────────
def opt11b(stop):
    # Same mountain shapes but rendered with shade chars — looks distant/foggy
    far = (
        "░░░▒▒▓▒▒░░░"
        "░░▒▓▒░░░"
        "░░▒▒▓▓▒▒░░░"
        "░░░▒▓▒▒░░░░"
        "░▒▒▓▓▒▒░░"
        "░░░▒▒▓▒░░░░"
        "░░▒▓▓▒▒░░░"
        "░░░▒▓▒░░░░"
    ) * 4

    near = (
        "▁▁▁▂▄▆█▆▄▂▁▁"
        "▁▁▁▂▃▅▇█▇▅▃▂▁▁"
        "▁▁▂▄▆█▆▄▂▁▁"
        "▁▁▁▁▂▄▆█▆▄▂▁▁▁"
        "▁▁▂▃▄▆█▆▄▃▂▁▁"
        "▁▁▁▂▅▇█▇▅▂▁▁"
        "▁▁▂▃▄▅▇█▇▅▄▃▂▁▁"
        "▁▁▁▂▄▅▇█▇▅▄▂▁▁"
    ) * 4

    window = 36
    fi = 0
    ni = 0
    last_f = 0
    last_n = 0
    start = time.time()
    tick = 0

    sys.stdout.write("\n")

    while not stop.is_set():
        elapsed = time.time() - start
        fview = (far  * 2)[fi % len(far):  fi % len(far)  + window]
        nview = (near * 2)[ni % len(near): ni % len(near) + window]

        far_line  = f"  {fview}"
        near_line = f"  {nview}  {PHRASE}  {elapsed:.1f}s"

        sys.stdout.write(f"\033[1A\r{' '*last_f}\r{far_line}\n")
        sys.stdout.write(f"\r{' '*last_n}\r{near_line}")
        sys.stdout.flush()
        last_f = len(far_line)
        last_n = len(near_line)

        time.sleep(0.06)
        tick += 1
        if tick % 3 == 0:
            fi += 1
        if tick % 2 == 0:
            ni += 1

    sys.stdout.write(f"\r{' '*last_n}\r")
    sys.stdout.write(f"\033[1A\r{' '*last_f}\r")
    sys.stdout.flush()


if __name__ == "__main__":
    print("\nSpinner style demo — each runs for 4s, Ctrl+C skips\n")
    run(opt1,  "1   ▲ [░░▓░░░░]          bouncing bar (current)")
    run(opt2,  "2   »»»                   DOS marquee arrows")
    run(opt3,  "3   [/]                   rotating pipe (classic unix)")
    run(opt4,  "4   ▲ ...                 dot pulse / ellipsis breath")
    run(opt5,  "5   [████▒▒▒▒]            block fill sweep")
    run(opt6,  "6   ⣾                     braille spinner")
    run(opt7,  "7   ▲ [>>> text <<<]      LCD marquee")
    run(opt8,  "8a  ▁▂▄███▄▂▁             mountains scrolling (single line)")
    run(opt8b, "8b  ▁▂▄███▄▂▁             mountains scrolling (phrase above)")
    run(opt9,  "9a  ▁▂▃/\\▃▂▁             Andes sharp peaks (single line)")
    run(opt9b, "9b  ▂▃/\\/\\▃▂            Andes sharp peaks (2 rows)")
    run(opt10, "10  ▁▂▄▆█▆▄▂▁            Andes pure blocks, single █ tip")
    run(opt11,  "11   ▁▂▄▅▄▂▁  (far)      Andes 2-row parallax (original)")
    run(opt11a, "11a  ▁▂▃▂▁    (clouds)   far layer = clouds")
    run(opt11b, "11b  ░▒▓▒░    (haze)     far layer = ░▒▓ distant haze")
    print("\nDone.")
