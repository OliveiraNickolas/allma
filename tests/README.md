# Tests

Run the whole suite with the standard library — no extra dependencies:

```bash
python -m unittest discover tests
```

Or a single module:

```bash
python -m unittest tests.test_config_roundtrip -v
```

## What's here

- **`test_config_roundtrip.py`** — the important one. Every config bug this
  project has hit is a round-trip failure: what you configure isn't what gets
  saved or loaded (`-ngl 0` coupling, dropped chat template, a GPU pin that
  wouldn't clear). These tests render a config, parse it back, and assert the
  values survived — including a headless TUI save/reload cycle that locks the
  chat-template and GPU-unpin fixes. Mutation-checked: disabling either fix
  makes the test fail.

Add a regression test here whenever a config field turns out not to
round-trip. It's cheap insurance against the class of bug that keeps
recurring.
