"""Config round-trip regression tests.

Every config bug this project has hit is the same shape: what you configure
is not what gets saved or loaded. The `-ngl 0` coupling, the dropped chat
template, the GPU pin that would not clear — all round-trip failures. These
tests lock the round-trip so those cannot silently regress.

Plain `unittest`, no pytest required:

    python -m unittest discover tests

The parser tests are pure and fast. The renderer test drives the real TUI
through Textual's headless harness, because `_render_base_allm` reads the
live form (that is how the GPU pin can be cleared), so there is no honest
way to exercise it without mounting the form.
"""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.allm_parser import parse_allm  # noqa: E402


class TestParserRoundTrip(unittest.TestCase):
    """A rendered .allm must parse back to the same values."""

    def test_llama_full_config_survives_parse(self):
        # Mirrors what _render_base_allm emits for a llama.cpp base with the
        # fields that have bitten us: mmproj, chat template, and bare flags.
        text = "\n".join([
            "# Base model: Test (llama.cpp backend)",
            "@llamacpp",
            "@path /models/test.gguf",
            "@gpu 1",
            "",
            "-c 98304",
            "-ngl -1",
            "--mmproj /models/mmproj-F16.gguf",
            "chat-template-file /templates/froggeric.jinja",
            "flash-attn on",
            "spec-type draft-mtp",
            "spec-draft-n-max 6",
        ])
        cfg = parse_allm(text, "test.allm")
        self.assertEqual(cfg.get("backend"), "llama.cpp")
        self.assertEqual(str(cfg.get("n_ctx")), "98304")
        self.assertEqual(str(cfg.get("n_gpu_layers")), "-1")
        self.assertEqual(cfg.get("mmproj"), "/models/mmproj-F16.gguf")
        # The field that used to be dropped on save.
        self.assertEqual(cfg.get("chat_template_file"), "/templates/froggeric.jinja")
        self.assertEqual(str(cfg.get("gpu_id")), "1")

    def test_no_gpu_line_means_no_pin(self):
        # Absence of @gpu must read as "auto", not as a stale pin.
        text = "@llamacpp\n@path /models/test.gguf\n\n-c 4096\n"
        cfg = parse_allm(text, "test.allm")
        self.assertIn(cfg.get("gpu_id"), (None, "", "-1"))


# Path to a base config the harness test can safely own and rewrite.
def _write_temp_base(cfg_dir: Path) -> None:
    (cfg_dir / "base").mkdir(parents=True, exist_ok=True)
    (cfg_dir / "profile").mkdir(parents=True, exist_ok=True)
    (cfg_dir / "base" / "RoundTrip.allm").write_text("\n".join([
        "@llamacpp",
        "@path /models/roundtrip.gguf",
        "@gpu 1",
        "",
        "-c 32768",
        "-ngl -1",
        "flash-attn on",
    ]) + "\n")


class TestRendererRoundTrip(unittest.TestCase):
    """_render_base_allm must not lose fields, and must honour 'auto'.

    Drives the TUI headlessly. Skips cleanly if Textual is unavailable.
    """

    def _run(self, coro):
        return asyncio.run(coro)

    def test_save_preserves_chat_template_and_clears_pin(self):
        tmp = Path(tempfile.mkdtemp())
        cfg_dir = tmp / "configs"
        _write_temp_base(cfg_dir)
        # core.config reads ALLMA_CONFIG_DIR at import time, so set it first.
        os.environ["ALLMA_CONFIG_DIR"] = str(cfg_dir)
        os.environ["ALLMA_TUI_LAYOUT"] = str(tmp / "tui.json")

        try:
            from allma_tui import AllmaTUI, RadioRow
            from textual.widgets import Input
        except Exception as e:  # pragma: no cover - environment guard
            self.skipTest(f"Textual/TUI unavailable: {e}")

        from configs.allm_parser import load_models_from_configs

        async def scenario():
            app = AllmaTUI()
            async with app.run_test(size=(160, 50)):
                await app_pause(app)
                m = next((x for x in app.models if x["key"] == "RoundTrip"), None)
                self.assertIsNotNone(m, "temp base model not discovered")
                app.selected = m
                await app._show_model(m)
                await app_pause(app)

                # (1) set a chat template that started absent
                app.query_one("#ld-chat-template", Input).value = "/templates/rt.jinja"
                # (2) move the GPU pin from 1 -> auto
                app.query_one("#ld-gpu_id", RadioRow).value = ""
                await app_pause(app)

                await app._save_base(m)
                await app_pause(app)

            bases, _ = load_models_from_configs(str(cfg_dir))
            return bases.get("RoundTrip", {})

        saved = self._run(scenario())
        # chat template persisted (the bug: silently dropped on save)
        self.assertEqual(saved.get("chat_template_file"), "/templates/rt.jinja")
        # pin cleared (the bug: auto could never remove a prior @gpu)
        self.assertIn(saved.get("gpu_id"), (None, "", "-1"))
        # untouched field stayed put
        self.assertEqual(str(saved.get("n_ctx")), "32768")


async def app_pause(app):
    """Let the Textual event loop settle without importing test internals."""
    try:
        await app.workers.wait_for_complete()
    except Exception:
        pass
    await asyncio.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
