"""Synthetic startup checks only; no real model weights or user footage."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "local" / "preflight.py"


class PreflightTests(unittest.TestCase):
    def run_preflight(self, folder, missing_audio=False):
        source = folder / "synthetic-source.jpg"
        audio = folder / "synthetic-audio.wav"
        ffmpeg = folder / "synthetic-ffmpeg.exe"
        source.write_bytes(b"synthetic source placeholder")
        ffmpeg.write_bytes(b"synthetic ffmpeg placeholder")
        if not missing_audio:
            audio.write_bytes(b"synthetic audio placeholder")
        cfg = folder / "preflight.json"
        cfg.write_text(json.dumps({
            "source": str(source), "audio": str(audio), "ffmpeg": str(ffmpeg),
            "output": str(folder / "out.mp4"), "engine": "puppet",
            "runtime": "openvino", "models": str(folder / "models")
        }), encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-I", str(PREFLIGHT), "--config", str(cfg)],
            capture_output=True, text=True, timeout=30
        )

    def test_missing_audio_returns_actionable_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_preflight(Path(tmp), missing_audio=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("語音 WAV", result.stdout)

    def test_puppet_preflight_does_not_require_neural_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_preflight(Path(tmp))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            self.assertTrue(any(e.get("stage") == "preflight_ok" for e in events))
            self.assertIn("不載入", events[0].get("message", ""))


if __name__ == "__main__":
    unittest.main()
