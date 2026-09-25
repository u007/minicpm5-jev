import subprocess
import sys
from pathlib import Path

import preset

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED = {
    "model_id": "mlx-community/MiniCPM5-2B-8bit",
    "revision": "2d20e8e672ce892d50f7265bfd3fc9b59b718f2a",
    "backend": "mlx",
    "prompt": "chat",
    "host": "127.0.0.1",
    "port": 8080,
}


def test_load_preset_returns_all_keys():
    loaded = preset.load_preset()
    for key, value in EXPECTED.items():
        assert loaded[key] == value


def test_model_dir_is_absolute_and_contains_short_revision():
    loaded = preset.load_preset()
    model_dir = Path(loaded["model_dir"])
    assert model_dir.is_absolute()
    assert model_dir == REPO_ROOT / "models" / "MiniCPM5-2B-8bit@2d20e8e672ce"


def test_cli_prints_single_value():
    result = subprocess.run(
        [sys.executable, "preset.py", "port"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "8080"


def test_cli_unknown_key_exits_non_zero():
    result = subprocess.run(
        [sys.executable, "preset.py", "nonexistent"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
