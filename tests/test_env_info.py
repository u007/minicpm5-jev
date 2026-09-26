import subprocess
import sys
from pathlib import Path

from env_info import collect

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_KEYS = {
    "chip",
    "ram_bytes",
    "macos",
    "python",
    "llm2jev",
    "mlx_lm",
    "transformers",
    "huggingface_hub",
    "model_id",
    "model_revision",
    "jevbench_commit",
}


def test_collect_returns_every_key_with_a_non_empty_value():
    info = collect()
    assert set(info) == EXPECTED_KEYS
    for key, value in info.items():
        assert value not in (None, ""), f"{key} is empty"


def test_collect_reads_preset_values():
    info = collect()
    assert info["model_id"] == "mlx-community/MiniCPM5-2B-8bit"
    assert info["model_revision"] == "2d20e8e672ce892d50f7265bfd3fc9b59b718f2a"


def test_main_writes_json_file(tmp_path):
    out_path = tmp_path / "env.json"
    result = subprocess.run(
        [sys.executable, "env_info.py", str(out_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()


def test_main_wrong_arg_count_exits_non_zero():
    result = subprocess.run(
        [sys.executable, "env_info.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()
