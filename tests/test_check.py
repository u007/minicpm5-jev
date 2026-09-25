import http.server
import json
import socket
import threading
import time
from pathlib import Path

import pytest
from llm2jev.prompt import ANSWER, render
from transformers import AutoTokenizer

import preset
from check import (
    PreflightError,
    check_health,
    check_labels,
    check_prompt,
    check_revision,
)

PRESET = preset.load_preset()
MODEL_DIR = Path(PRESET["model_dir"])

requires_model = pytest.mark.skipif(
    not (MODEL_DIR / "REVISION").exists(),
    reason="model not fetched; run fetch_model.sh first",
)


def _probe_prompt(tokenizer):
    _, probe, _ = render(tokenizer, "x", {"q": {"type": "noul"}}, ["A", "B"], PRESET["prompt"])
    return probe["q"][0]


@requires_model
def test_check_labels_returns_255_with_real_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    assert check_labels(tokenizer, _probe_prompt(tokenizer)) == 255


@requires_model
def test_check_prompt_accepts_real_rendered_prompt():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt = _probe_prompt(tokenizer)
    assert prompt.endswith("</think>\n\n" + ANSWER)
    check_prompt(prompt)


def test_check_prompt_rejects_unclosed_think():
    prompt = f"<|im_start|>assistant\n<think>\n<think>\n\n</think>\n\n{ANSWER}"
    with pytest.raises(PreflightError, match="unclosed"):
        check_prompt(prompt)


def test_check_prompt_rejects_missing_answer_suffix():
    prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\nNot the answer"
    with pytest.raises(PreflightError, match="does not end with"):
        check_prompt(prompt)


def test_check_revision_rejects_mismatch(tmp_path):
    (tmp_path / "REVISION").write_text("deadbeef")
    with pytest.raises(PreflightError):
        check_revision(str(tmp_path), "somethingelse")


def _start_stub_server(model_value):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "model": model_value, "temperature": 1.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_check_health_accepts_matching_model():
    server = _start_stub_server("/models/expected")
    try:
        check_health(f"http://127.0.0.1:{server.server_port}/health", "/models/expected")
    finally:
        server.shutdown()
        server.server_close()


def test_check_health_rejects_mismatched_model():
    server = _start_stub_server("/models/other")
    try:
        with pytest.raises(PreflightError):
            check_health(f"http://127.0.0.1:{server.server_port}/health", "/models/expected")
    finally:
        server.shutdown()
        server.server_close()


def test_check_health_rejects_unreachable_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # port is free again, so the request below hits a closed connection
    with pytest.raises(PreflightError, match="failed"):
        check_health(f"http://127.0.0.1:{port}/health", "/models/expected")


def test_check_health_rejects_server_that_accepts_but_never_responds():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def accept_and_hang():
        conn, _ = listener.accept()
        time.sleep(10)  # outlives the 5s urlopen timeout under test
        conn.close()

    threading.Thread(target=accept_and_hang, daemon=True).start()
    try:
        with pytest.raises(PreflightError, match="failed"):
            check_health(f"http://127.0.0.1:{port}/health", "/models/expected")
    finally:
        listener.close()
