"""Tests for M3 - evaluation/baseline_runner.py."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evaluation import baseline_runner
from evaluation.baseline_runner import BaselineError, main, run_baseline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def two_messages(tmp_path: Path, messages: list[dict[str, str]]) -> Path:
    """A 2-message input file: one activity report (msg_002) and one small-talk message."""
    by_id = {m["id"]: m for m in messages}
    return _write_json(tmp_path / "two.json", [by_id["msg_002"], by_id["msg_031"]])


EXPECTED_TWO = [
    {
        "id": "msg_002",
        "relevant": True,
        "activity_type": "health_screening",
        "location": "Nakuru",
        "reported_count": 48,
        "verification_status": "pending",
    },
    {
        "id": "msg_031",
        "relevant": False,
        "activity_type": None,
        "location": None,
        "reported_count": None,
        "verification_status": "not_applicable",
    },
]


def test_default_script_is_repo_baseline(repo_root: Path) -> None:
    assert repo_root / "src" / "legacy" / "baseline.py" == baseline_runner.DEFAULT_BASELINE_SCRIPT
    assert baseline_runner.DEFAULT_BASELINE_SCRIPT.is_file()


def test_real_baseline_on_two_messages(two_messages: Path) -> None:
    records = run_baseline(two_messages)
    assert records == EXPECTED_TWO
    assert [list(r) for r in records] == [list(EXPECTED_TWO[0])] * 2


def test_committed_baseline_output_unchanged(repo_root: Path, two_messages: Path) -> None:
    committed = repo_root / "examples" / "baseline_output.json"
    before = _sha256(committed)
    run_baseline(two_messages)
    assert _sha256(committed) == before
    # Nothing is written next to the input either.
    assert not (two_messages.parent / "examples").exists()


def test_relative_input_path_is_resolved(
    two_messages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(two_messages.parent)
    assert run_baseline(Path(two_messages.name)) == EXPECTED_TWO


def test_non_ascii_message_round_trips(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "accents.json",
        [{"id": "m1", "sender_id": "w1", "message": "Tulitembelea 5 households — Nakuru ✓"}],
    )
    [record] = run_baseline(path)
    assert record["id"] == "m1"
    assert record["location"] == "Nakuru"
    assert record["reported_count"] == 5


def test_subprocess_call_contract(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["cmd"] = cmd
        seen.update(kwargs)
        out = Path(kwargs["cwd"]) / "examples"
        out.mkdir()
        (out / "baseline_output.json").write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    input_json = tmp_path / "in.json"
    assert run_baseline(input_json) == []
    assert seen["cmd"] == [
        sys.executable,
        str((repo_root / "src" / "legacy" / "baseline.py").resolve()),
        str(input_json.resolve()),
    ]
    assert seen["check"] is True
    assert seen["capture_output"] is True
    assert seen["text"] is True
    assert seen["encoding"] == "utf-8"
    assert seen["timeout"] == 60
    assert Path(seen["cwd"]).resolve() != repo_root
    assert not Path(seen["cwd"]).exists()  # the temporary directory is cleaned up


def test_broken_input_raises_with_stderr(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text('[{"id": "msg_001", ', encoding="utf-8")
    with pytest.raises(BaselineError) as info:
        run_baseline(broken)
    assert "JSONDecodeError" in info.value.stderr
    assert "Traceback" in info.value.stderr
    assert info.value.stderr in str(info.value)
    assert "exited with code 1" in str(info.value)
    assert isinstance(info.value.__cause__, subprocess.CalledProcessError)


def test_missing_input_raises_with_stderr(tmp_path: Path) -> None:
    with pytest.raises(BaselineError) as info:
        run_baseline(tmp_path / "does_not_exist.json")
    assert "FileNotFoundError" in info.value.stderr


def test_timeout_raises_with_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _write_script(
        tmp_path / "slow.py",
        "import sys, time\nprint('still working', file=sys.stderr, flush=True)\ntime.sleep(30)\n",
    )
    monkeypatch.setattr(baseline_runner, "TIMEOUT_SECONDS", 1)
    with pytest.raises(BaselineError) as info:
        run_baseline(tmp_path / "in.json", baseline_script=script)
    assert "timed out after 1 s" in str(info.value)
    assert "still working" in info.value.stderr
    assert isinstance(info.value.__cause__, subprocess.TimeoutExpired)


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [(b"partial \xe2\x9c\x93", "partial ✓"), (None, "")],
)
def test_timeout_stderr_bytes_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: bytes | None, expected: str
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(BaselineError) as info:
        run_baseline(tmp_path / "in.json")
    assert info.value.stderr == expected
    assert "timed out after 60 s" in str(info.value)


def test_missing_output_raises_with_stderr(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path / "silent.py", "import sys\nprint('nothing written', file=sys.stderr)\n"
    )
    with pytest.raises(BaselineError) as info:
        run_baseline(tmp_path / "in.json", baseline_script=script)
    assert "wrote no examples/baseline_output.json" in str(info.value)
    assert "nothing written" in info.value.stderr
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_invalid_output_json_raises(tmp_path: Path) -> None:
    script = _write_script(
        tmp_path / "garbage.py",
        "from pathlib import Path\n"
        "Path('examples').mkdir()\n"
        "Path('examples', 'baseline_output.json').write_text('not json', encoding='utf-8')\n",
    )
    with pytest.raises(BaselineError, match="not valid JSON") as info:
        run_baseline(tmp_path / "in.json", baseline_script=script)
    assert info.value.stderr == ""


@pytest.mark.parametrize("payload", ['{"id": "msg_001"}', "[1, 2]"])
def test_output_not_a_list_of_objects_raises(tmp_path: Path, payload: str) -> None:
    script = _write_script(
        tmp_path / "wrong_shape.py",
        "from pathlib import Path\n"
        "Path('examples').mkdir()\n"
        f"Path('examples', 'baseline_output.json').write_text({payload!r}, encoding='utf-8')\n",
    )
    with pytest.raises(BaselineError, match="not a list of objects"):
        run_baseline(tmp_path / "in.json", baseline_script=script)


def test_error_without_stderr_has_plain_message() -> None:
    error = BaselineError("boom")
    assert str(error) == "boom"
    assert error.stderr == ""


def test_main_prints_records(two_messages: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(two_messages)]) == 0
    assert json.loads(capsys.readouterr().out) == EXPECTED_TWO


def test_main_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.json")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: baseline exited with code 1")
    assert "FileNotFoundError" in err
