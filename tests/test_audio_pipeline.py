"""Offline tests for src/audio_pipeline.py covering Stage B -> Stage C v2 contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import audio_pipeline
import contracts
import make_audio
import transcribe

FIXED_NOW = datetime(2026, 11, 13, 12, 30, 0, tzinfo=UTC)


class FakeASR:
    """Offline ASR stand-in tracking loads and returning scripted transcripts."""

    def __init__(
        self,
        model_id: str,
        device: str,
        batch_size: int,
        initial_prompt: str | None = None,
        outputs: dict[int, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.initial_prompt = initial_prompt
        self.outputs = outputs or {}
        self.loads = 0
        self.calls: list[int] = []

    def load(self) -> None:
        self.loads += 1

    def __call__(self, waves: list[np.ndarray[Any, np.dtype[np.float32]]]) -> list[str]:
        self.loads += 1
        self.calls.append(len(waves))
        return [
            self.outputs.get(i, "We screened forty eight women in Narak.")
            for i in range(len(waves))
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "language": transcribe.LANGUAGE,
            "task": transcribe.TASK,
            "decoding": dict(transcribe.DECODING),
            "initial_prompt": self.initial_prompt,
            "device": self.device,
            "dtype": "float32",
            "torch_version": "test",
            "transformers_version": "test",
        }


def _make_manifest(
    root: Path, clips_meta: list[tuple[str, str, str, float]] | None = None
) -> tuple[Path, dict[str, Any]]:
    rows = clips_meta or [
        ("msg_002", "worker_05", "clean", 1.5),
        ("msg_001", "worker_02", "clean", 2.0),
        ("msg_002", "worker_05", "noisy", 1.5),
        ("msg_001", "worker_02", "noisy", 2.0),
    ]
    audio_dir = root / "examples" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    for msg_id, sender_id, version, dur in rows:
        rel = Path("examples") / "audio" / f"{msg_id}_{version}.wav"
        wav_path = root / rel
        wave = np.zeros(int(transcribe.ASR_SAMPLE_RATE * dur), dtype=np.float32)
        make_audio.write_wav(wav_path, wave, transcribe.ASR_SAMPLE_RATE)
        clips.append(
            {
                "clip_id": f"{msg_id}_{version}",
                "message_id": msg_id,
                "sender_id": sender_id,
                "version": version,
                "path": rel.as_posix(),
                "sample_rate": transcribe.ASR_SAMPLE_RATE,
                "duration_s": dur,
            }
        )
    manifest = {"generator": {"script": "test"}, "clips": clips}
    manifest_path = root / "examples" / "audio_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_detect_phone_spans_and_adapt_v1_baseline() -> None:
    msgs = [
        {
            "id": "msg_001",
            "sender_id": "worker_02",
            "source": "audio",
            "message": "Visited 12 households in Narok, call 0722 123 456.",
        },
        {
            "id": "msg_002",
            "sender_id": "worker_01",
            "source": "audio",
            "message": "Good morning everyone!",
        },
    ]
    v1 = [
        {
            "id": "msg_001",
            "relevant": True,
            "activity_type": "household_outreach",
            "location": "Narok",
            "reported_count": 12,
            "verification_status": "pending",
        },
        {
            "id": "msg_002",
            "relevant": False,
            "activity_type": None,
            "location": None,
            "reported_count": None,
            "verification_status": "not_applicable",
        },
    ]
    records = audio_pipeline.adapt_v1_baseline_records(msgs, v1)
    assert contracts.validate_file("records", records, msgs) == []
    assert records[0]["source"] == "audio"
    assert records[0]["sender_id"] == "worker_02"
    assert records[0]["redactions"] == [{"type": "PHONE", "start": 37, "end": 49}]
    assert records[0]["text_redacted"] == "Visited 12 households in Narok, call [PHONE]."
    assert records[1]["verification_status"] == "not_applicable"


def test_adapt_v1_baseline_rejects_mismatch() -> None:
    msg = {"id": "msg_001", "sender_id": "w1", "source": "audio", "message": "hi"}
    with pytest.raises(ValueError, match="returned 0 records"):
        audio_pipeline.adapt_v1_baseline_records([msg], [])
    with pytest.raises(ValueError, match="do not match"):
        audio_pipeline.adapt_v1_baseline_records([msg], [{"id": "msg_002", "relevant": False}])


def test_baseline_v2_system_end_to_end_and_error(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = repo_root / "examples" / "baseline_output.json"
    before = hashlib.sha256(committed.read_bytes()).hexdigest()

    msgs = [
        {
            "id": "msg_001",
            "sender_id": "worker_02",
            "source": "audio",
            "message": "We visited 22 households in Baringo today, call 0722 123 456.",
        },
        {
            "id": "msg_002",
            "sender_id": "worker_01",
            "source": "audio",
            "message": "Hello team, wishing you a good morning.",
        },
    ]
    records = audio_pipeline.baseline_v2_system(msgs)
    assert contracts.validate_file("records", records, msgs) == []
    assert records[0]["relevant"] is True
    assert records[0]["location"] == "Baringo"
    assert records[0]["reported_count"] == 22
    assert records[0]["text_redacted"] == "We visited 22 households in Baringo today, call [PHONE]."
    assert records[1]["relevant"] is False
    assert hashlib.sha256(committed.read_bytes()).hexdigest() == before

    import baseline_runner

    def boom(_path: Path) -> list[dict[str, Any]]:
        raise baseline_runner.BaselineError("broken baseline")

    monkeypatch.setattr(baseline_runner, "run_baseline", boom)
    with pytest.raises(ValueError, match="baseline failed: broken baseline"):
        audio_pipeline.baseline_v2_system(msgs)


def test_run_pipeline_uses_local_v3_model_and_clean_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_manifest(tmp_path)
    local_v3 = tmp_path / "models" / "whisper-large-v3-turbo-v3"
    local_v3.mkdir(parents=True)
    (local_v3 / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(transcribe, "FINETUNED_MODEL_DIR", local_v3)

    factory_calls: list[tuple[str, str, int, str | None]] = []

    def make_asr(
        model_id: str, device: str, batch_size: int, initial_prompt: str | None = None
    ) -> FakeASR:
        factory_calls.append((model_id, device, batch_size, initial_prompt))
        return FakeASR(
            model_id,
            device,
            batch_size,
            initial_prompt,
            outputs={
                0: "We visited twenty two households in Baringo today.",
                1: "We screened forty eight women in Narak.",
            },
        )

    out_dir = tmp_path / "transcripts"
    records_dir = tmp_path / "records"
    manifest = transcribe.load_manifest(tmp_path / "examples" / "audio_manifest.json")
    results = audio_pipeline.run_pipeline(
        manifest,
        ["clean", "noisy"],
        out_dir,
        make_asr,
        device="cpu",
        batch_size=4,
        clean=True,
        records_dir=records_dir,
        root=tmp_path,
        now=lambda: FIXED_NOW,
    )

    assert len(factory_calls) == 1
    assert factory_calls[0][0] == str(local_v3)
    assert len(results) == 2

    clean_res = results[0]
    assert [m["id"] for m in clean_res.messages] == ["msg_001", "msg_002"]
    assert clean_res.messages[0]["message"] == "We visited 22 households in Baringo today."
    assert (
        clean_res.messages[0]["asr_text"]
        == "We visited twenty two households in Baringo today."
    )
    assert clean_res.messages[1]["message"] == "We screened 48 women in Narok."
    assert contracts.validate_file("transcripts", clean_res.messages) == []
    assert contracts.validate_file("records", clean_res.records, clean_res.messages) == []

    meta = json.loads(clean_res.meta_path.read_text(encoding="utf-8"))
    assert meta["model"] == "models/whisper-large-v3-turbo-v3"
    assert meta["cleaned"] is True
    assert meta["created_utc"] == "2026-11-13T12:30:00Z"


def test_run_pipeline_raw_mode_and_system_contract_isolation(tmp_path: Path) -> None:
    _make_manifest(tmp_path, [("msg_001", "worker_02", "clean", 1.0)])
    manifest = transcribe.load_manifest(tmp_path / "examples" / "audio_manifest.json")

    def make_asr(
        model_id: str, device: str, batch_size: int, initial_prompt: str | None = None
    ) -> FakeASR:
        return FakeASR(
            model_id,
            device,
            batch_size,
            initial_prompt,
            outputs={0: "We visited 22 households in Baringo today."},
        )

    def mutating_v2_system(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        original = dict(msgs[0])
        msgs[0]["message"] = "MUTATED"
        return [
            contracts.build_record(
                original,
                relevant=True,
                activity_type="household_outreach",
                location="Baringo",
                reported_count=22,
                date_text="today",
                event_name=None,
                event_desc=None,
                redactions=[],
                confidence=0.95,
            )
        ]

    [res] = audio_pipeline.run_pipeline(
        manifest,
        ["clean"],
        tmp_path / "out",
        make_asr,
        model="custom/whisper-v3",
        device="cpu",
        clean=False,
        system_fn=mutating_v2_system,
        root=tmp_path,
    )
    assert "asr_text" not in res.messages[0]
    assert res.messages[0]["message"] == "We visited 22 households in Baringo today."
    assert res.records[0]["confidence"] == 0.95
    assert res.records[0]["date_text"] == "today"
    raw_bytes = res.records_path.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert raw_bytes.endswith(b"\n")


def test_preflight_and_atomic_validation_prevent_writes(tmp_path: Path) -> None:
    _make_manifest(
        tmp_path,
        [("msg_001", "w1", "clean", 1.0), ("msg_002", "w2", "noisy", 1.0)],
    )
    (tmp_path / "examples" / "audio" / "msg_002_noisy.wav").unlink()
    manifest = transcribe.load_manifest(tmp_path / "examples" / "audio_manifest.json")

    called = False

    def fail_if_called(*_args: Any) -> FakeASR:
        nonlocal called
        called = True
        return FakeASR("m", "cpu", 1)

    out_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="missing 1 audio files"):
        audio_pipeline.run_pipeline(
            manifest, ["clean", "noisy"], out_dir, fail_if_called, root=tmp_path
        )
    assert called is False
    assert not out_dir.exists()

    # Empty versions or invalid batch_size fail before loading ASR.
    with pytest.raises(ValueError, match="at least one version"):
        audio_pipeline.run_pipeline(manifest, [], out_dir, fail_if_called, root=tmp_path)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        audio_pipeline.run_pipeline(
            manifest, ["clean"], out_dir, fail_if_called, batch_size=0, root=tmp_path
        )


def test_atomic_rollback_on_later_version_contract_failure(tmp_path: Path) -> None:
    _make_manifest(tmp_path)
    manifest = transcribe.load_manifest(tmp_path / "examples" / "audio_manifest.json")
    calls = 0

    def bad_on_second_version(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return audio_pipeline.baseline_v2_system(msgs)
        # Return out-of-order / invalid records on version 2.
        return [{"id": "msg_999"}]

    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="Invalid system output records"):
        audio_pipeline.run_pipeline(
            manifest,
            ["clean", "noisy"],
            out_dir,
            lambda m, d, b, p: FakeASR(m, d, b, p),
            device="cpu",
            system_fn=bad_on_second_version,
            root=tmp_path,
        )
    assert not (out_dir / "transcripts_clean.json").exists()
    assert not (out_dir / "records_clean.json").exists()


def test_run_system_contract_error_cases() -> None:
    with pytest.raises(ValueError, match="Invalid input messages"):
        audio_pipeline.run_system_contract([{"id": "bad"}], audio_pipeline.baseline_v2_system)

    valid_msgs = [
        {"id": "msg_001", "sender_id": "w1", "source": "audio", "message": "Hello"},
        {"id": "msg_002", "sender_id": "w2", "source": "audio", "message": "World"},
    ]
    def non_list_fn(_m: list[dict[str, Any]]) -> Any:
        return "not a list"

    with pytest.raises(ValueError, match="must be a list"):
        audio_pipeline.run_system_contract(valid_msgs, non_list_fn)

    def reversed_v2_system(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recs = audio_pipeline.baseline_v2_system(msgs)
        return list(reversed(recs))

    with pytest.raises(ValueError, match="IDs not sorted"):
        audio_pipeline.run_system_contract(valid_msgs, reversed_v2_system)


def test_load_system_resolution() -> None:
    assert audio_pipeline.load_system("baseline") is audio_pipeline.baseline_v2_system
    loaded = audio_pipeline.load_system("audio_pipeline:baseline_v2_system")
    assert loaded is audio_pipeline.baseline_v2_system
    with pytest.raises(ValueError, match="invalid system specification"):
        audio_pipeline.load_system(":attr")
    with pytest.raises(ValueError, match="not callable"):
        audio_pipeline.load_system("audio_pipeline:_PHONE_PATTERN")


def test_cli_main_success_and_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, _ = _make_manifest(tmp_path, [("msg_001", "worker_02", "clean", 1.0)])
    out_dir = tmp_path / "out"
    code = audio_pipeline.main(
        [
            "--manifest",
            str(manifest_path),
            "--version",
            "clean",
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--clean",
        ],
        make_asr=lambda m, d, b, p: FakeASR(
            m, d, b, p, outputs={0: "We visited twenty two households in Narak."}
        ),
        root=tmp_path,
        now=lambda: FIXED_NOW,
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "clean: 1 messages" in stdout
    assert "1 records (1 relevant)" in stdout
    saved_msgs = json.loads((out_dir / "transcripts_clean.json").read_text(encoding="utf-8"))
    assert saved_msgs[0]["message"] == "We visited 22 households in Narok."

    err_code = audio_pipeline.main(
        ["--manifest", str(tmp_path / "missing.json"), "--device", "cpu"],
        root=tmp_path,
    )
    assert err_code == 1
    assert "error:" in capsys.readouterr().err
