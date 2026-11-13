"""Tests for M2 (src/asr/transcribe.py). No GPU, network or model download except @slow."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile

from evaluation.baseline_runner import run_baseline
from src.asr import transcribe
from src.asr.transcribe import (
    ASR_METADATA_KEYS,
    WhisperASR,
    build_metadata,
    clips_for_version,
    load_for_asr,
    main,
    missing_wavs,
    to_transcript_records,
    transcribe_clips,
    transcribe_versions,
    write_transcripts,
)

FIXED_NOW = datetime(2026, 11, 8, 20, 0, 0, tzinfo=UTC)
MESSAGES = {
    "msg_001": ("worker_02", "We visited 22 households in Baringo today."),
    "msg_002": ("worker_05", "We screened 48 women in Nakuru."),
}
SAMPLE_RATES = {"clean": 16_000, "noisy": 16_000, "phone": 8_000}


def tone(sr: int, seconds: float = 0.5, freq: float = 440.0) -> npt.NDArray[np.float32]:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_dataset(root: Path, versions: tuple[str, ...] = ("clean", "noisy", "phone")) -> Path:
    """Write WAVs and a §2.4-style manifest under root; return the manifest path."""
    clips: list[dict[str, Any]] = []
    # Written in reverse id order to check that sorting happens.
    for message_id in sorted(MESSAGES, reverse=True):
        sender, text = MESSAGES[message_id]
        for version in versions:
            sr = SAMPLE_RATES[version]
            rel = f"examples/audio/{message_id}_{version}.wav"
            path = root / "examples" / "audio" / f"{message_id}_{version}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(path, tone(sr), sr, subtype="PCM_16")
            clips.append(
                {
                    "clip_id": f"{message_id}_{version}",
                    "message_id": message_id,
                    "sender_id": sender,
                    "version": version,
                    "voice": "en-KE-AsiliaNeural",
                    "path": rel,
                    "sample_rate": sr,
                    "duration_s": 0.5,
                    "reference_text": text,
                    "snr_db_measured": 10.0 if version == "noisy" else None,
                    "noise_seed": 1 if version == "noisy" else None,
                }
            )
    manifest_path = root / "examples" / "audio_manifest.json"
    manifest_path.write_text(json.dumps({"generator": {}, "clips": clips}), encoding="utf-8")
    return manifest_path


class FakeASR:
    """Records what it receives and returns fixed texts (with whitespace to strip)."""

    def __init__(self, texts: dict[int, str] | None = None) -> None:
        self.calls: list[list[tuple[int, npt.DTypeLike, int]]] = []
        self.loads = 0
        self.texts = texts or {}

    def __call__(self, waves: list[npt.NDArray[np.float32]]) -> list[str]:
        self.calls.append([(len(w), w.dtype, w.ndim) for w in waves])
        return [self.texts.get(i, f"  text {i}  ") for i in range(len(waves))]

    def load(self) -> None:
        self.loads += 1

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "fake/model",
            "language": "en",
            "task": "transcribe",
            "decoding": {"num_beams": 1, "do_sample": False},
            "initial_prompt": None,
            "device": "cpu",
            "dtype": "float32",
            "torch_version": "0",
            "transformers_version": "0",
        }


def manifest_clips(n: int = 2, version: str = "clean") -> list[dict[str, Any]]:
    return [
        {
            "message_id": f"msg_{i:03d}",
            "sender_id": f"worker_{i:02d}",
            "version": version,
            "path": f"examples/audio/msg_{i:03d}_{version}.wav",
        }
        for i in range(1, n + 1)
    ]


# ------------------------------------------------------------------ clips_for_version


def test_clips_for_version_filters_and_sorts() -> None:
    clips = manifest_clips(3, "clean")[::-1] + manifest_clips(3, "phone")
    result = clips_for_version({"clips": clips}, "clean")
    assert [c["message_id"] for c in result] == ["msg_001", "msg_002", "msg_003"]
    assert all(c["version"] == "clean" for c in result)


def test_clips_for_version_none_raises() -> None:
    with pytest.raises(ValueError, match="no clips for version 'noisy'"):
        clips_for_version({"clips": manifest_clips(2, "clean")}, "noisy")


def test_clips_for_version_duplicate_raises() -> None:
    clips = manifest_clips(2) + manifest_clips(1)
    with pytest.raises(ValueError, match="duplicate message_ids"):
        clips_for_version({"clips": clips}, "clean")


# ----------------------------------------------------------------------- load_for_asr


def test_load_for_asr_upsamples_8k_to_16k_float32_mono() -> None:
    wave = tone(8_000, seconds=1.0).astype(np.float64)
    out = load_for_asr(wave, 8_000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] == 16_000
    assert out.flags["C_CONTIGUOUS"]


def test_load_for_asr_16k_passthrough() -> None:
    wave = tone(16_000)
    out = load_for_asr(wave, 16_000)
    np.testing.assert_array_equal(out, wave)
    assert out.dtype == np.float32


def test_load_for_asr_stereo_downmixed() -> None:
    left = tone(16_000)
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    out = load_for_asr(stereo, 16_000)
    assert out.ndim == 1
    np.testing.assert_allclose(out, left / 2, atol=1e-7)


def test_load_for_asr_rejects_3d() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        load_for_asr(np.zeros((2, 2, 2), dtype=np.float32), 16_000)


# --------------------------------------------------------------- to_transcript_records


def test_to_transcript_records_keys_and_strip() -> None:
    records = to_transcript_records(manifest_clips(2), ["  hello there \n", ""])
    assert [list(r) for r in records] == [["id", "sender_id", "source", "message"]] * 2
    assert records[0] == {
        "id": "msg_001",
        "sender_id": "worker_01",
        "source": "audio",
        "message": "hello there",
    }
    assert records[1]["message"] == ""


def test_to_transcript_records_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="1 transcripts for 2 clips"):
        to_transcript_records(manifest_clips(2), ["only one"])


# ------------------------------------------------------------------- transcribe_clips


def test_transcribe_clips_phone_arrives_at_16k_in_order(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    manifest = json.loads((tmp_path / "examples" / "audio_manifest.json").read_text("utf-8"))
    clips = clips_for_version(manifest, "phone")
    read_paths: list[Path] = []

    def reader(path: Path) -> tuple[npt.NDArray[Any], int]:
        read_paths.append(path)
        return transcribe.read_wav(path)

    asr = FakeASR({0: " first ", 1: "second"})
    records = transcribe_clips(clips, reader, asr, root=tmp_path)

    assert read_paths == [
        tmp_path / "examples" / "audio" / "msg_001_phone.wav",
        tmp_path / "examples" / "audio" / "msg_002_phone.wav",
    ]
    assert len(asr.calls) == 1
    # 0.5 s at 8 kHz on disk -> 0.5 s at 16 kHz = 8000 samples, float32, mono.
    assert asr.calls[0] == [(8_000, np.dtype(np.float32), 1)] * 2
    assert [r["id"] for r in records] == ["msg_001", "msg_002"]
    assert [r["message"] for r in records] == ["first", "second"]
    assert records[1]["sender_id"] == "worker_05"


def test_transcribe_clips_asr_length_mismatch_raises() -> None:
    def reader(path: Path) -> tuple[npt.NDArray[Any], int]:
        return tone(16_000), 16_000

    with pytest.raises(ValueError, match="transcripts for 2 clips"):
        transcribe_clips(manifest_clips(2), reader, lambda waves: ["x"])


def test_read_wav_returns_float32_and_rate(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    soundfile.write(path, tone(8_000), 8_000, subtype="PCM_16")
    wave, sr = transcribe.read_wav(path)
    assert sr == 8_000
    assert wave.dtype == np.float32
    assert wave.shape == (4_000,)


# ------------------------------------------------------------- missing WAV / metadata


def test_missing_wavs_lists_absent_files(tmp_path: Path) -> None:
    make_dataset(tmp_path, versions=("clean",))
    clips = manifest_clips(3)
    assert missing_wavs(clips, tmp_path) == [tmp_path / "examples" / "audio" / "msg_003_clean.wav"]


def test_build_metadata_key_order_and_format() -> None:
    meta = build_metadata(
        "phone",
        FakeASR().metadata(),
        30,
        141.3049,
        12.7449,
        datetime(2026, 11, 8, 20, 0, 1, 9).replace(tzinfo=UTC),
    )
    assert list(meta) == [
        "version",
        *ASR_METADATA_KEYS,
        "n_clips",
        "audio_seconds_total",
        "wall_seconds",
        "created_utc",
    ]
    assert meta["audio_seconds_total"] == 141.3
    assert meta["wall_seconds"] == 12.74
    assert meta["created_utc"] == "2026-11-08T20:00:01Z"


def test_whisper_asr_metadata_without_loading() -> None:
    asr = WhisperASR("openai/whisper-small", "cuda:0", 4)
    meta = asr.metadata()
    assert list(meta) == list(ASR_METADATA_KEYS)
    assert meta["model"] == "openai/whisper-small"
    assert meta["language"] == "en"
    assert meta["task"] == "transcribe"
    assert meta["decoding"] == {"num_beams": 1, "do_sample": False}
    assert meta["initial_prompt"] is None
    assert meta["device"] == "cuda:0"
    assert meta["dtype"] == "float16"
    assert meta["transformers_version"] == "5.22.1"
    assert isinstance(meta["torch_version"], str)
    assert not asr.loaded
    assert WhisperASR("m", "cpu").dtype_name == "float32"

    prompted = WhisperASR("openai/whisper-small", "cpu", initial_prompt="Nakuru, Narok")
    assert prompted.metadata()["initial_prompt"] == "Nakuru, Narok"


def test_whisper_asr_call_without_and_with_initial_prompt() -> None:
    seen_kwargs: list[dict[str, Any]] = []
    seen_prompt_calls: list[tuple[str, str]] = []
    sentinel_prompt_ids = object()

    class FakeTokenizer:
        def get_prompt_ids(self, text: str, return_tensors: str = "np") -> object:
            seen_prompt_calls.append((text, return_tensors))
            return sentinel_prompt_ids

    class FakePipe:
        tokenizer = FakeTokenizer()

        def __call__(self, inputs: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, str]]:
            seen_kwargs.append(kwargs)
            return [{"text": f"out {i}"} for i in range(len(inputs))]

    wave = tone(16_000, seconds=0.1)

    asr_default = WhisperASR("openai/whisper-small", "cpu", batch_size=2)
    asr_default._pipe = FakePipe()
    assert asr_default([wave]) == ["out 0"]
    assert seen_prompt_calls == []
    assert "prompt_ids" not in seen_kwargs[-1]["generate_kwargs"]

    asr_prompted = WhisperASR(
        "openai/whisper-small",
        "cpu",
        batch_size=2,
        initial_prompt="Nakuru, Narok, Kajiado",
    )
    asr_prompted._pipe = FakePipe()
    assert asr_prompted([wave]) == ["out 0"]
    assert seen_prompt_calls == [("Nakuru, Narok, Kajiado", "pt")]
    assert seen_kwargs[-1]["generate_kwargs"]["prompt_ids"] is sentinel_prompt_ids


def test_whisper_asr_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        WhisperASR("m", "cpu", 0)


def test_package_version_missing() -> None:
    assert transcribe._package_version("no-such-distribution-xyz") == "not installed"


def test_resolve_device() -> None:
    assert transcribe.resolve_device("cpu") == "cpu"
    assert transcribe.resolve_device("auto", lambda: True) == "cuda:0"
    assert transcribe.resolve_device("auto", lambda: False) == "cpu"


def test_resolve_device_auto_uses_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert transcribe.resolve_device("auto") == "cpu"


def test_make_whisper_prints_device_and_does_not_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    asr = transcribe.make_whisper("openai/whisper-small", "auto", 2)
    assert isinstance(asr, WhisperASR)
    assert asr.device == "cuda:0"
    assert asr.batch_size == 2
    assert asr.initial_prompt is None
    assert not asr.loaded
    assert "device: cuda:0" in capsys.readouterr().out

    prompted = transcribe.make_whisper("openai/whisper-small", "cpu", 1, initial_prompt="Kisumu")
    assert isinstance(prompted, WhisperASR)
    assert prompted.initial_prompt == "Kisumu"


def test_load_manifest_rejects_non_manifest(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not an audio manifest"):
        transcribe.load_manifest(path)


# ---------------------------------------------------------------------- orchestration


def test_transcribe_versions_all_writes_files(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    manifest = transcribe.load_manifest(tmp_path / "examples" / "audio_manifest.json")
    out_dir = tmp_path / "out"
    asr = FakeASR({1: "   "})  # second clip: empty output
    factory_calls: list[int] = []

    def factory() -> FakeASR:
        factory_calls.append(1)
        return asr

    written = transcribe_versions(
        manifest,
        ["clean", "noisy", "phone"],
        out_dir,
        factory,
        root=tmp_path,
        now=lambda: FIXED_NOW,
    )

    assert factory_calls == [1]  # model built once for all versions
    assert asr.loads == 1
    assert len(asr.calls) == 3
    assert [p.name for p in written] == [
        f"transcripts_{v}{suffix}"
        for v in ("clean", "noisy", "phone")
        for suffix in (".json", ".meta.json")
    ]
    for version in ("clean", "noisy", "phone"):
        raw = (out_dir / f"transcripts_{version}.json").read_bytes()
        records = json.loads(raw.decode("utf-8"))
        assert raw.decode("utf-8") == json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        assert records == [
            {"id": "msg_001", "sender_id": "worker_02", "source": "audio", "message": "text 0"},
            {"id": "msg_002", "sender_id": "worker_05", "source": "audio", "message": ""},
        ]
        meta = json.loads((out_dir / f"transcripts_{version}.meta.json").read_text("utf-8"))
        assert list(meta) == [
            "version",
            *ASR_METADATA_KEYS,
            "n_clips",
            "audio_seconds_total",
            "wall_seconds",
            "created_utc",
        ]
        assert meta["version"] == version
        assert meta["n_clips"] == 2
        assert meta["audio_seconds_total"] == 1.0
        assert meta["wall_seconds"] >= 0
        assert meta["created_utc"] == "2026-11-08T20:00:00Z"


def test_write_transcripts_utf8_sorted(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "t.json"
    write_transcripts(
        path,
        [
            {"message": "Habari ya asubuhi, café", "sender_id": "w2", "id": "msg_002"},
            {"id": "msg_001", "sender_id": "w1", "message": "a"},
        ],
    )
    text = path.read_text(encoding="utf-8")
    assert "café" in text
    data = json.loads(text)
    assert [r["id"] for r in data] == ["msg_001", "msg_002"]
    assert list(data[1]) == ["id", "sender_id", "source", "message"]
    assert data[1]["source"] == "audio"


def test_missing_wav_stops_before_asr(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    missing = tmp_path / "examples" / "audio" / "msg_002_phone.wav"
    missing.unlink()
    manifest = transcribe.load_manifest(tmp_path / "examples" / "audio_manifest.json")

    def factory() -> FakeASR:
        raise AssertionError("ASR must not be built when a WAV is missing")

    out_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match=r"msg_002_phone\.wav"):
        transcribe_versions(manifest, ["clean", "noisy", "phone"], out_dir, factory, root=tmp_path)
    assert not out_dir.exists()  # clean was not written either: all versions checked first


def test_no_clips_for_requested_version_stops_before_asr(tmp_path: Path) -> None:
    make_dataset(tmp_path, versions=("clean",))
    manifest = transcribe.load_manifest(tmp_path / "examples" / "audio_manifest.json")

    def factory() -> FakeASR:
        raise AssertionError("ASR must not be built")

    with pytest.raises(ValueError, match="no clips for version 'noisy'"):
        transcribe_versions(manifest, ["clean", "noisy"], tmp_path / "out", factory, root=tmp_path)


# ------------------------------------------------------------------------------- CLI


def test_parse_args_defaults() -> None:
    args = transcribe.parse_args([])
    assert args.manifest == transcribe.REPO_ROOT / "examples" / "audio_manifest.json"
    assert args.version == "all"
    assert args.model == transcribe.resolve_default_model()
    assert args.out_dir == transcribe.REPO_ROOT / "examples"
    assert args.device == "auto"
    assert args.batch_size == 8
    assert args.initial_prompt is None


def test_parse_args_initial_prompt() -> None:
    args = transcribe.parse_args(["--initial-prompt", "Nakuru, Narok, Baringo"])
    assert args.initial_prompt == "Nakuru, Narok, Baringo"


def test_parse_args_rejects_unknown_version() -> None:
    with pytest.raises(SystemExit):
        transcribe.parse_args(["--version", "loud"])


def test_main_single_version_with_fake_asr(tmp_path: Path) -> None:
    manifest_path = make_dataset(tmp_path)
    seen: list[tuple[str, str, int]] = []

    def make_asr(model: str, device: str, batch_size: int) -> FakeASR:
        seen.append((model, device, batch_size))
        return FakeASR()

    out_dir = tmp_path / "out"
    argv = ["--manifest", str(manifest_path), "--version", "phone", "--out-dir", str(out_dir)]
    argv += ["--model", "openai/whisper-small", "--batch-size", "4"]
    assert main(argv, make_asr=make_asr, root=tmp_path) == 0
    assert seen == [("openai/whisper-small", "auto", 4)]
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "transcripts_phone.json",
        "transcripts_phone.meta.json",
    ]


def test_main_passes_initial_prompt(tmp_path: Path) -> None:
    manifest_path = make_dataset(tmp_path)
    seen: list[tuple[str, str, int, str | None]] = []

    def make_asr(
        model: str, device: str, batch_size: int, initial_prompt: str | None = None
    ) -> FakeASR:
        seen.append((model, device, batch_size, initial_prompt))
        return FakeASR()

    out_dir = tmp_path / "out"
    argv = [
        "--manifest",
        str(manifest_path),
        "--version",
        "clean",
        "--out-dir",
        str(out_dir),
        "--initial-prompt",
        "Nakuru, Narok, Kajiado",
    ]
    assert main(argv, make_asr=make_asr, root=tmp_path) == 0
    assert seen == [(transcribe.resolve_default_model(), "auto", 8, "Nakuru, Narok, Kajiado")]


def test_resolve_default_model_prefers_local_v3_and_falls_back(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"
    assert transcribe.resolve_default_model(missing) == transcribe.BASE_MODEL

    local_dir = tmp_path / "whisper-large-v3-turbo-v3"
    local_dir.mkdir()
    assert transcribe.resolve_default_model(local_dir) == transcribe.BASE_MODEL

    (local_dir / "config.json").write_text("{}", encoding="utf-8")
    assert transcribe.resolve_default_model(local_dir) == str(local_dir)


def test_parse_args_uses_fallback_when_local_model_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transcribe, "FINETUNED_MODEL_DIR", tmp_path / "absent")
    assert transcribe.parse_args([]).model == transcribe.BASE_MODEL


def test_write_transcripts_enforces_interface_contract(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="Invalid transcript records"):
        write_transcripts(
            bad_path,
            [{"id": "not-an-id", "sender_id": "w1", "source": "audio", "message": "hi"}],
        )
    assert not bad_path.exists()

    wrong_source = tmp_path / "wrong_source.json"
    with pytest.raises(ValueError, match="audio source"):
        write_transcripts(
            wrong_source,
            [{"id": "msg_001", "sender_id": "w1", "source": "text", "message": "hi"}],
        )
    assert not wrong_source.exists()

    good_path = tmp_path / "good.json"
    write_transcripts(
        good_path,
        [
            {
                "id": "msg_001",
                "sender_id": "w1",
                "source": "audio",
                "message": "We visited 5 households in Narok.",
                "asr_text": "We visited five households in Narok.",
            }
        ],
    )
    saved = json.loads(good_path.read_text(encoding="utf-8"))
    assert saved[0]["asr_text"] == "We visited five households in Narok."


def test_main_missing_wav_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = make_dataset(tmp_path)
    (tmp_path / "examples" / "audio" / "msg_001_noisy.wav").unlink()

    def make_asr(model: str, device: str, batch_size: int) -> FakeASR:
        raise AssertionError("ASR must not be built when a WAV is missing")

    argv = ["--manifest", str(manifest_path), "--out-dir", str(tmp_path / "o")]
    code = main(argv, make_asr, root=tmp_path)
    assert code == 1
    assert "msg_001_noisy.wav" in capsys.readouterr().err


def test_main_rejects_bad_batch_size(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--batch-size", "0"]) == 2
    assert "--batch-size" in capsys.readouterr().err


def test_main_missing_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--manifest", str(tmp_path / "nope.json")]) == 1
    assert "nope.json" in capsys.readouterr().err


# ---------------------------------------------------------------------- compatibility


def test_m2_transcript_runs_through_real_baseline(tmp_path: Path) -> None:
    path = tmp_path / "transcripts_clean.json"
    clips = [
        {"message_id": mid, "sender_id": sender, "path": "unused.wav"}
        for mid, (sender, _) in MESSAGES.items()
    ]
    write_transcripts(path, to_transcript_records(clips, [t for _, t in MESSAGES.values()]))
    results = run_baseline(path)
    assert len(results) == 2
    assert {r["id"] for r in results} == {"msg_001", "msg_002"}


# ------------------------------------------------------------------------------- slow


@pytest.mark.slow
def test_whisper_small_single_clip_real_model() -> None:
    device = transcribe.resolve_device("auto")
    asr = WhisperASR("openai/whisper-small", device, batch_size=1)
    texts = asr([load_for_asr(tone(8_000, seconds=2.0), 8_000)])
    assert len(texts) == 1
    assert isinstance(texts[0], str)
    assert asr.loaded
