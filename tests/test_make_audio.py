"""Tests for M1 src/make_audio.py (design §3, §8.2). No network unless marked slow."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

import make_audio
from make_audio import (
    SELECTED_CHAT_IDS,
    SELECTED_IDS,
    SELECTED_REPORT_IDS,
    VERSIONS,
    VOICES,
    EdgeTTSBackend,
    FileNoiseSource,
    PinkNoiseSource,
    Settings,
    SNRError,
    TTSCache,
    TTSError,
    assign_voices,
    build_manifest_entry,
    clip_seed,
    decode_audio_file,
    format_utc,
    generate,
    mix_at_snr,
    pad,
    peak_normalize,
    pink_noise,
    repo_relative,
    select_messages,
    summarize,
    to_mono_16k,
    to_phone,
    write_wav,
)

FIXED_NOW = datetime(2026, 11, 7, 18, 0, 0, tzinfo=UTC)
CLIP_KEYS = [
    "clip_id",
    "message_id",
    "sender_id",
    "version",
    "voice",
    "path",
    "sample_rate",
    "duration_s",
    "reference_text",
    "snr_db_measured",
    "noise_seed",
]
GENERATOR_KEYS = [
    "script",
    "created_utc",
    "tts_backend",
    "tts_backend_version",
    "voices",
    "pad_seconds",
    "snr_db",
    "noise_source",
    "base_seed",
    "source_messages",
]


def tone(freq: float, sr: int, seconds: float = 1.0, amp: float = 0.5) -> npt.NDArray[np.float32]:
    t = np.arange(round(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def no_sleep(_seconds: float) -> None:
    return None


class FakeTTS:
    """Array-only backend: a 1 s 440 Hz tone at 24 kHz, counting calls."""

    name = "fake-tts"
    version = "0.0"

    def __init__(self, failures: int = 0) -> None:
        self.calls = 0
        self.failures = failures

    def synthesize(self, text: str, voice: str) -> tuple[npt.NDArray[Any], int]:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("network down")
        return tone(440.0, 24000), 24000


class FakeFileTTS(FakeTTS):
    """File-producing backend (like edge-tts): writes an MP3."""

    file_suffix = ".mp3"

    def synthesize_to_file(self, text: str, voice: str, path: Path) -> None:
        wave, sr = self.synthesize(text, voice)
        sf.write(path, wave, sr, format="MP3")


def run_generate(
    messages: list[dict[str, str]],
    root: Path,
    tts: Any,
    *,
    ids: list[str] | None = None,
    settings: Settings | None = None,
    sleep: Any = no_sleep,
) -> dict[str, Any]:
    return generate(
        messages,
        tts=tts,
        noise=PinkNoiseSource(),
        out_dir=root / "examples" / "audio",
        manifest_path=root / "examples" / "audio_manifest.json",
        settings=settings or Settings(),
        repo_root=root,
        source_messages=root / "examples" / "synthetic_messages.json",
        ids=ids,
        sleep=sleep,
        now=lambda: FIXED_NOW,
    )


def validate_manifest(manifest: dict[str, Any], root: Path, n_messages: int) -> None:
    """Check a manifest against design §2.3 / §2.4."""
    assert list(manifest) == ["generator", "clips"]
    gen = manifest["generator"]
    assert list(gen) == GENERATOR_KEYS
    assert gen["script"] == "src/make_audio.py"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", gen["created_utc"])
    assert gen["voices"] == VOICES
    assert gen["source_messages"] == "examples/synthetic_messages.json"
    clips = manifest["clips"]
    assert len(clips) == 3 * n_messages
    order = [(c["message_id"], VERSIONS.index(c["version"])) for c in clips]
    assert order == sorted(order)
    for clip in clips:
        assert list(clip) == CLIP_KEYS
        assert clip["clip_id"] == f"{clip['message_id']}_{clip['version']}"
        assert clip["path"] == f"examples/audio/{clip['clip_id']}.wav"
        info = sf.info(root / clip["path"])
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.samplerate == clip["sample_rate"]
        assert clip["sample_rate"] == (8000 if clip["version"] == "phone" else 16000)
        assert clip["duration_s"] == pytest.approx(info.frames / info.samplerate, abs=0.005)
        if clip["version"] == "noisy":
            assert abs(clip["snr_db_measured"] - gen["snr_db"]) <= 0.1
            assert clip["noise_seed"] == clip_seed(gen["base_seed"], clip["message_id"])
        else:
            assert clip["snr_db_measured"] is None
            assert clip["noise_seed"] is None


# --------------------------------------------------------------------------------------
# select_messages / assign_voices
# --------------------------------------------------------------------------------------


def test_select_messages_returns_ids_order() -> None:
    msgs = [{"id": f"m{i}", "sender_id": "s", "message": str(i)} for i in range(4)]
    assert [m["id"] for m in select_messages(msgs, ["m3", "m0", "m2"])] == ["m3", "m0", "m2"]


def test_select_messages_missing_id_raises() -> None:
    msgs = [{"id": "m0", "sender_id": "s", "message": "x"}]
    with pytest.raises(ValueError, match=r"missing.*m7, m9"):
        select_messages(msgs, ["m0", "m7", "m9"])


def test_select_messages_duplicate_selected_id_raises() -> None:
    msgs = [{"id": "m0", "sender_id": "s", "message": "x"}]
    with pytest.raises(ValueError, match="duplicate selected IDs: m0"):
        select_messages(msgs, ["m0", "m0"])


def test_select_messages_duplicate_record_raises() -> None:
    msgs = [{"id": "m0", "sender_id": "s", "message": "x"}] * 2
    with pytest.raises(ValueError, match="duplicate message records: m0"):
        select_messages(msgs, ["m0"])


def test_shipped_selection_is_30_unique_existing(messages: list[dict[str, str]]) -> None:
    assert len(SELECTED_REPORT_IDS) == 24
    assert len(SELECTED_CHAT_IDS) == 6
    assert len(set(SELECTED_IDS)) == 30
    selected = select_messages(messages, SELECTED_IDS)
    assert [m["id"] for m in selected] == SELECTED_IDS


def test_assign_voices_alternates() -> None:
    assert assign_voices(["a", "b", "c"], ["v1", "v2"]) == {"a": "v1", "b": "v2", "c": "v1"}


def test_assign_voices_shipped_selection_is_balanced() -> None:
    voices = assign_voices(SELECTED_IDS, VOICES)
    counts = [list(voices.values()).count(v) for v in VOICES]
    assert counts == [15, 15]
    report_counts = [[voices[i] for i in SELECTED_REPORT_IDS].count(v) for v in VOICES]
    assert report_counts == [12, 12]


def test_assign_voices_needs_a_voice() -> None:
    with pytest.raises(ValueError, match="voice"):
        assign_voices(["a"], [])


# --------------------------------------------------------------------------------------
# Signal functions
# --------------------------------------------------------------------------------------


def test_to_mono_16k_averages_stereo_and_resamples() -> None:
    left = tone(440.0, 24000)
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    out = to_mono_16k(stereo, 24000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == 16000
    assert np.max(np.abs(out)) == pytest.approx(0.25, abs=0.01)


def test_to_mono_16k_keeps_16k_input() -> None:
    wave = tone(440.0, 16000).astype(np.float64)
    out = to_mono_16k(wave, 16000)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, wave, atol=1e-7)


def test_to_mono_16k_rejects_3d() -> None:
    with pytest.raises(ValueError, match="shape"):
        to_mono_16k(np.zeros((2, 2, 2)), 16000)


def test_peak_normalize_sets_peak() -> None:
    out = peak_normalize(np.array([0.1, -0.5, 0.25]))
    assert float(np.max(np.abs(out))) == pytest.approx(0.89, abs=1e-6)
    assert peak_normalize(np.array([0.2]), peak=0.5)[0] == pytest.approx(0.5)


def test_peak_normalize_all_zero_unchanged() -> None:
    zeros = np.zeros(10, dtype=np.float32)
    np.testing.assert_array_equal(peak_normalize(zeros), zeros)


def test_pad_adds_zeros_at_both_ends() -> None:
    wave = np.ones(100, dtype=np.float32)
    out = pad(wave, 16000, 0.0123)
    n = round(0.0123 * 16000)
    assert out.size == wave.size + 2 * n
    assert not out[:n].any()
    assert not out[-n:].any()
    np.testing.assert_array_equal(out[n:-n], wave)


def test_pink_noise_reproducible_and_seed_dependent() -> None:
    a = pink_noise(16000, 7)
    assert a.tobytes() == pink_noise(16000, 7).tobytes()
    assert not np.array_equal(a, pink_noise(16000, 8))
    assert float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) == pytest.approx(1.0, abs=1e-4)


def test_pink_noise_spectrum_slope_is_minus_10_db_per_decade() -> None:
    sr, seg = 16000, 4096
    noise = pink_noise(seg * 64, 3).astype(np.float64)
    frames = noise.reshape(64, seg) * np.hanning(seg)
    power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(seg, 1 / sr)
    band = (freqs >= 100) & (freqs <= 4000)
    slope = np.polyfit(np.log10(freqs[band]), 10 * np.log10(power[band]), 1)[0]
    assert slope == pytest.approx(-10.0, abs=2.0)


def test_pink_noise_needs_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        pink_noise(1, 0)


def padded_sine() -> tuple[npt.NDArray[np.float32], slice]:
    speech = tone(300.0, 16000, amp=0.89)
    return pad(speech, 16000, 1.0), slice(16000, 16000 + speech.size)


@pytest.mark.parametrize("snr", [0.0, 10.0, 20.0])
def test_mix_at_snr_measured_over_speech_region(snr: float) -> None:
    clean, region = padded_sine()
    noise = pink_noise(clean.size, 1)
    mixed, measured = mix_at_snr(clean, noise, snr, region)
    assert mixed.dtype == np.float32
    assert mixed.size == clean.size
    assert measured == pytest.approx(snr, abs=0.05)
    if float(np.max(np.abs(mixed))) < 0.99:  # no anti-clipping: re-measure independently
        noise_part = mixed.astype(np.float64) - clean
        independent = 10 * np.log10(
            np.mean(clean[region].astype(np.float64) ** 2) / np.mean(noise_part**2)
        )
        assert independent == pytest.approx(snr, abs=0.05)
    # Noise covers the padding too.
    assert np.abs(mixed[:16000]).max() > 0


def test_mix_at_snr_anti_clipping_keeps_snr() -> None:
    clean, region = padded_sine()
    noise = pink_noise(clean.size, 2)
    mixed, measured = mix_at_snr(clean, noise, 0.0, region)
    assert float(np.max(np.abs(mixed))) == pytest.approx(0.99, abs=1e-6)
    assert measured == pytest.approx(0.0, abs=0.05)
    # Recover the scale from the speech-only reference and re-measure independently.
    scale = 0.99 / float(np.max(np.abs(clean.astype(np.float64) + _gain(clean, noise, region))))
    speech_part = clean.astype(np.float64) * scale
    noise_part = mixed.astype(np.float64) - speech_part
    independent = 10 * np.log10(np.mean(speech_part[region] ** 2) / np.mean(noise_part**2))
    assert independent == pytest.approx(0.0, abs=0.05)


def _gain(
    clean: npt.NDArray[np.float32], noise: npt.NDArray[np.float32], region: slice
) -> npt.NDArray[np.float64]:
    p_s = np.mean(clean[region].astype(np.float64) ** 2)
    p_n = np.mean(noise.astype(np.float64) ** 2)
    gain: npt.NDArray[np.float64] = np.sqrt(p_s / p_n) * noise.astype(np.float64)
    return gain


def test_mix_at_snr_rejects_bad_inputs() -> None:
    clean, region = padded_sine()
    with pytest.raises(ValueError, match="lengths differ"):
        mix_at_snr(clean, np.ones(3), 10.0, region)
    with pytest.raises(ValueError, match="speech region has zero power"):
        mix_at_snr(np.zeros(10), np.ones(10), 10.0, slice(2, 8))
    with pytest.raises(ValueError, match="noise has zero power"):
        mix_at_snr(clean, np.zeros_like(clean), 10.0, region)


def _band_energy(wave: npt.NDArray[np.float32], sr: int, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(wave.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(wave.size, 1 / sr)
    return float(spectrum[np.abs(freqs - freq) < 50].sum())


def test_to_phone_halves_length_and_low_passes() -> None:
    low = tone(1000.0, 16000)
    high = tone(6000.0, 16000)
    out_low = to_phone(low)
    out_high = to_phone(high)
    assert out_low.dtype == np.float32
    assert abs(out_low.size - low.size // 2) <= 1
    kept = _band_energy(out_low, 8000, 1000.0) / _band_energy(low, 16000, 1000.0)
    assert kept == pytest.approx(0.25, rel=0.1)  # same amplitude, half the samples
    assert float(np.sqrt(np.mean(out_high.astype(np.float64) ** 2))) < 0.01


def test_clip_seed_formula_and_range() -> None:
    import zlib

    assert clip_seed(1234, "msg_002") == (1234 + zlib.crc32(b"msg_002")) % 2**32
    assert clip_seed(2**32 - 1, "msg_002") == (2**32 - 1 + zlib.crc32(b"msg_002")) % 2**32
    assert 0 <= clip_seed(2**40, "x") < 2**32
    assert clip_seed(1234, "msg_001") != clip_seed(1234, "msg_002")


def test_build_manifest_entry_matches_contract() -> None:
    message = {"id": "msg_002", "sender_id": "worker_05", "message": "We screened 48 women."}
    entry = build_manifest_entry(
        message=message,
        version="noisy",
        voice="en-KE-ChilembaNeural",
        path="examples/audio/msg_002_noisy.wav",
        sample_rate=16000,
        n_samples=77123,
        snr_db_measured=10.00004,
        noise_seed=42,
    )
    assert list(entry) == CLIP_KEYS
    assert entry["clip_id"] == "msg_002_noisy"
    assert entry["duration_s"] == 4.82
    assert entry["snr_db_measured"] == 10.0
    assert entry["reference_text"] == "We screened 48 women."
    clean = build_manifest_entry(
        message=message, version="clean", voice="v", path="p", sample_rate=8000, n_samples=8000
    )
    assert clean["snr_db_measured"] is None
    assert clean["noise_seed"] is None
    with pytest.raises(ValueError, match="unknown version"):
        build_manifest_entry(
            message=message, version="loud", voice="v", path="p", sample_rate=1, n_samples=1
        )


def test_repo_relative_uses_forward_slashes(tmp_path: Path) -> None:
    inside = tmp_path / "examples" / "audio" / "x.wav"
    assert repo_relative(inside, tmp_path) == "examples/audio/x.wav"
    outside = tmp_path.parent / "elsewhere.wav"
    assert repo_relative(outside, tmp_path) == outside.resolve().as_posix()


def test_format_utc() -> None:
    eat = timezone(timedelta(hours=3))
    assert format_utc(datetime(2026, 11, 7, 21, 0, 5, 999, tzinfo=eat)) == "2026-11-07T18:00:05Z"


def test_summarize() -> None:
    clips = [
        {"version": "clean", "duration_s": 2.5, "snr_db_measured": None},
        {"version": "noisy", "duration_s": 2.5, "snr_db_measured": 9.99},
        {"version": "noisy", "duration_s": 1.0, "snr_db_measured": 10.01},
        {"version": "phone", "duration_s": 2.5, "snr_db_measured": None},
    ]
    text = summarize({"clips": clips})
    assert "clips: 4" in text
    assert "total duration noisy: 3.50 s" in text
    assert "min 9.990 dB, max 10.010 dB" in text
    assert "no noisy clips" in summarize({"clips": clips[:1]})


# --------------------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------------------


def test_package_version() -> None:
    assert re.fullmatch(r"\d+\.\d+.*", make_audio.package_version("numpy"))
    assert make_audio.package_version("no-such-distribution-xyz") == "unknown"


def test_write_wav_and_decode_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    wave = tone(440.0, 8000)
    write_wav(path, wave, 8000)
    info = sf.info(path)
    assert (info.samplerate, info.channels, info.subtype) == (8000, 1, "PCM_16")
    decoded, sr = decode_audio_file(path)
    assert sr == 8000
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, wave, atol=1e-4)


def test_decode_audio_file_reads_mp3(tmp_path: Path) -> None:
    path = tmp_path / "a.mp3"
    sf.write(path, tone(440.0, 24000), 24000, format="MP3")
    decoded, sr = decode_audio_file(path)
    assert sr == 24000
    assert decoded.size > 20000


def test_decode_audio_file_falls_back_to_librosa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.wav"
    write_wav(path, tone(440.0, 16000), 16000)

    def broken_read(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("cannot decode")

    monkeypatch.setattr(sf, "read", broken_read)
    decoded, sr = decode_audio_file(path)
    assert sr == 16000
    assert decoded.dtype == np.float32
    assert decoded.size == 16000


def test_edge_tts_backend_decodes_its_mp3(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = EdgeTTSBackend()
    assert backend.name == "edge-tts"
    assert backend.version == make_audio.package_version("edge-tts")
    assert isinstance(backend, make_audio.FileTTSBackend)

    def fake_save(text: str, voice: str, path: Path) -> None:
        assert path.suffix == ".mp3"
        sf.write(path, tone(440.0, 24000), 24000, format="MP3")

    monkeypatch.setattr(backend, "synthesize_to_file", fake_save)
    wave, sr = backend.synthesize("hello", VOICES[0])
    assert sr == 24000
    assert wave.size > 20000


def test_edge_tts_is_imported_lazily(repo_root: Path) -> None:
    code = "import sys, make_audio; print('edge_tts' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root / "src",
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "False"


def test_pink_noise_source() -> None:
    source = PinkNoiseSource()
    assert source.name == "pink"
    np.testing.assert_array_equal(source.get(1000, 16000, 5), pink_noise(1000, 5))


def test_file_noise_source(tmp_path: Path) -> None:
    write_wav(tmp_path / "b.wav", np.full(100, 0.5, dtype=np.float32), 8000)
    stereo = np.stack([tone(100.0, 16000, 0.01), tone(100.0, 16000, 0.01)], axis=1)
    sf.write(tmp_path / "a.WAV", stereo, 16000, subtype="PCM_16")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    source = FileNoiseSource(tmp_path)
    assert [p.name for p in source.files] == ["a.WAV", "b.wav"]
    assert source.name == f"files:{tmp_path.as_posix()}"
    # seed 0 -> a.WAV (stereo, 160 samples at 16 kHz), cropped / looped to length
    first = source.get(500, 16000, 0)
    assert first.shape == (500,)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, source.get(500, 16000, 0))
    assert not np.array_equal(first, source.get(500, 16000, 2))
    # seed 1 -> b.wav (100 samples at 8 kHz) resampled to 16 kHz and looped
    looped = source.get(1000, 16000, 1)
    assert looped.shape == (1000,)
    assert float(np.median(looped)) == pytest.approx(0.5, abs=0.02)


def test_file_noise_source_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \.wav noise files"):
        FileNoiseSource(tmp_path)

    def empty_reader(_path: Path) -> tuple[npt.NDArray[Any], int]:
        return np.zeros(0, dtype=np.float32), 16000

    (tmp_path / "x.wav").write_bytes(b"")
    source = FileNoiseSource(tmp_path, reader=empty_reader)
    with pytest.raises(ValueError, match="is empty"):
        source.get(10, 16000, 0)


def test_synthesize_with_retry_recovers() -> None:
    slept: list[float] = []
    tts = FakeTTS(failures=2)
    _wave, sr = make_audio.synthesize_with_retry(
        lambda: tts.synthesize("t", "v"), "msg_009", sleep=slept.append
    )
    assert sr == 24000
    assert tts.calls == 3
    assert slept == [2.0, 4.0]


def test_synthesize_with_retry_gives_up_naming_message() -> None:
    slept: list[float] = []
    tts = FakeTTS(failures=99)
    with pytest.raises(TTSError, match="msg_009 after 3 tries"):
        make_audio.synthesize_with_retry(
            lambda: tts.synthesize("t", "v"), "msg_009", sleep=slept.append
        )
    assert tts.calls == 3
    assert slept == [2.0, 4.0]


def test_tts_cache_array_backend(tmp_path: Path) -> None:
    tts = FakeTTS()
    cache = TTSCache(tts, tmp_path / "cache", sleep=no_sleep)
    first, sr = cache.get("msg_001", "text", "voiceA")
    path = tmp_path / "cache" / "msg_001__voiceA.wav"
    assert path.exists()
    assert sf.info(path).subtype == "FLOAT"
    again, sr2 = cache.get("msg_001", "text", "voiceA")
    assert tts.calls == 1
    assert sr2 == sr
    np.testing.assert_array_equal(first, again)
    TTSCache(tts, tmp_path / "cache", force=True, sleep=no_sleep).get("msg_001", "t", "voiceA")
    assert tts.calls == 2


def test_tts_cache_file_backend_stores_raw_mp3(tmp_path: Path) -> None:
    tts = FakeFileTTS()
    cache = TTSCache(tts, tmp_path / "cache", sleep=no_sleep)
    _, sr = cache.get("msg_002", "text", VOICES[1])
    assert sr == 24000
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [f"msg_002__{VOICES[1]}.mp3"]
    cache.get("msg_002", "text", VOICES[1])
    assert tts.calls == 1


def test_tts_cache_failure_leaves_no_file(tmp_path: Path) -> None:
    tts = FakeFileTTS(failures=99)
    cache = TTSCache(tts, tmp_path / "cache", sleep=no_sleep)
    with pytest.raises(TTSError, match="msg_003"):
        cache.get("msg_003", "text", "v")
    assert list((tmp_path / "cache").iterdir()) == []


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_run(
    tmp_path_factory: pytest.TempPathFactory, messages: list[dict[str, str]]
) -> tuple[Path, dict[str, Any], FakeTTS]:
    root = tmp_path_factory.mktemp("repo")
    tts = FakeTTS()
    manifest = run_generate(messages, root, tts)
    return root, manifest, tts


def test_end_to_end_writes_90_clips_and_valid_manifest(
    full_run: tuple[Path, dict[str, Any], FakeTTS], messages: list[dict[str, str]]
) -> None:
    root, manifest, tts = full_run
    assert tts.calls == 30
    wavs = sorted((root / "examples" / "audio").glob("*.wav"))
    assert len(wavs) == 90
    validate_manifest(manifest, root, 30)
    on_disk = json.loads((root / "examples" / "audio_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    gen = manifest["generator"]
    assert gen["created_utc"] == "2026-11-07T18:00:00Z"
    assert (gen["tts_backend"], gen["tts_backend_version"]) == ("fake-tts", "0.0")
    assert (gen["pad_seconds"], gen["snr_db"], gen["base_seed"]) == (1.0, 10.0, 1234)
    assert gen["noise_source"] == "pink"
    by_id = {m["id"]: m for m in messages}
    voices = assign_voices(SELECTED_IDS, VOICES)
    for clip in manifest["clips"]:
        assert clip["reference_text"] == by_id[clip["message_id"]]["message"]
        assert clip["sender_id"] == by_id[clip["message_id"]]["sender_id"]
        assert clip["voice"] == voices[clip["message_id"]]
        # 1 s tone + 2 x 1 s padding
        assert clip["duration_s"] == 3.0
    assert not list((root / "examples").glob("*.tmp"))


def test_end_to_end_clean_audio_is_padded_and_normalized(
    full_run: tuple[Path, dict[str, Any], FakeTTS],
) -> None:
    root = full_run[0]
    clean, sr = sf.read(root / "examples" / "audio" / "msg_001_clean.wav", dtype="float32")
    assert sr == 16000
    assert not clean[:16000].any()
    assert not clean[-16000:].any()
    assert float(np.max(np.abs(clean))) == pytest.approx(0.89, abs=1e-3)
    noisy, _ = sf.read(root / "examples" / "audio" / "msg_001_noisy.wav", dtype="float32")
    assert np.abs(noisy[:16000]).max() > 0


def test_second_run_reuses_tts_cache(tmp_path: Path, messages: list[dict[str, str]]) -> None:
    ids = SELECTED_IDS[:3]
    first = run_generate(messages, tmp_path, FakeTTS(), ids=ids)
    wav = tmp_path / "examples" / "audio" / "msg_002_noisy.wav"
    first_bytes = wav.read_bytes()
    tts = FakeTTS()
    second = run_generate(messages, tmp_path, tts, ids=ids)
    assert tts.calls == 0
    assert second == first
    assert wav.read_bytes() == first_bytes  # bit-reproducible noise and audio
    cache_files = sorted(p.name for p in (tmp_path / "examples" / "audio" / ".tts_cache").iterdir())
    assert cache_files == [f"{i}__{v}.wav" for i, v in zip(ids, VOICES * 2, strict=False)]


def test_force_ignores_tts_cache(tmp_path: Path, messages: list[dict[str, str]]) -> None:
    ids = SELECTED_IDS[:2]
    run_generate(messages, tmp_path, FakeTTS(), ids=ids)
    tts = FakeTTS()
    run_generate(messages, tmp_path, tts, ids=ids, settings=Settings(force=True))
    assert tts.calls == 2


def test_missing_id_stops_before_any_tts_call(
    tmp_path: Path, messages: list[dict[str, str]]
) -> None:
    tts = FakeTTS()
    without_msg_005 = [m for m in messages if m["id"] != "msg_005"]
    with pytest.raises(ValueError, match="msg_005"):
        run_generate(without_msg_005, tmp_path, tts)
    assert tts.calls == 0
    assert not (tmp_path / "examples" / "audio_manifest.json").exists()


def test_tts_failure_stops_naming_message_without_manifest(
    tmp_path: Path, messages: list[dict[str, str]]
) -> None:
    manifest_path = tmp_path / "examples" / "audio_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("old", encoding="utf-8")
    slept: list[float] = []
    tts = FakeTTS(failures=99)
    with pytest.raises(TTSError, match="msg_001"):
        run_generate(messages, tmp_path, tts, sleep=slept.append)
    assert tts.calls == 3
    assert slept == [2.0, 4.0]
    assert manifest_path.read_text(encoding="utf-8") == "old"


def test_tts_retry_succeeds_within_three_tries(
    tmp_path: Path, messages: list[dict[str, str]]
) -> None:
    slept: list[float] = []
    tts = FakeTTS(failures=2)
    manifest = run_generate(messages, tmp_path, tts, ids=["msg_001"], sleep=slept.append)
    assert len(manifest["clips"]) == 3
    assert slept == [2.0, 4.0]


def test_snr_off_target_stops(
    tmp_path: Path, messages: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_mix(
        speech: npt.NDArray[Any], noise: npt.NDArray[Any], snr_db: float, region: slice
    ) -> tuple[npt.NDArray[np.float32], float]:
        return np.asarray(speech, dtype=np.float32), snr_db + 0.11

    monkeypatch.setattr(make_audio, "mix_at_snr", bad_mix)
    with pytest.raises(SNRError, match=r"more than 0.1 dB"):
        run_generate(messages, tmp_path, FakeTTS(), ids=["msg_001"])
    assert not (tmp_path / "examples" / "audio_manifest.json").exists()


def test_snr_within_tolerance_is_accepted(
    tmp_path: Path, messages: list[dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def close_mix(
        speech: npt.NDArray[Any], noise: npt.NDArray[Any], snr_db: float, region: slice
    ) -> tuple[npt.NDArray[np.float32], float]:
        return np.asarray(speech, dtype=np.float32), snr_db + 0.09

    monkeypatch.setattr(make_audio, "mix_at_snr", close_mix)
    manifest = run_generate(messages, tmp_path, FakeTTS(), ids=["msg_001"])
    assert manifest["clips"][1]["snr_db_measured"] == 10.09


def test_generate_without_source_messages_and_other_settings(
    tmp_path: Path, messages: list[dict[str, str]]
) -> None:
    manifest = generate(
        messages,
        tts=FakeFileTTS(),
        noise=PinkNoiseSource(),
        out_dir=tmp_path / "audio",
        manifest_path=tmp_path / "m.json",
        settings=Settings(snr_db=20.0, pad_seconds=0.5, base_seed=7),
        repo_root=tmp_path,
        ids=["msg_031"],
        voices=["only-voice"],
        sleep=no_sleep,
    )
    gen = manifest["generator"]
    assert gen["source_messages"] is None
    assert gen["voices"] == ["only-voice"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", gen["created_utc"])
    noisy = manifest["clips"][1]
    assert noisy["snr_db_measured"] == pytest.approx(20.0, abs=0.1)
    assert noisy["noise_seed"] == clip_seed(7, "msg_031")
    assert (tmp_path / "audio" / ".tts_cache" / "msg_031__only-voice.mp3").exists()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_main_runs_with_fake_backend(
    tmp_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--messages",
        str(repo_root / "examples" / "synthetic_messages.json"),
        "--out-dir",
        str(tmp_path / "audio"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--snr-db",
        "15",
        "--pad-seconds",
        "0.5",
        "--seed",
        "99",
    ]
    assert make_audio.main(argv, tts_factory=FakeTTS) == 0
    out = capsys.readouterr().out
    assert "clips: 90" in out
    assert "total duration phone: 60.00 s" in out
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    gen = manifest["generator"]
    assert (gen["snr_db"], gen["pad_seconds"], gen["base_seed"]) == (15.0, 0.5, 99)
    assert gen["source_messages"] == "examples/synthetic_messages.json"
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 90


def test_main_with_noise_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    write_wav(noise_dir / "hum.wav", pink_noise(8000, 1) * np.float32(0.1), 16000)
    messages_file = tmp_path / "messages.json"
    messages_file.write_text(
        json.dumps([{"id": i, "sender_id": "w", "message": f"Text {i}."} for i in SELECTED_IDS]),
        encoding="utf-8",
    )
    argv = [
        "--messages",
        str(messages_file),
        "--out-dir",
        str(tmp_path / "audio"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--noise-dir",
        str(noise_dir),
        "--force",
    ]
    assert make_audio.main(argv, tts_factory=FakeTTS) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"]["noise_source"] == f"files:{noise_dir.as_posix()}"
    assert "measured SNR: min" in capsys.readouterr().out


def test_main_reports_error_and_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    messages_file = tmp_path / "messages.json"
    messages_file.write_text("[]", encoding="utf-8")
    argv = ["--messages", str(messages_file), "--out-dir", str(tmp_path / "audio")]
    argv += ["--manifest", str(tmp_path / "manifest.json")]
    assert make_audio.main(argv, tts_factory=FakeTTS) == 1
    assert "msg_001" in capsys.readouterr().err
    assert not (tmp_path / "manifest.json").exists()


def test_parse_args_defaults(repo_root: Path) -> None:
    args = make_audio.parse_args([])
    assert args.messages == repo_root / "examples" / "synthetic_messages.json"
    assert args.out_dir == repo_root / "examples" / "audio"
    assert args.manifest == repo_root / "examples" / "audio_manifest.json"
    assert (args.snr_db, args.pad_seconds, args.seed) == (10.0, 1.0, 1234)
    assert args.noise_dir is None
    assert args.force is False
    assert args.ids is None


def test_main_custom_ids_and_all(tmp_path: Path) -> None:
    messages_file = tmp_path / "custom_messages.json"
    messages_file.write_text(
        json.dumps(
            [
                {"id": "hf_001", "sender_id": "w1", "message": "First custom message."},
                {"id": "hf_002", "sender_id": "w2", "message": "Second custom message."},
            ]
        ),
        encoding="utf-8",
    )
    argv_one = [
        "--messages",
        str(messages_file),
        "--out-dir",
        str(tmp_path / "audio_one"),
        "--manifest",
        str(tmp_path / "manifest_one.json"),
        "--ids",
        "hf_002",
    ]
    assert make_audio.main(argv_one, tts_factory=FakeTTS) == 0
    m1 = json.loads((tmp_path / "manifest_one.json").read_text(encoding="utf-8"))
    assert [c["message_id"] for c in m1["clips"]] == ["hf_002"] * 3

    argv_all = [
        "--messages",
        str(messages_file),
        "--out-dir",
        str(tmp_path / "audio_all"),
        "--manifest",
        str(tmp_path / "manifest_all.json"),
        "--ids",
        "all",
    ]
    assert make_audio.main(argv_all, tts_factory=FakeTTS) == 0
    m2 = json.loads((tmp_path / "manifest_all.json").read_text(encoding="utf-8"))
    assert len(m2["clips"]) == 6
    assert {c["message_id"] for c in m2["clips"]} == {"hf_001", "hf_002"}


# --------------------------------------------------------------------------------------
# Network (slow)
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_real_edge_tts_call() -> None:
    wave, sr = EdgeTTSBackend().synthesize("We screened 48 women in Nakuru.", VOICES[0])
    assert sr > 0
    assert wave.size > sr  # more than one second of audio
    assert float(np.max(np.abs(wave))) > 0.01
