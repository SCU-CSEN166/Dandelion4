"""M1: generate the test-audio clips and their manifest (design §3).

30 selected synthetic messages are rendered with text-to-speech and saved in three
versions: ``clean`` (16 kHz), ``noisy`` (16 kHz, noise at a target SNR) and ``phone``
(8 kHz). ``examples/audio_manifest.json`` records each clip's ground truth and the
generation settings.

Layout (design §1.2):

* core functions: pure, on numpy arrays / strings / dicts;
* adapters: ``EdgeTTSBackend``, ``PinkNoiseSource``, ``FileNoiseSource``, ``TTSCache``,
  WAV and JSON I/O;
* ``generate``: the orchestration, with every adapter passed in;
* ``main``: argument parsing that wires the real adapters to ``generate``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]

SELECTED_REPORT_IDS = [
    "msg_001", "msg_002", "msg_003", "msg_004", "msg_005", "msg_006",
    "msg_007", "msg_008", "msg_010", "msg_011", "msg_012", "msg_013",
    "msg_015", "msg_016", "msg_017", "msg_018", "msg_019", "msg_021",
    "msg_023", "msg_024", "msg_025", "msg_028", "msg_029", "msg_030",
]  # fmt: skip
SELECTED_CHAT_IDS = [
    "msg_031", "msg_032", "msg_033", "msg_034", "msg_036", "msg_038",
]  # fmt: skip
SELECTED_IDS = SELECTED_REPORT_IDS + SELECTED_CHAT_IDS

VOICES = ["en-KE-AsiliaNeural", "en-KE-ChilembaNeural"]  # Kenyan English, female / male
VERSIONS = ("clean", "noisy", "phone")

SR_16K = 16000
SR_PHONE = 8000
PEAK = 0.89
CLIP_LIMIT = 0.99
SNR_TOLERANCE_DB = 0.1

TTS_TRIES = 3
TTS_BACKOFF_S = 2.0

Wave = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.floating[Any]]
Message = dict[str, str]
WavWriter = Callable[[Path, Wave, int], None]


class TTSError(RuntimeError):
    """Text-to-speech failed for a message after all retries."""


class SNRError(RuntimeError):
    """A measured SNR is further from the target than ``SNR_TOLERANCE_DB``."""


# --------------------------------------------------------------------------------------
# Core functions (pure)
# --------------------------------------------------------------------------------------


def select_messages(messages: list[Message], ids: list[str]) -> list[Message]:
    """Return the records with the given IDs, in ``ids`` order.

    Raises ``ValueError`` listing duplicate IDs (in ``ids`` or in ``messages``) or IDs
    missing from ``messages``.
    """
    duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate selected IDs: {', '.join(duplicate_ids)}")
    by_id: dict[str, Message] = {}
    duplicate_records: set[str] = set()
    for record in messages:
        if record["id"] in by_id:
            duplicate_records.add(record["id"])
        by_id[record["id"]] = record
    wanted_duplicates = sorted(duplicate_records.intersection(ids))
    if wanted_duplicates:
        raise ValueError(f"duplicate message records: {', '.join(wanted_duplicates)}")
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"selected IDs missing from the messages: {', '.join(missing)}")
    return [by_id[i] for i in ids]


def assign_voices(ids: list[str], voices: list[str]) -> dict[str, str]:
    """Assign ``voices`` to ``ids`` alternately, in input order (``voices[i % len]``)."""
    if not voices:
        raise ValueError("at least one voice is required")
    return {message_id: voices[i % len(voices)] for i, message_id in enumerate(ids)}


def to_mono_16k(wave: FloatArray, sr: int) -> Wave:
    """Average channels (shape ``(frames, channels)``) and resample to 16 kHz, float32."""
    mono = np.asarray(wave, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1, dtype=np.float32)
    elif mono.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {mono.shape}")
    if sr != SR_16K:
        mono = np.asarray(librosa.resample(mono, orig_sr=sr, target_sr=SR_16K))
    return mono.astype(np.float32, copy=False)


def peak_normalize(wave: FloatArray, peak: float = PEAK) -> Wave:
    """Scale so that ``max(abs(wave)) == peak``; an all-zero input is returned unchanged."""
    out = np.asarray(wave, dtype=np.float32)
    current = float(np.max(np.abs(out))) if out.size else 0.0
    if current == 0.0:
        return out
    return (out * np.float32(peak / current)).astype(np.float32)


def pad(wave: FloatArray, sr: int, seconds: float) -> Wave:
    """Add ``round(seconds * sr)`` zeros at each end."""
    n = round(seconds * sr)
    return np.pad(np.asarray(wave, dtype=np.float32), (n, n)).astype(np.float32)


def pink_noise(n: int, seed: int) -> Wave:
    """Unit-RMS pink (1/f power) noise; the same ``(n, seed)`` gives bit-identical output.

    White Gaussian noise from ``np.random.default_rng(seed)`` is shaped in the frequency
    domain by ``1/sqrt(f)`` with the DC bin set to 0.
    """
    if n < 2:
        raise ValueError(f"pink noise needs at least 2 samples, got {n}")
    white = np.random.default_rng(seed).standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    shaping = np.zeros_like(freqs)
    shaping[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * shaping, n)
    rms = float(np.sqrt(np.mean(shaped**2)))
    return (shaped / rms).astype(np.float32)


def mix_at_snr(
    speech: FloatArray, noise: FloatArray, snr_db: float, speech_region: slice
) -> tuple[Wave, float]:
    """Add ``noise`` to ``speech`` at ``snr_db`` (design §3.6).

    Speech power is measured over ``speech_region`` only; the noise covers the whole clip.
    If the mix would exceed 0.99 in magnitude, both parts are scaled together, which keeps
    the SNR. Returns ``(mixed, measured_snr_db)``, measured on the final components.
    """
    s = np.asarray(speech, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    if s.shape != n.shape:
        raise ValueError(f"speech and noise lengths differ: {s.shape} vs {n.shape}")
    p_s = float(np.mean(s[speech_region] ** 2))
    p_n = float(np.mean(n**2))
    if p_s == 0.0:
        raise ValueError("speech region has zero power")
    if p_n == 0.0:
        raise ValueError("noise has zero power")
    gain = np.sqrt(p_s / (p_n * 10.0 ** (snr_db / 10.0)))
    noise_part = gain * n
    mixed = s + noise_part
    peak = float(np.max(np.abs(mixed)))
    if peak > CLIP_LIMIT:
        scale = CLIP_LIMIT / peak
        mixed *= scale
        s = s * scale
        noise_part *= scale
    measured = 10.0 * np.log10(np.mean(s[speech_region] ** 2) / np.mean(noise_part**2))
    return mixed.astype(np.float32), float(measured)


def to_phone(wave_16k: FloatArray) -> Wave:
    """Resample 16 kHz audio to 8 kHz (the phone condition)."""
    out = librosa.resample(
        np.asarray(wave_16k, dtype=np.float32), orig_sr=SR_16K, target_sr=SR_PHONE
    )
    return np.asarray(out, dtype=np.float32)


def clip_seed(base_seed: int, message_id: str) -> int:
    """Per-message noise seed: ``(base_seed + crc32(message_id)) % 2**32``."""
    return (base_seed + zlib.crc32(message_id.encode())) % 2**32


def build_manifest_entry(
    *,
    message: Message,
    version: str,
    voice: str,
    path: str,
    sample_rate: int,
    n_samples: int,
    snr_db_measured: float | None = None,
    noise_seed: int | None = None,
) -> dict[str, Any]:
    """Build one ``clips`` entry of the manifest (design §2.4), keys in contract order.

    ``duration_s`` is rounded to 2 decimals (10 ms) and ``snr_db_measured`` to 3.
    """
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}")
    return {
        "clip_id": f"{message['id']}_{version}",
        "message_id": message["id"],
        "sender_id": message["sender_id"],
        "version": version,
        "voice": voice,
        "path": path,
        "sample_rate": sample_rate,
        "duration_s": round(n_samples / sample_rate, 2),
        "reference_text": message["message"],
        "snr_db_measured": None if snr_db_measured is None else round(snr_db_measured, 3),
        "noise_seed": noise_seed,
    }


def repo_relative(path: Path, repo_root: Path) -> str:
    """``path`` relative to ``repo_root`` with forward slashes (absolute if outside it)."""
    resolved = path.resolve()
    root = repo_root.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return resolved.as_posix()


def format_utc(moment: datetime) -> str:
    """Format a datetime as ``2026-11-07T18:00:00Z`` (converted to UTC)."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def summarize(manifest: dict[str, Any]) -> str:
    """Clip count, total duration per version and min / max measured SNR, as text."""
    clips: list[dict[str, Any]] = manifest["clips"]
    lines = [f"clips: {len(clips)}"]
    for version in VERSIONS:
        total = sum(c["duration_s"] for c in clips if c["version"] == version)
        lines.append(f"total duration {version}: {total:.2f} s")
    snrs = [c["snr_db_measured"] for c in clips if c["snr_db_measured"] is not None]
    if snrs:
        lines.append(f"measured SNR: min {min(snrs):.3f} dB, max {max(snrs):.3f} dB")
    else:
        lines.append("measured SNR: no noisy clips")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Adapter interfaces (design §3.7)
# --------------------------------------------------------------------------------------


class TTSBackend(Protocol):
    """Text-to-speech engine returning ``(waveform, sample_rate)``."""

    name: str
    version: str

    def synthesize(self, text: str, voice: str) -> tuple[npt.NDArray[Any], int]:
        """Synthesize speech audio and sample rate for ``text`` using ``voice``."""
        ...


@runtime_checkable
class FileTTSBackend(Protocol):
    """Optional extension: a backend that natively produces an encoded file (e.g. MP3).

    ``TTSCache`` stores that file as-is, so the cache holds the backend's raw output.
    """

    name: str
    version: str
    file_suffix: str

    def synthesize(self, text: str, voice: str) -> tuple[npt.NDArray[Any], int]:
        """Synthesize speech audio and sample rate for ``text`` using ``voice``."""
        ...

    def synthesize_to_file(self, text: str, voice: str, path: Path) -> None:
        """Synthesize ``text`` with ``voice`` directly to ``path``."""
        ...


class NoiseSource(Protocol):
    """Noise generator for the noisy version."""

    name: str

    def get(self, n_samples: int, sr: int, seed: int) -> npt.NDArray[Any]:
        """Return ``n_samples`` of noise at sample rate ``sr`` for ``seed``."""
        ...


# --------------------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------------------


def package_version(distribution: str) -> str:
    """Installed version of ``distribution``, or ``"unknown"`` if it is not installed."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def decode_audio_file(path: Path) -> tuple[Wave, int]:
    """Decode an audio file with ``soundfile``, falling back to ``librosa.load``."""
    try:
        data, sr = sf.read(path, dtype="float32")
    except RuntimeError:  # soundfile.LibsndfileError subclasses RuntimeError
        loaded, loaded_sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        return np.asarray(loaded, dtype=np.float32), int(loaded_sr)
    return np.asarray(data, dtype=np.float32), int(sr)


def write_wav(path: Path, wave: Wave, sr: int) -> None:
    """Write mono 16-bit PCM WAV."""
    sf.write(path, wave, sr, subtype="PCM_16")


class EdgeTTSBackend:
    """Microsoft Edge online TTS via the ``edge-tts`` package (imported lazily)."""

    name = "edge-tts"
    file_suffix = ".mp3"

    def __init__(self) -> None:
        self.version = package_version("edge-tts")

    def synthesize_to_file(self, text: str, voice: str, path: Path) -> None:  # pragma: no cover
        """Synthesize ``text`` and save the raw MP3 at ``path`` (network call)."""
        import edge_tts

        asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))

    def synthesize(self, text: str, voice: str) -> tuple[npt.NDArray[Any], int]:
        """Synthesize to a temporary MP3 and decode it."""
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / f"tts{self.file_suffix}"
            self.synthesize_to_file(text, voice, mp3)
            return decode_audio_file(mp3)


class PinkNoiseSource:
    """Seeded pink noise (the default); ``sr`` does not affect the samples."""

    name = "pink"

    def get(self, n_samples: int, sr: int, seed: int) -> npt.NDArray[Any]:
        """Generate ``n_samples`` of deterministic pink noise."""
        return pink_noise(n_samples, seed)


class FileNoiseSource:
    """Noise from WAV files in a directory (design §3.7).

    File ``sorted(files)[seed % len(files)]`` is converted to mono, resampled to ``sr``,
    and read cyclically (looped or cropped) from a seed-chosen start offset.
    """

    def __init__(
        self,
        directory: Path,
        reader: Callable[[Path], tuple[npt.NDArray[Any], int]] = decode_audio_file,
    ) -> None:
        self.files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".wav")
        if not self.files:
            raise ValueError(f"no .wav noise files in {directory}")
        self.name = f"files:{directory.as_posix()}"
        self._reader = reader

    def get(self, n_samples: int, sr: int, seed: int) -> npt.NDArray[Any]:
        """Slice or loop ``n_samples`` of noise from a pre-recorded file."""
        source = self.files[seed % len(self.files)]
        data, file_sr = self._reader(source)
        wave = np.asarray(data, dtype=np.float32)
        if wave.ndim == 2:
            wave = wave.mean(axis=1, dtype=np.float32)
        if file_sr != sr:
            wave = np.asarray(librosa.resample(wave, orig_sr=file_sr, target_sr=sr))
        if wave.size == 0:
            raise ValueError(f"noise file {source} is empty")
        offset = int(np.random.default_rng(seed).integers(0, wave.size))
        index = (offset + np.arange(n_samples)) % wave.size
        return wave[index].astype(np.float32)


def synthesize_with_retry(
    call: Callable[[], tuple[npt.NDArray[Any], int]],
    message_id: str,
    *,
    tries: int = TTS_TRIES,
    backoff_s: float = TTS_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[npt.NDArray[Any], int]:
    """Run ``call`` up to ``tries`` times, sleeping ``backoff_s * 2**k`` between tries.

    Raises ``TTSError`` naming ``message_id`` if every try fails.
    """
    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            return call()
        except Exception as error:
            # Intentional resilience boundary: TTS backends raise network, voice, or decode errors.
            last_error = error
            if attempt < tries - 1:
                sleep(backoff_s * 2**attempt)
    raise TTSError(
        f"TTS failed for message {message_id} after {tries} tries: {last_error!r}"
    ) from last_error


class TTSCache:
    """Per-``(message_id, voice)`` cache of raw TTS output around any ``TTSBackend``.

    A ``FileTTSBackend`` (such as ``EdgeTTSBackend``) writes its encoded file straight
    into ``<cache_dir>/<message_id>__<voice><file_suffix>`` (the raw MP3 for edge-tts).
    Any other backend's waveform is stored losslessly as a float32 WAV. A cached file is
    decoded instead of calling the backend; ``force=True`` re-synthesizes and overwrites.
    Cache files are written to a temporary name and renamed, so a failed call leaves
    nothing behind.
    """

    def __init__(
        self,
        backend: TTSBackend,
        cache_dir: Path,
        *,
        force: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.cache_dir = cache_dir
        self.force = force
        self._sleep = sleep

    def path_for(self, message_id: str, voice: str) -> Path:
        """Return the cache file path for ``message_id`` and ``voice``."""
        suffix = backend_suffix(self.backend)
        return self.cache_dir / f"{message_id}__{voice}{suffix}"

    def get(self, message_id: str, text: str, voice: str) -> tuple[Wave, int]:
        """Load cached TTS audio or synthesize and cache it on miss."""
        path = self.path_for(message_id, voice)
        if path.exists() and not self.force:
            return decode_audio_file(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.partial{path.suffix}")
        backend = self.backend
        try:
            if isinstance(backend, FileTTSBackend):

                def call() -> tuple[npt.NDArray[Any], int]:
                    backend.synthesize_to_file(text, voice, partial)
                    return decode_audio_file(partial)

                wave, sr = synthesize_with_retry(call, message_id, sleep=self._sleep)
            else:
                wave, sr = synthesize_with_retry(
                    lambda: backend.synthesize(text, voice), message_id, sleep=self._sleep
                )
                sf.write(partial, np.asarray(wave, dtype=np.float32), sr, subtype="FLOAT")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        return np.asarray(wave, dtype=np.float32), sr


def backend_suffix(backend: TTSBackend) -> str:
    """Cache-file suffix: the backend's native one, or ``.wav`` for array-only backends."""
    if isinstance(backend, FileTTSBackend):
        return backend.file_suffix
    return ".wav"


def load_messages(path: Path) -> list[Message]:
    """Read the messages JSON file."""
    with path.open(encoding="utf-8") as handle:
        data: list[Message] = json.load(handle)
    return data


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON (UTF-8, ``indent=2``) to a temporary file, then rename it over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Generation settings (CLI flags)."""

    snr_db: float = 10.0
    pad_seconds: float = 1.0
    base_seed: int = 1234
    force: bool = False


def render_message(
    wave: npt.NDArray[Any], sr: int, settings: Settings, noise: NoiseSource, seed: int
) -> tuple[dict[str, Wave], float]:
    """Turn one TTS rendering into the three versions (design §3.4).

    Returns ``({version: wave}, measured_snr_db)``; raises ``SNRError`` if the measured
    SNR is more than ``SNR_TOLERANCE_DB`` from the target.
    """
    speech = peak_normalize(to_mono_16k(wave, sr))
    clean = pad(speech, SR_16K, settings.pad_seconds)
    n_pad = round(settings.pad_seconds * SR_16K)
    region = slice(n_pad, n_pad + speech.size)
    noise_wave = noise.get(clean.size, SR_16K, seed)
    noisy, measured = mix_at_snr(clean, noise_wave, settings.snr_db, region)
    if not abs(measured - settings.snr_db) <= SNR_TOLERANCE_DB:
        raise SNRError(
            f"measured SNR {measured:.4f} dB differs from target {settings.snr_db} dB "
            f"by more than {SNR_TOLERANCE_DB} dB"
        )
    return {"clean": clean, "noisy": noisy, "phone": to_phone(clean)}, measured


def _render_and_write_message_clips(
    message: Message,
    *,
    voice: str,
    cache: TTSCache,
    noise: NoiseSource,
    settings: Settings,
    out_dir: Path,
    repo_root: Path,
    wav_writer: WavWriter,
) -> list[dict[str, Any]]:
    """Synthesize, render, write WAVs, and build manifest entries for one message."""
    message_id = message["id"]
    wave, sr = cache.get(message_id, message["message"], voice)
    seed = clip_seed(settings.base_seed, message_id)
    rendered, measured = render_message(wave, sr, settings, noise, seed)
    entries: list[dict[str, Any]] = []
    for version in VERSIONS:
        clip_wave = rendered[version]
        rate = SR_PHONE if version == "phone" else SR_16K
        wav_path = out_dir / f"{message_id}_{version}.wav"
        wav_writer(wav_path, clip_wave, rate)
        is_noisy = version == "noisy"
        entries.append(
            build_manifest_entry(
                message=message,
                version=version,
                voice=voice,
                path=repo_relative(wav_path, repo_root),
                sample_rate=rate,
                n_samples=clip_wave.size,
                snr_db_measured=measured if is_noisy else None,
                noise_seed=seed if is_noisy else None,
            )
        )
    return entries


def _build_audio_manifest(
    *,
    created: str,
    tts: TTSBackend,
    voices: Sequence[str],
    settings: Settings,
    noise: NoiseSource,
    source_messages: Path | None,
    repo_root: Path,
    clips: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the ``audio_manifest.json`` dictionary for generated clips."""
    return {
        "generator": {
            "script": "src/training/make_audio.py",
            "created_utc": created,
            "tts_backend": tts.name,
            "tts_backend_version": tts.version,
            "voices": list(voices),
            "pad_seconds": float(settings.pad_seconds),
            "snr_db": float(settings.snr_db),
            "noise_source": noise.name,
            "base_seed": settings.base_seed,
            "source_messages": (
                None if source_messages is None else repo_relative(source_messages, repo_root)
            ),
        },
        "clips": clips,
    }


def generate(
    messages: list[Message],
    *,
    tts: TTSBackend,
    noise: NoiseSource,
    out_dir: Path,
    manifest_path: Path,
    settings: Settings,
    repo_root: Path = REPO_ROOT,
    source_messages: Path | None = None,
    ids: list[str] | None = None,
    voices: list[str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] | None = None,
    wav_writer: WavWriter = write_wav,
) -> dict[str, Any]:
    """Generate all clips and write the manifest last (atomically). Returns the manifest.

    Every selected ID is checked before any TTS call. TTS output is cached in
    ``<out_dir>/.tts_cache``.
    """
    ids = SELECTED_IDS if ids is None else ids
    voices = VOICES if voices is None else voices
    selected = select_messages(messages, ids)
    voice_of = assign_voices(ids, voices)
    cache = TTSCache(tts, out_dir / ".tts_cache", force=settings.force, sleep=sleep)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = format_utc((now or (lambda: datetime.now(UTC)))())

    clips: list[dict[str, Any]] = []
    for message in selected:
        clips.extend(
            _render_and_write_message_clips(
                message,
                voice=voice_of[message["id"]],
                cache=cache,
                noise=noise,
                settings=settings,
                out_dir=out_dir,
                repo_root=repo_root,
                wav_writer=wav_writer,
            )
        )
    clips.sort(key=lambda c: (c["message_id"], VERSIONS.index(c["version"])))

    manifest = _build_audio_manifest(
        created=created,
        tts=tts,
        voices=voices,
        settings=settings,
        noise=noise,
        source_messages=source_messages,
        repo_root=repo_root,
        clips=clips,
    )
    write_json_atomic(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for synthetic audio dataset generation."""
    examples = REPO_ROOT / "examples"
    parser = argparse.ArgumentParser(description="Generate the test-audio clips (M1).")
    parser.add_argument("--messages", type=Path, default=examples / "synthetic_messages.json")
    parser.add_argument("--out-dir", type=Path, default=examples / "audio")
    parser.add_argument("--manifest", type=Path, default=examples / "audio_manifest.json")
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--pad-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-dir", type=Path, default=None, help="default: pink noise")
    parser.add_argument("--force", action="store_true", help="ignore the TTS cache")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="message IDs to voice, or 'all' for all in --messages (default: 30 Week 7 IDs)",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    tts_factory: Callable[[], TTSBackend] = EdgeTTSBackend,
) -> int:
    """CLI entry point; returns the process exit code."""
    args = parse_args(argv)
    settings = Settings(
        snr_db=args.snr_db, pad_seconds=args.pad_seconds, base_seed=args.seed, force=args.force
    )
    try:
        noise: NoiseSource = (
            PinkNoiseSource() if args.noise_dir is None else FileNoiseSource(args.noise_dir)
        )
        messages = load_messages(args.messages)
        if args.ids == ["all"]:
            ids: list[str] | None = [m["id"] for m in messages]
        else:
            ids = args.ids
        manifest = generate(
            messages,
            tts=tts_factory(),
            noise=noise,
            out_dir=args.out_dir,
            manifest_path=args.manifest,
            settings=settings,
            source_messages=args.messages,
            ids=ids,
        )
    except (ValueError, TTSError, SNRError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(summarize(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
