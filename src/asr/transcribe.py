"""M2 - Transcriber: turn the manifest's audio clips into transcript files with Whisper.

Reads the clips of one or all versions (clean, noisy, phone) from the audio manifest,
transcribes them in English with greedy decoding, and writes for each version:

- ``<out-dir>/transcripts_<version>.json``: the v2 message format
  (``id``, ``sender_id``, ``source``, ``message``), with ``source="audio"``.
- ``<out-dir>/transcripts_<version>.meta.json``: model and runtime facts for the run.

The core functions contain no model code. ``torch`` and ``transformers`` are imported
only inside ``WhisperASR`` and ``resolve_device``.
"""

import argparse
import importlib.metadata
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import librosa
import numpy as np
import numpy.typing as npt
import soundfile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "examples" / "audio_manifest.json"
DEFAULT_OUT_DIR = REPO_ROOT / "examples"
BASE_MODEL = "openai/whisper-large-v3-turbo"
FINETUNED_MODEL_DIR = REPO_ROOT / "models" / "whisper-large-v3-turbo-v3"


def resolve_default_model(model_dir: Path | None = None) -> str:
    """Return the local fine-tuned checkpoint path when present, else BASE_MODEL."""
    target = FINETUNED_MODEL_DIR if model_dir is None else model_dir
    if (target / "config.json").is_file():
        return str(target)
    return BASE_MODEL


DEFAULT_MODEL = resolve_default_model()
DEFAULT_BATCH_SIZE = 8

VERSIONS: tuple[str, ...] = ("clean", "noisy", "phone")
ASR_SAMPLE_RATE = 16_000
LANGUAGE = "en"
TASK = "transcribe"
DECODING: dict[str, Any] = {"num_beams": 1, "do_sample": False}

# Keys WhisperASR.metadata() supplies, in the order they appear in the .meta.json file.
ASR_METADATA_KEYS: tuple[str, ...] = (
    "model",
    "language",
    "task",
    "decoding",
    "initial_prompt",
    "device",
    "dtype",
    "torch_version",
    "transformers_version",
)

FloatArray = npt.NDArray[np.float32]
Clip = dict[str, Any]
Record = dict[str, str]
ReadWav = Callable[[Path], tuple[npt.NDArray[Any], int]]
ASR = Callable[[list[FloatArray]], list[str]]


class ASRBackend(Protocol):
    """What the orchestration needs from a speech recognizer."""

    def __call__(self, waves: list[FloatArray]) -> list[str]:
        """Transcribe 16 kHz float32 mono waves, returning one text per wave."""
        ...

    def load(self) -> None:
        """Load the model now (idempotent), so loading is not timed as transcription."""
        ...

    def metadata(self) -> dict[str, Any]:
        """Return the model and runtime fields named in ASR_METADATA_KEYS."""
        ...


# --------------------------------------------------------------------------- core


def clips_for_version(manifest: dict[str, Any], version: str) -> list[Clip]:
    """Return the manifest clips with that version, sorted by message_id.

    Raises ValueError if there are none, or if a message_id appears twice.
    """
    clips = [clip for clip in manifest.get("clips", []) if clip.get("version") == version]
    if not clips:
        raise ValueError(f"manifest has no clips for version {version!r}")
    clips.sort(key=lambda clip: str(clip["message_id"]))
    ids = [clip["message_id"] for clip in clips]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"version {version!r} has duplicate message_ids: {duplicates}")
    return clips


def load_for_asr(wave: npt.NDArray[Any], sr: int) -> FloatArray:
    """Return the wave as float32 mono at 16 kHz.

    Multi-channel input (frames x channels, as soundfile returns it) is averaged to mono.
    Phone clips (8 kHz) are upsampled here; what the downsampling removed stays removed.
    """
    mono = np.asarray(wave, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1, dtype=np.float32)
    elif mono.ndim != 1:
        raise ValueError(f"expected a 1-D or 2-D wave, got shape {mono.shape}")
    if sr != ASR_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def to_transcript_records(clips: Sequence[Clip], texts: Sequence[str]) -> list[Record]:
    """Zip clips with texts into {"id", "sender_id", "source": "audio", "message": text.strip()}.

    Raises ValueError if the lengths differ.
    """
    if len(clips) != len(texts):
        raise ValueError(f"got {len(texts)} transcripts for {len(clips)} clips")
    return [
        {
            "id": str(clip["message_id"]),
            "sender_id": str(clip["sender_id"]),
            "source": "audio",
            "message": text.strip(),
        }
        for clip, text in zip(clips, texts, strict=True)
    ]


def clip_path(clip: Clip, root: Path = REPO_ROOT) -> Path:
    """Resolve a clip's repo-root-relative POSIX path against root."""
    return root.joinpath(*PurePosixPath(str(clip["path"])).parts)


def transcribe_clips(
    clips: Sequence[Clip], read_wav: ReadWav, asr: ASR, root: Path = REPO_ROOT
) -> list[Record]:
    """Read every clip, convert it to 16 kHz float32 mono, and transcribe the list in one call.

    ``read_wav`` reads audio and ``asr`` transcribes it; this function has no model code.
    """
    waves = [load_for_asr(*read_wav(clip_path(clip, root))) for clip in clips]
    texts = asr(waves)
    return to_transcript_records(clips, texts)


def missing_wavs(clips: Sequence[Clip], root: Path = REPO_ROOT) -> list[Path]:
    """Return the paths of the clips whose WAV file does not exist."""
    return [path for path in (clip_path(clip, root) for clip in clips) if not path.is_file()]


def build_metadata(
    version: str,
    asr_metadata: dict[str, Any],
    n_clips: int,
    audio_seconds_total: float,
    wall_seconds: float,
    created_utc: datetime,
) -> dict[str, Any]:
    """Assemble the .meta.json record with keys in the contract order (design §2.6)."""
    meta: dict[str, Any] = {"version": version}
    for key in ASR_METADATA_KEYS:
        meta[key] = asr_metadata[key]
    meta["n_clips"] = n_clips
    meta["audio_seconds_total"] = round(audio_seconds_total, 2)
    meta["wall_seconds"] = round(wall_seconds, 2)
    meta["created_utc"] = created_utc.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return meta


# ----------------------------------------------------------------------- adapters


CONTAINER_AUDIO_SUFFIXES = frozenset({".mp4", ".m4a", ".aac", ".mov", ".mp3", ".webm", ".opus"})
WHISPER_CHUNK_SECONDS = 30
WHISPER_STRIDE_SECONDS = (4, 2)


def read_wav(path: Path) -> tuple[npt.NDArray[Any], int]:
    """Read a WAV (or container audio/video file via PyAV) as float32 samples and sample rate."""
    if path.suffix.lower() in CONTAINER_AUDIO_SUFFIXES:
        import av

        resampler = av.AudioResampler(format="flt", layout="mono", rate=ASR_SAMPLE_RATE)
        chunks: list[npt.NDArray[np.float32]] = []
        with av.open(str(path)) as container:
            stream = next(
                (s for s in container.streams if s.type == "audio"),
                None,
            )
            if stream is None:
                raise ValueError(f"no audio stream found in {path}")
            for frame in container.decode(stream):
                if not isinstance(frame, av.AudioFrame):
                    continue
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().reshape(-1).astype(np.float32)
                    if arr.size:
                        chunks.append(arr)
            for resampled in resampler.resample(None):
                arr = resampled.to_ndarray().reshape(-1).astype(np.float32)
                if arr.size:
                    chunks.append(arr)
        wave = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return wave, ASR_SAMPLE_RATE
    data, sample_rate = soundfile.read(path, dtype="float32")
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def write_json(path: Path, data: object) -> None:
    """Write UTF-8 JSON with indent=2 and ensure_ascii=False, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _import_contracts() -> Any:
    """Import ``src.core.contracts`` (falling back to ``contracts`` if run standalone)."""
    import importlib

    try:
        return importlib.import_module("src.core.contracts")
    except ModuleNotFoundError:  # pragma: no cover
        return importlib.import_module("contracts")


def validate_transcript_records(records: Sequence[Record]) -> list[str]:
    """Return Interface Spec v2 ``transcripts`` validation problems (empty if valid)."""
    problems = _import_contracts().validate_file("transcripts", list(records))
    return [str(problem) for problem in problems]


def write_transcripts(path: Path, records: Sequence[Record]) -> None:
    """Validate and write transcript records in Interface Spec v2 Message format."""
    ordered: list[Record] = []
    for r in sorted(records, key=lambda rec: str(rec.get("id", ""))):
        item: Record = {
            "id": r["id"],
            "sender_id": r["sender_id"],
            "source": r.get("source", "audio"),
            "message": r["message"],
        }
        if "asr_text" in r:
            item["asr_text"] = r["asr_text"]
        ordered.append(item)
    problems = validate_transcript_records(ordered)
    if problems:
        raise ValueError("Invalid transcript records:\n" + "\n".join(problems))
    write_json(path, ordered)


def load_manifest(path: Path) -> dict[str, Any]:
    """Read the audio manifest (design §2.4)."""
    with path.open(encoding="utf-8") as handle:
        data: object = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        raise ValueError(f"{path} is not an audio manifest (no 'clips' list)")
    return data


def resolve_device(requested: str, cuda_available: Callable[[], bool] | None = None) -> str:
    """Map "auto" to "cuda:0" if CUDA is available, else "cpu"; pass other values through."""
    if requested != "auto":
        return requested
    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available
    return "cuda:0" if cuda_available() else "cpu"


def _package_version(name: str) -> str:
    """Installed version of a distribution, read without importing it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


class WhisperASR:
    """Whisper through the transformers ASR pipeline, built lazily on first use.

    English mode, transcription task, greedy decoding (num_beams=1, no sampling),
    no timestamps. float16 on CUDA, float32 otherwise.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        initial_prompt: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.initial_prompt = initial_prompt
        self._pipe: Any = None
        self._chunked_pipe: Any = None

    @property
    def dtype_name(self) -> str:
        """Name of the weight dtype used on this device."""
        return "float16" if self.device.startswith("cuda") else "float32"

    @property
    def loaded(self) -> bool:
        """Whether the pipeline has been built."""
        return self._pipe is not None

    def load(self) -> None:
        """Build the pipeline if it has not been built yet."""
        if self._pipe is None:  # pragma: no cover - needs the model weights
            import torch
            from transformers import pipeline

            dtype = torch.float16 if self.dtype_name == "float16" else torch.float32
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                dtype=dtype,
                device=self.device,
            )

    def _get_chunked_pipe(self) -> Any:  # pragma: no cover - long-audio Whisper helper
        """Build a chunked pipeline reusing the loaded model for clips >30 s."""
        if self._chunked_pipe is None:
            from transformers import pipeline

            self._chunked_pipe = pipeline(
                "automatic-speech-recognition",
                model=self._pipe.model,
                tokenizer=self._pipe.tokenizer,
                feature_extractor=self._pipe.feature_extractor,
                device=self.device,
                chunk_length_s=WHISPER_CHUNK_SECONDS,
                stride_length_s=WHISPER_STRIDE_SECONDS,
                ignore_warning=True,
            )
        return self._chunked_pipe

    def __call__(self, waves: list[FloatArray]) -> list[str]:  # pragma: no cover - real model
        """Transcribe 16 kHz float32 mono waves, returning one text per wave."""
        if not waves:
            return []
        self.load()
        assert self._pipe is not None
        generate_kwargs: dict[str, Any] = {"language": LANGUAGE, "task": TASK, **DECODING}
        if self.initial_prompt:
            generate_kwargs["prompt_ids"] = self._pipe.tokenizer.get_prompt_ids(
                self.initial_prompt, return_tensors="pt"
            )
        max_short = WHISPER_CHUNK_SECONDS * ASR_SAMPLE_RATE
        short_indices = [i for i, wave in enumerate(waves) if len(wave) <= max_short]
        long_indices = [i for i, wave in enumerate(waves) if len(wave) > max_short]

        texts = [""] * len(waves)
        if short_indices:
            outputs = self._pipe(
                [{"raw": waves[i], "sampling_rate": ASR_SAMPLE_RATE} for i in short_indices],
                batch_size=self.batch_size,
                return_timestamps=False,
                generate_kwargs=generate_kwargs,
            )
            if len(outputs) != len(short_indices):
                raise RuntimeError(
                    f"pipeline returned {len(outputs)} results for {len(short_indices)} clips"
                )
            for idx, output in zip(short_indices, outputs, strict=True):
                texts[idx] = str(output["text"])

        if long_indices:
            chunked_pipe = self._get_chunked_pipe()
            for idx in long_indices:
                output = chunked_pipe(
                    {"array": waves[idx], "sampling_rate": ASR_SAMPLE_RATE},
                    batch_size=self.batch_size,
                    generate_kwargs=generate_kwargs,
                )
                texts[idx] = str(output["text"])
        return texts

    def metadata(self) -> dict[str, Any]:
        """Model and runtime fields for the .meta.json file (design §2.6)."""
        return {
            "model": self.model_id,
            "language": LANGUAGE,
            "task": TASK,
            "decoding": dict(DECODING),
            "initial_prompt": self.initial_prompt,
            "device": self.device,
            "dtype": self.dtype_name,
            "torch_version": _package_version("torch"),
            "transformers_version": _package_version("transformers"),
        }


# -------------------------------------------------------------------- orchestration


def transcribe_versions(
    manifest: dict[str, Any],
    versions: Sequence[str],
    out_dir: Path,
    make_asr: Callable[[], ASRBackend],
    read_wav: ReadWav = read_wav,
    root: Path = REPO_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[Path]:
    """Transcribe each version and write its transcript and metadata files.

    Every requested version is checked first: if any of them has no clips or a missing
    WAV, nothing is written and ``make_asr`` is never called. The model is built once
    and shared by all versions. Returns the written paths.
    """
    selected = {version: clips_for_version(manifest, version) for version in versions}
    missing = [path for clips in selected.values() for path in missing_wavs(clips, root)]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            f"{len(missing)} WAV file(s) listed in the manifest are missing:\n{listing}"
        )

    asr = make_asr()
    asr.load()
    written: list[Path] = []
    for version, clips in selected.items():
        seconds = 0.0

        def counting_read(path: Path) -> tuple[npt.NDArray[Any], int]:
            nonlocal seconds
            wave, sr = read_wav(path)
            seconds += wave.shape[0] / sr
            return wave, sr

        start = time.perf_counter()
        records = transcribe_clips(clips, counting_read, asr, root)
        wall = time.perf_counter() - start

        transcripts_path = out_dir / f"transcripts_{version}.json"
        meta_path = out_dir / f"transcripts_{version}.meta.json"
        write_transcripts(transcripts_path, records)
        write_json(
            meta_path, build_metadata(version, asr.metadata(), len(records), seconds, wall, now())
        )
        empty = sum(1 for r in records if not r["message"])
        print(
            f"{version}: {len(records)} clips, {seconds:.1f} s audio, {wall:.1f} s wall, "
            f"{empty} empty -> {transcripts_path}"
        )
        written += [transcripts_path, meta_path]
    return written


def make_whisper(
    model_id: str,
    device: str,
    batch_size: int,
    initial_prompt: str | None = None,
) -> ASRBackend:
    """Resolve the device, print it, and return an (unloaded) WhisperASR."""
    resolved = resolve_device(device)
    print(f"device: {resolved}")
    return WhisperASR(model_id, resolved, batch_size, initial_prompt=initial_prompt)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (design §4.5)."""
    parser = argparse.ArgumentParser(
        description="Transcribe the manifest's audio clips with Whisper."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--version", choices=[*VERSIONS, "all"], default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda:0", ...')
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional Whisper initial prompt (e.g. domain terms or place names).",
    )
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = resolve_default_model()
    return args


def main(
    argv: Sequence[str] | None = None,
    make_asr: Callable[..., ASRBackend] = make_whisper,
    read_wav: ReadWav = read_wav,
    root: Path = REPO_ROOT,
) -> int:
    """Command-line entry point. Returns the process exit code.

    ``make_asr``, ``read_wav`` and ``root`` (what manifest clip paths are relative to)
    are injectable so the whole flow can be tested with a fake recognizer.
    """
    args = parse_args(argv)
    if args.batch_size < 1:
        print(f"error: --batch-size must be >= 1, got {args.batch_size}", file=sys.stderr)
        return 2
    versions = list(VERSIONS) if args.version == "all" else [args.version]

    def _build_asr() -> ASRBackend:
        if args.initial_prompt is not None:
            return make_asr(args.model, args.device, args.batch_size, args.initial_prompt)
        return make_asr(args.model, args.device, args.batch_size)

    try:
        manifest = load_manifest(args.manifest)
        transcribe_versions(
            manifest,
            versions,
            args.out_dir,
            _build_asr,
            read_wav=read_wav,
            root=root,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
