"""LoRA fine-tuning for Whisper on Kenyan/African-accented English speech.

Streams samples from ``intronhealth/afrispeech-200`` (filtered to Kenyan accents by default,
or all accents with ``--all-accents``) and optionally mixes in a local audio manifest
(``audio_manifest.json`` format), trains a PEFT LoRA adapter on ``WhisperForConditionalGeneration``,
and merges + saves the resulting model + processor so ``src/asr/transcribe.py --model <out-dir>``
can load it directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import soundfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import make_audio  # noqa: E402
from src.training.whisper_base import (  # noqa: E402
    BASE_WHISPER_LARGE_V3_MODEL_ID,
    resolve_training_base_model,
)

DEFAULT_BASE_MODEL = BASE_WHISPER_LARGE_V3_MODEL_ID
DEFAULT_DATASET = "robello2/mini_afrispeech"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_OUT_DIR = REPO_ROOT / ".dev_models" / "whisper-african-lora"
DEFAULT_TRAIN_CACHE_DIR = REPO_ROOT / ".dev_models" / "_train_cache"
DEFAULT_HF_DATA_CACHE_DIR = REPO_ROOT / ".dev_models" / "_hf_data_cache"
ASR_SAMPLE_RATE = 16_000

# Accent substrings in AfriSpeech-200 that correspond to Kenyan speakers.
KENYAN_ACCENT_KEYWORDS: tuple[str, ...] = (
    "kenya",
    "kikuyu",
    "luo",
    "luhya",
    "kalenjin",
    "kamba",
    "kisii",
    "meru",
    "maasai",
    "swahili",
    "kiswahili",
    "turkana",
    "embu",
    "taita",
)


@dataclass(frozen=True)
class TrainingSample:
    """One 16 kHz mono float32 audio sample paired with its reference transcript."""

    audio: npt.NDArray[np.float32]
    text: str
    audio_id: str | None = None
    accent: str = ""


_DEDUP_PUNCT_RE = re.compile(r"[^\w\s]+")


def normalize_dedup_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for leakage deduplication."""
    return " ".join(_DEDUP_PUNCT_RE.sub(" ", text.lower()).split())


def deduplicate_samples(
    samples: Sequence[TrainingSample],
    *,
    exclude_texts: Iterable[str] = (),
    exclude_audio_ids: Iterable[str] = (),
) -> tuple[list[TrainingSample], dict[str, int]]:
    """Remove samples matching held-out exclusions or duplicate text/audio_id entries."""
    blocked_texts = {
        norm for norm in (normalize_dedup_text(t) for t in exclude_texts) if norm
    }
    blocked_ids = {str(aid).strip() for aid in exclude_audio_ids if str(aid).strip()}
    seen_texts: set[str] = set()
    seen_ids: set[str] = set()
    kept: list[TrainingSample] = []
    removed_excluded = 0
    removed_duplicate = 0

    for sample in samples:
        norm_text = normalize_dedup_text(sample.text)
        aid = sample.audio_id.strip() if sample.audio_id else None
        if norm_text in blocked_texts or (aid is not None and aid in blocked_ids):
            removed_excluded += 1
            continue
        if (norm_text and norm_text in seen_texts) or (aid is not None and aid in seen_ids):
            removed_duplicate += 1
            continue
        if norm_text:
            seen_texts.add(norm_text)
        if aid is not None:
            seen_ids.add(aid)
        kept.append(sample)

    stats = {
        "input": len(samples),
        "kept": len(kept),
        "removed_excluded": removed_excluded,
        "removed_duplicate": removed_duplicate,
    }
    return kept, stats


def is_kenyan_accent(accent: str | None) -> bool:
    """Return True if an AfriSpeech accent label matches a Kenyan accent."""
    if not accent:
        return False
    lowered = accent.lower()
    return any(keyword in lowered for keyword in KENYAN_ACCENT_KEYWORDS)


def resample_to_16k(wave: npt.NDArray[Any], sr: int) -> npt.NDArray[np.float32]:
    """Convert audio to 16 kHz float32 mono."""
    mono = np.asarray(wave, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1, dtype=np.float32)
    elif mono.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {mono.shape}")
    if sr != ASR_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=ASR_SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def normalize_all_caps_transcript(text: str) -> str:
    """Collapse excessive whitespace and lowercase ALL-CAPS clinical notes for Whisper BPE."""
    cleaned = " ".join(text.split())
    alpha = [ch for ch in cleaned if ch.isalpha()]
    if len(alpha) >= 8 and sum(ch.isupper() for ch in alpha) / len(alpha) > 0.6:
        return cleaned.lower()
    return cleaned


def iter_afrispeech_samples(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
    kenyan_only: bool = True,
    max_duration_s: float | None = None,
    clean_transcripts: bool = False,
) -> Iterator[TrainingSample]:
    """Yield normalized TrainingSample objects from an AfriSpeech-like row iterator."""
    import io

    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    count = 0
    for row in rows:
        if kenyan_only and not is_kenyan_accent(str(row.get("accent", ""))):
            continue
        raw_text = str(row.get("transcript") or row.get("transcription") or "")
        text = (
            normalize_all_caps_transcript(raw_text)
            if clean_transcripts
            else " ".join(raw_text.split())
        )
        audio_info = row.get("audio")
        if not text or not isinstance(audio_info, dict):
            continue
        if "array" in audio_info and "sampling_rate" in audio_info:
            array = np.asarray(audio_info["array"])
            sr = int(audio_info["sampling_rate"])
        elif audio_info.get("bytes"):
            raw, sr = soundfile.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
            array = np.asarray(raw)
        elif audio_info.get("path"):
            raw, sr = soundfile.read(audio_info["path"], dtype="float32")
            array = np.asarray(raw)
        else:
            continue
        wave = resample_to_16k(array, sr)
        if max_duration_s is not None and len(wave) > int(max_duration_s * ASR_SAMPLE_RATE):
            continue
        raw_id = row.get("audio_id") or row.get("id")
        audio_id = str(raw_id).strip() if raw_id else None
        accent = str(row.get("accent") or "").strip().lower()
        yield TrainingSample(audio=wave, text=text, audio_id=audio_id, accent=accent)
        count += 1
        if limit is not None and count >= limit:
            return


def load_manifest_samples(manifest_path: Path, root: Path = REPO_ROOT) -> list[TrainingSample]:
    """Load TrainingSample objects from a local audio_manifest.json file."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = data.get("clips")
    if not isinstance(clips, list):
        raise ValueError(f"{manifest_path} has no 'clips' list")
    samples: list[TrainingSample] = []
    for clip in clips:
        ref = str(clip.get("reference_text") or "").strip()
        rel_path = str(clip.get("path") or "")
        if not ref or not rel_path:
            continue
        wav_path = root / rel_path
        wave, sr = soundfile.read(wav_path, dtype="float32")
        samples.append(TrainingSample(audio=resample_to_16k(np.asarray(wave), int(sr)), text=ref))
    return samples


class WhisperDataCollator:
    """Pad Whisper input_features and label sequences into training batches."""

    def __init__(self, processor: Any, decoder_start_token_id: int) -> None:
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch: dict[str, Any] = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def augment_wave(
    wave: npt.NDArray[np.float32], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """Apply a random acoustic perturbation (pink noise or 8 kHz phone bandwidth) to a wave."""
    if wave.size < 4:
        return wave
    if float(rng.random()) < 0.5:
        sig_rms = float(np.sqrt(np.mean(wave**2)))
        if sig_rms <= 0.0:
            return wave
        noise_seed = int(rng.integers(0, 2**31 - 1))
        noise = make_audio.pink_noise(len(wave), noise_seed)
        snr_db = float(rng.uniform(12.0, 24.0))
        scale = sig_rms / (10.0 ** (snr_db / 20.0))
        mixed = wave + noise * np.float32(scale)
        return np.clip(mixed, -0.99, 0.99).astype(np.float32)
    down = librosa.resample(wave, orig_sr=ASR_SAMPLE_RATE, target_sr=make_audio.SR_PHONE)
    up = librosa.resample(down, orig_sr=make_audio.SR_PHONE, target_sr=ASR_SAMPLE_RATE)
    if len(up) < len(wave):
        up = np.pad(up, (0, len(wave) - len(up)))
    elif len(up) > len(wave):
        up = up[: len(wave)]
    clipped: npt.NDArray[np.float32] = np.ascontiguousarray(
        np.clip(up, -0.99, 0.99), dtype=np.float32
    )
    return clipped


def _maybe_augment_wave(
    wave: npt.NDArray[np.float32],
    idx: int,
    noise_aug_prob: float,
    aug_seed: int,
    call_counts: dict[int, int],
) -> npt.NDArray[np.float32]:
    call_num = call_counts.get(idx, 0)
    call_counts[idx] = call_num + 1
    if noise_aug_prob > 0.0:
        rng = np.random.default_rng((aug_seed, idx, call_num))
        if float(rng.random()) < noise_aug_prob:
            return augment_wave(wave, rng)
    return wave


class LazyWhisperDataset:
    """On-the-fly Whisper feature dataset storing compact int16 PCM waveforms in RAM."""

    def __init__(
        self,
        samples: Sequence[TrainingSample],
        processor: Any,
        *,
        noise_aug_prob: float = 0.0,
        aug_seed: int = 1234,
    ) -> None:
        if not 0.0 <= noise_aug_prob <= 1.0:
            raise ValueError(f"noise_aug_prob must be in [0, 1], got {noise_aug_prob}")
        self._pcm16 = [
            (np.clip(s.audio, -1.0, 1.0) * 32767.0).astype(np.int16) for s in samples
        ]
        self._texts = [s.text for s in samples]
        self.processor = processor
        self._noise_aug_prob = float(noise_aug_prob)
        self._aug_seed = int(aug_seed)
        self._call_counts: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wave = self._pcm16[idx].astype(np.float32) / 32767.0
        wave = _maybe_augment_wave(
            wave, idx, self._noise_aug_prob, self._aug_seed, self._call_counts
        )
        sample = TrainingSample(audio=wave, text=self._texts[idx])
        return prepare_feature_dict(sample, self.processor)


def build_pcm16_cache(
    cache_dir: Path,
    sample_iterables: Sequence[Iterable[TrainingSample]],
    *,
    exclude_texts: Iterable[str] = (),
    exclude_audio_ids: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Stream samples to a disk-backed int16 PCM binary + JSONL index with <1 GB RAM."""
    blocked_texts = {
        norm for norm in (normalize_dedup_text(t) for t in exclude_texts) if norm
    }
    blocked_ids = {str(aid).strip() for aid in exclude_audio_ids if str(aid).strip()}
    seen_texts: set[str] = set()
    seen_ids: set[str] = set()

    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_path = cache_dir / "audio_pcm16.bin"
    idx_path = cache_dir / "index.jsonl"
    tmp_bin = cache_dir / "audio_pcm16.bin.tmp"
    tmp_idx = cache_dir / "index.jsonl.tmp"

    entries: list[dict[str, Any]] = []
    raw_count = 0
    removed_excluded = 0
    removed_duplicate = 0
    offset = 0

    try:
        with (
            tmp_bin.open("wb") as bin_fh,
            tmp_idx.open("w", encoding="utf-8", newline="\n") as idx_fh,
        ):
            for iterable in sample_iterables:
                for sample in iterable:
                    raw_count += 1
                    norm_text = normalize_dedup_text(sample.text)
                    aid = sample.audio_id.strip() if sample.audio_id else None
                    if norm_text in blocked_texts or (aid is not None and aid in blocked_ids):
                        removed_excluded += 1
                        continue
                    if (norm_text and norm_text in seen_texts) or (
                        aid is not None and aid in seen_ids
                    ):
                        removed_duplicate += 1
                        continue
                    if norm_text:
                        seen_texts.add(norm_text)
                    if aid is not None:
                        seen_ids.add(aid)

                    pcm16 = (np.clip(sample.audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                    n_samples = int(pcm16.size)
                    if n_samples == 0:
                        continue
                    bin_fh.write(pcm16.tobytes())
                    entry: dict[str, Any] = {
                        "offset": offset,
                        "n_samples": n_samples,
                        "text": sample.text,
                        "accent": sample.accent,
                        "audio_id": aid,
                    }
                    idx_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    entries.append(entry)
                    offset += n_samples
        if not entries:
            raise ValueError("no valid samples kept for PCM16 cache")
        tmp_bin.replace(bin_path)
        tmp_idx.replace(idx_path)
    except Exception:
        tmp_bin.unlink(missing_ok=True)
        tmp_idx.unlink(missing_ok=True)
        raise

    stats = {
        "raw": raw_count,
        "kept": len(entries),
        "removed_excluded": removed_excluded,
        "removed_duplicate": removed_duplicate,
    }
    return entries, stats


def load_pcm16_cache(cache_dir: Path) -> list[dict[str, Any]]:
    """Load index metadata for an existing disk-backed PCM16 cache directory."""
    idx_path = cache_dir / "index.jsonl"
    bin_path = cache_dir / "audio_pcm16.bin"
    if not idx_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"PCM16 cache missing in {cache_dir}")
    entries: list[dict[str, Any]] = []
    for raw_line in idx_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"empty PCM16 cache index in {idx_path}")
    return entries


class MemmapWhisperDataset:
    """Disk-backed memmap Whisper feature dataset keeping process RAM < 1 GB."""

    def __init__(
        self,
        cache_dir: Path,
        processor: Any,
        *,
        entries: Sequence[dict[str, Any]] | None = None,
        indices: Sequence[int] | None = None,
        noise_aug_prob: float = 0.0,
        aug_seed: int = 1234,
    ) -> None:
        if not 0.0 <= noise_aug_prob <= 1.0:
            raise ValueError(f"noise_aug_prob must be in [0, 1], got {noise_aug_prob}")
        self._bin_path = cache_dir / "audio_pcm16.bin"
        if not self._bin_path.is_file():
            raise FileNotFoundError(f"missing {self._bin_path}")
        self._entries = list(entries) if entries is not None else load_pcm16_cache(cache_dir)
        self._indices = (
            list(indices) if indices is not None else list(range(len(self._entries)))
        )
        if not self._indices:
            raise ValueError("MemmapWhisperDataset indices must not be empty")
        self.processor = processor
        self._noise_aug_prob = float(noise_aug_prob)
        self._aug_seed = int(aug_seed)
        self._call_counts: dict[int, int] = {}
        self._mmap: Any = None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._mmap is None:
            self._mmap = np.memmap(self._bin_path, dtype=np.int16, mode="r")
        entry = self._entries[self._indices[idx]]
        offset = int(entry["offset"])
        n_samples = int(entry["n_samples"])
        pcm = np.array(self._mmap[offset : offset + n_samples], copy=True)
        wave = pcm.astype(np.float32) / 32767.0
        wave = _maybe_augment_wave(
            wave, idx, self._noise_aug_prob, self._aug_seed, self._call_counts
        )
        sample = TrainingSample(audio=wave, text=str(entry["text"]))
        return prepare_feature_dict(sample, self.processor)


def prepare_feature_dict(sample: TrainingSample, processor: Any) -> dict[str, Any]:
    """Convert one TrainingSample into Whisper input_features and label token IDs."""
    input_features = processor.feature_extractor(
        sample.audio, sampling_rate=ASR_SAMPLE_RATE
    ).input_features[0]
    labels = processor.tokenizer(sample.text).input_ids
    return {"input_features": input_features, "labels": labels}


def _stream_parquet_shards(
    dataset_name: str,
    shards: Sequence[str],
    *,
    seed: int | None = None,
    columns: Sequence[str] | None = None,
) -> Iterable[dict[str, Any]]:  # pragma: no cover - network
    import importlib

    from huggingface_hub import hf_hub_download

    pa: Any = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")

    shard_list = list(shards)
    rng = np.random.default_rng(seed) if seed is not None else None
    if rng is not None:
        order = rng.permutation(len(shard_list)).tolist()
        shard_list = [shard_list[i] for i in order]

    DEFAULT_HF_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for shard in shard_list:
        local_path = hf_hub_download(
            dataset_name,
            repo_type="dataset",
            filename=shard,
            cache_dir=str(DEFAULT_HF_DATA_CACHE_DIR),
        )
        table = pq.read_table(local_path, columns=list(columns) if columns is not None else None)
        rows = table.to_pylist()
        del table
        pa.default_memory_pool().release_unused()
        if rng is not None:
            row_order = rng.permutation(len(rows)).tolist()
            rows = [rows[i] for i in row_order]
        yield from rows


def stream_hf_parquet_files(
    dataset_name: str,
    filenames: Sequence[str],
    *,
    seed: int | None = None,
    columns: Sequence[str] | None = None,
) -> Iterable[dict[str, Any]]:  # pragma: no cover - network
    """Stream rows from explicit Parquet shard filenames in a Hugging Face dataset."""
    yield from _stream_parquet_shards(dataset_name, filenames, seed=seed, columns=columns)


def stream_hf_afrispeech(
    dataset_name: str, config: str, split: str, *, seed: int | None = None
) -> Iterable[dict[str, Any]]:  # pragma: no cover - network
    """Stream AfriSpeech rows from Parquet shards on Hugging Face without torchcodec.

    When ``seed`` is provided, shard order is deterministically shuffled so limited runs
    mix diverse accents across the dataset.
    """
    from huggingface_hub import HfApi

    all_files = HfApi().list_repo_files(dataset_name, repo_type="dataset")
    split_prefix = f"data/{split}-"
    shards = sorted(f for f in all_files if f.startswith(split_prefix) and f.endswith(".parquet"))
    if not shards:
        shards = sorted(f for f in all_files if split in f and f.endswith(".parquet"))
    if not shards:
        raise ValueError(f"no parquet shards found for split {split!r} in {dataset_name}")
    yield from _stream_parquet_shards(dataset_name, shards, seed=seed)


DEFAULT_LORA_TARGETS: tuple[str, ...] = ("q_proj", "v_proj")


def train_lora(
    train_samples: Sequence[TrainingSample],
    eval_samples: Sequence[TrainingSample],
    *,
    base_model: str,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 2,
    eval_batch_size: int | None = None,
    grad_accum: int = 4,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    save_best: bool = False,
    eval_steps: int | None = None,
    spec_augment: bool = False,
    noise_aug_prob: float = 0.0,
    aug_seed: int = 1234,
    cache_dir: Path | None = None,
    train_indices: Sequence[int] | None = None,
    load_in_8bit: bool = False,
    gradient_checkpointing: bool = True,
) -> Path:  # pragma: no cover - heavy GPU training
    """Fine-tune Whisper with PEFT LoRA (optionally 8-bit QLoRA), merge, and save."""
    import gc

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        BitsAndBytesConfig,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    if not train_samples and cache_dir is None:
        raise ValueError("train_samples must not be empty")
    if eval_steps is not None and eval_steps < 1:
        raise ValueError(f"eval_steps must be >= 1, got {eval_steps}")
    if eval_batch_size is not None and eval_batch_size < 1:
        raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")

    base_model = resolve_training_base_model(base_model)
    processor = WhisperProcessor.from_pretrained(base_model, language="English", task="transcribe")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    load_kwargs: dict[str, Any] = {}
    if load_in_8bit:
        bnb_cls: Any = BitsAndBytesConfig
        load_kwargs["quantization_config"] = bnb_cls(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        load_kwargs["dtype"] = torch.bfloat16 if use_bf16 else torch.float16

    model = WhisperForConditionalGeneration.from_pretrained(base_model, **load_kwargs)
    model_any: Any = model
    if load_in_8bit:
        prep_kbit: Any = prepare_model_for_kbit_training
        model = prep_kbit(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model_any.enable_input_require_grads()

    gen_cfg: Any = model.generation_config
    if gen_cfg is not None:
        gen_cfg.language = "english"
        gen_cfg.task = "transcribe"
        gen_cfg.forced_decoder_ids = None
    model.config.use_cache = False
    if spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.04
        model.config.mask_feature_length = 10

    max_target_pos = int(getattr(model.config, "max_target_positions", 448) or 448)
    train_ds: Any
    if cache_dir is not None:
        entries = load_pcm16_cache(cache_dir)
        base_idx = list(range(len(entries))) if train_indices is None else list(train_indices)
        filtered_idx = [
            i
            for i in base_idx
            if len(processor.tokenizer(str(entries[i]["text"])).input_ids) <= max_target_pos
        ]
        dropped_train = len(base_idx) - len(filtered_idx)
        if dropped_train:
            print(
                f"Filtered {dropped_train} train sample(s) exceeding {max_target_pos} label tokens"
            )
        if not filtered_idx:
            raise ValueError("all cached train samples exceeded max_target_positions")
        train_ds = MemmapWhisperDataset(
            cache_dir,
            processor,
            entries=entries,
            indices=filtered_idx,
            noise_aug_prob=noise_aug_prob,
            aug_seed=aug_seed,
        )
        n_train_items = len(filtered_idx)
    else:
        filtered_train = [
            s
            for s in train_samples
            if len(processor.tokenizer(s.text).input_ids) <= max_target_pos
        ]
        dropped_train = len(train_samples) - len(filtered_train)
        if dropped_train:
            print(
                f"Filtered {dropped_train} train sample(s) exceeding {max_target_pos} label tokens"
            )
        if not filtered_train:
            raise ValueError("all train_samples exceeded max_target_positions")
        train_ds = LazyWhisperDataset(
            filtered_train, processor, noise_aug_prob=noise_aug_prob, aug_seed=aug_seed
        )
        n_train_items = len(filtered_train)

    filtered_eval = [
        s for s in eval_samples if len(processor.tokenizer(s.text).input_ids) <= max_target_pos
    ]
    dropped_eval = len(eval_samples) - len(filtered_eval)
    if dropped_eval:
        print(f"Filtered {dropped_eval} eval sample(s) exceeding {max_target_pos} label tokens")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.05,
        bias="none",
    )
    peft_model: Any = get_peft_model(model, lora_config)

    eval_ds: Any = LazyWhisperDataset(filtered_eval, processor) if filtered_eval else None

    use_save_best = save_best and eval_ds is not None
    use_step_eval = eval_ds is not None and eval_steps is not None
    eval_strategy = "steps" if use_step_eval else ("epoch" if eval_ds is not None else "no")
    save_strategy = ("steps" if use_step_eval else "epoch") if use_save_best else "no"
    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, (n_train_items + eff_batch - 1) // eff_batch)
    warmup_steps = max(1, int(steps_per_epoch * epochs * warmup_ratio)) if warmup_ratio > 0 else 0
    step_kwargs: dict[str, Any] = {}
    if use_step_eval and eval_steps is not None:
        step_kwargs["eval_steps"] = eval_steps
        if use_save_best:
            step_kwargs["save_steps"] = eval_steps
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "_checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size if eval_batch_size is not None else batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=2 if use_save_best else None,
        load_best_model_at_end=use_save_best,
        metric_for_best_model="eval_loss" if use_save_best else None,
        greater_is_better=False if use_save_best else None,
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        **step_kwargs,
    )

    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=peft_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    if load_in_8bit:
        adapter_dir = out_dir / "_adapter"
        peft_model.save_pretrained(adapter_dir)
        del trainer, peft_model, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_base = WhisperForConditionalGeneration.from_pretrained(
            base_model, dtype=torch.float16, low_cpu_mem_usage=True
        )
        merged_model: Any = PeftModel.from_pretrained(cpu_base, adapter_dir).merge_and_unload()
        merged_model.config.apply_spec_augment = False
        merged_model.config.use_cache = True
        merged_model.save_pretrained(out_dir)
    else:
        merged: Any = peft_model.merge_and_unload()
        merged.config.apply_spec_augment = False
        merged.config.use_cache = True
        merged.save_pretrained(out_dir)
        del trainer, peft_model, merged, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    processor.save_pretrained(out_dir)
    return out_dir


def make_k_folds(
    samples: Sequence[TrainingSample], k: int, seed: int = 1234
) -> list[tuple[list[TrainingSample], list[TrainingSample]]]:
    """Split samples into k deterministic (train, val) folds."""
    if k < 2:
        raise ValueError(f"k must be >= 2 for k-fold CV, got {k}")
    if len(samples) < k:
        raise ValueError(f"need at least {k} samples for {k}-fold CV, got {len(samples)}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples)).tolist()
    folds: list[list[int]] = [[] for _ in range(k)]
    for pos, idx in enumerate(indices):
        folds[pos % k].append(idx)
    splits: list[tuple[list[TrainingSample], list[TrainingSample]]] = []
    for fold_idx in range(k):
        val_set = set(folds[fold_idx])
        val = [samples[i] for i in folds[fold_idx]]
        train = [samples[i] for i in range(len(samples)) if i not in val_set]
        splits.append((train, val))
    return splits


def evaluate_model_wer(
    model_dir_or_id: str | Path,
    samples: Sequence[TrainingSample],
    *,
    batch_size: int = 8,
) -> float:  # pragma: no cover - heavy GPU inference
    """Compute corpus WER (0-100%) of a Whisper model on TrainingSample clips."""
    import gc
    import re

    import jiwer
    import torch
    from transformers import pipeline

    if not samples:
        raise ValueError("samples must not be empty")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe_kwargs: dict[str, Any] = {
        "model": str(model_dir_or_id),
        "dtype": dtype,
        "device": device,
    }
    if any(len(s.audio) > 30 * ASR_SAMPLE_RATE for s in samples):
        pipe_kwargs["chunk_length_s"] = 30
    pipe: Any = pipeline("automatic-speech-recognition", **pipe_kwargs)
    gen_kwargs = {"language": "en", "task": "transcribe", "num_beams": 1, "do_sample": False}
    hyps = [str(pipe(s.audio, generate_kwargs=gen_kwargs)["text"]).strip() for s in samples]
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _norm(t: str) -> str:
        t = t.lower().replace("'", "")
        return re.sub(r"[^a-z0-9]+", " ", t).strip()

    refs_norm = [_norm(s.text) or "<empty>" for s in samples]
    hyps_norm = [_norm(h) for h in hyps]
    return float(100.0 * jiwer.wer(refs_norm, hyps_norm))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Whisper on Kenyan/African-accented English."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--train-limit", type=int, default=500, help="max train clips (0 = all valid clips)"
    )
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument(
        "--kenyan-only", action="store_true", help="filter rows to Kenyan accents only"
    )
    parser.add_argument(
        "--extra-dataset",
        default=None,
        help="Optional extra HF Parquet dataset (e.g. Professor/afrispeech-200-16khz).",
    )
    parser.add_argument(
        "--extra-accents",
        nargs="+",
        default=None,
        help="Accent names matching data/train-<accent>.parquet in --extra-dataset.",
    )
    parser.add_argument(
        "--exclude-test-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude samples matching held-out test/validation splits from training.",
    )
    parser.add_argument("--local-manifest", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Disk-backed int16 PCM cache directory (builds if missing, reuses if present).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-targets", nargs="+", default=list(DEFAULT_LORA_TARGETS),
        help="module names to wrap with LoRA",
    )
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="drop clips longer than this many seconds (<=0 keeps all)",
    )
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument(
        "--eval-steps", type=int, default=None, help="evaluate/save every N steps instead of epoch"
    )
    parser.add_argument(
        "--spec-augment", action="store_true", help="enable Whisper SpecAugment during training"
    )
    parser.add_argument(
        "--noise-aug-prob",
        type=float,
        default=0.0,
        help="probability of on-the-fly pink-noise or 8kHz telephony training augmentation",
    )
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument(
        "--clean-transcripts", action="store_true",
        help="lowercase ALL-CAPS clinical notes in training transcripts",
    )
    parser.add_argument("--load-in-8bit", action="store_true", help="use 8-bit QLoRA for 8GB GPUs")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--k-folds", type=int, default=1, help="set >=2 for k-fold cross-validation"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument(
        "--cv-report", type=Path, default=None, help="optional JSON path for eval/CV metrics"
    )
    args = parser.parse_args(argv)
    if args.train_limit < 0:
        parser.error("--train-limit must be >= 0")
    if args.eval_steps is not None and args.eval_steps < 1:
        parser.error("--eval-steps must be >= 1")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be >= 1")
    if not 0.0 <= args.noise_aug_prob <= 1.0:
        parser.error("--noise-aug-prob must be in [0.0, 1.0]")
    return args


def _collect_holdout_exclusions(
    dataset: str,
    extra_dataset: str | None,
    extra_accents: Sequence[str] | None,
) -> tuple[list[str], list[str]]:  # pragma: no cover - network/cache
    """Collect held-out transcripts and audio_ids to exclude from training."""
    from huggingface_hub import HfApi

    api = HfApi()
    exclude_texts: list[str] = []
    exclude_ids: list[str] = []

    base_files = api.list_repo_files(dataset, repo_type="dataset")
    base_test_shards = sorted(
        f for f in base_files if f.startswith("data/test-") and f.endswith(".parquet")
    )
    for row in stream_hf_parquet_files(
        dataset, base_test_shards, columns=["transcript", "audio_id"]
    ):
        if row.get("transcript"):
            exclude_texts.append(str(row["transcript"]))
        if row.get("audio_id"):
            exclude_ids.append(str(row["audio_id"]))

    if extra_dataset:
        extra_files = set(api.list_repo_files(extra_dataset, repo_type="dataset"))
        holdout_accents = (
            *(extra_accents or ()),
            "swahili",
            "yoruba",
            "igbo",
            "hausa",
            "afrikaans",
        )
        extra_holdout = sorted(
            f
            for f in extra_files
            if (f.startswith("data/test-") or f.startswith("data/validation-"))
            and f.endswith(".parquet")
            and (
                not extra_accents
                or any(
                    f"data/test-{a}.parquet" == f or f"data/validation-{a}.parquet" == f
                    for a in holdout_accents
                )
            )
        )
        for row in stream_hf_parquet_files(
            extra_dataset, extra_holdout, columns=["transcript", "audio_id"]
        ):
            if row.get("transcript"):
                exclude_texts.append(str(row["transcript"]))
            if row.get("audio_id"):
                exclude_ids.append(str(row["audio_id"]))
    return exclude_texts, exclude_ids


def _resolve_extra_train_shards(
    extra_dataset: str, extra_accents: Sequence[str] | None
) -> list[str]:  # pragma: no cover - network/cache
    from huggingface_hub import HfApi

    extra_repo_files = set(HfApi().list_repo_files(extra_dataset, repo_type="dataset"))
    if extra_accents:
        shards = [
            f"data/train-{accent}.parquet"
            for accent in extra_accents
            if f"data/train-{accent}.parquet" in extra_repo_files
        ]
    else:
        shards = sorted(
            f for f in extra_repo_files if f.startswith("data/train-") and f.endswith(".parquet")
        )
    if not shards:
        raise ValueError(f"no matching train shards found in {extra_dataset}")
    return shards


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = parse_args(argv)
    kenyan_only = bool(args.kenyan_only)
    grad_ckpt = not args.no_gradient_checkpointing
    max_dur = float(args.max_duration) if args.max_duration > 0 else None
    train_limit = None if args.train_limit == 0 else int(args.train_limit)

    if args.cache_dir is not None and args.k_folds >= 2:
        print("error: --cache-dir cannot be combined with --k-folds >= 2", file=sys.stderr)
        return 2

    train_kw: dict[str, Any] = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.lora_targets,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "save_best": bool(args.save_best),
        "eval_steps": args.eval_steps,
        "spec_augment": bool(args.spec_augment),
        "noise_aug_prob": float(args.noise_aug_prob),
        "aug_seed": int(args.seed),
        "load_in_8bit": args.load_in_8bit,
        "gradient_checkpointing": grad_ckpt,
    }

    cache_dir: Path | None = (
        args.cache_dir
        if args.cache_dir is not None
        else (DEFAULT_TRAIN_CACHE_DIR if args.k_folds < 2 else None)
    )
    train_samples: list[TrainingSample] = []
    train_indices: list[int] | None = None
    n_train_count = 0

    if cache_dir is not None:
        idx_file = cache_dir / "index.jsonl"
        bin_file = cache_dir / "audio_pcm16.bin"
        if idx_file.is_file() and bin_file.is_file():
            entries = load_pcm16_cache(cache_dir)
            print(f"Reusing PCM16 cache at {cache_dir}: {len(entries)} clips", flush=True)
        else:
            exclude_texts, exclude_ids = (
                _collect_holdout_exclusions(args.dataset, args.extra_dataset, args.extra_accents)
                if args.exclude_test_datasets
                else ([], [])
            )
            iterables: list[Iterable[TrainingSample]] = [
                iter_afrispeech_samples(
                    stream_hf_afrispeech(
                        args.dataset, args.dataset_config, "train", seed=args.seed
                    ),
                    limit=train_limit,
                    kenyan_only=kenyan_only,
                    max_duration_s=max_dur,
                    clean_transcripts=bool(args.clean_transcripts),
                )
            ]
            if args.local_manifest is not None:
                iterables.append(load_manifest_samples(args.local_manifest))
            if args.extra_dataset:
                try:
                    extra_shards = _resolve_extra_train_shards(
                        args.extra_dataset, args.extra_accents
                    )
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
                iterables.append(
                    iter_afrispeech_samples(
                        stream_hf_parquet_files(args.extra_dataset, extra_shards, seed=args.seed),
                        kenyan_only=kenyan_only,
                        max_duration_s=max_dur,
                        clean_transcripts=bool(args.clean_transcripts),
                    )
                )
            entries, dedup_stats = build_pcm16_cache(
                cache_dir,
                iterables,
                exclude_texts=exclude_texts,
                exclude_audio_ids=exclude_ids,
            )
            print(
                f"Built PCM16 cache at {cache_dir}: raw={dedup_stats['raw']} "
                f"kept={dedup_stats['kept']} excluded={dedup_stats['removed_excluded']} "
                f"duplicates={dedup_stats['removed_duplicate']}",
                flush=True,
            )
        n_train_count = len(entries)
        rng = np.random.default_rng(args.seed)
        train_indices = rng.permutation(n_train_count).tolist()
    else:
        train_samples = list(
            iter_afrispeech_samples(
                stream_hf_afrispeech(args.dataset, args.dataset_config, "train", seed=args.seed),
                limit=train_limit,
                kenyan_only=kenyan_only,
                max_duration_s=max_dur,
                clean_transcripts=bool(args.clean_transcripts),
            )
        )
        if args.local_manifest is not None:
            train_samples.extend(load_manifest_samples(args.local_manifest))
        if args.extra_dataset:
            try:
                extra_shards = _resolve_extra_train_shards(
                    args.extra_dataset, args.extra_accents
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            extra_samples = list(
                iter_afrispeech_samples(
                    stream_hf_parquet_files(args.extra_dataset, extra_shards, seed=args.seed),
                    kenyan_only=kenyan_only,
                    max_duration_s=max_dur,
                    clean_transcripts=bool(args.clean_transcripts),
                )
            )
            print(
                f"Collected {len(train_samples)} base + {len(extra_samples)} extra samples "
                f"from {len(extra_shards)} shard(s) in {args.extra_dataset}"
            )
            train_samples.extend(extra_samples)

        if args.extra_dataset or args.exclude_test_datasets:
            exclude_texts, exclude_ids = (
                _collect_holdout_exclusions(args.dataset, args.extra_dataset, args.extra_accents)
                if args.exclude_test_datasets
                else ([], [])
            )
            train_samples, dedup_stats = deduplicate_samples(
                train_samples,
                exclude_texts=exclude_texts,
                exclude_audio_ids=exclude_ids,
            )
            print(
                f"Deduplication: input={dedup_stats['input']} kept={dedup_stats['kept']} "
                f"excluded={dedup_stats['removed_excluded']} "
                f"duplicates={dedup_stats['removed_duplicate']}"
            )

        if len(train_samples) > 1:
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(len(train_samples)).tolist()
            train_samples = [train_samples[i] for i in perm]

        if not train_samples:
            print("error: no training samples collected", file=sys.stderr)
            return 1
        n_train_count = len(train_samples)

    if args.k_folds >= 2:
        folds = make_k_folds(train_samples, args.k_folds, seed=args.seed)
        fold_rows: list[dict[str, float | int]] = []
        for idx, (fold_train, fold_val) in enumerate(folds):
            fold_dir = args.out_dir / f"fold_{idx + 1}"
            base_wer = evaluate_model_wer(args.base_model, fold_val)
            saved_dir = train_lora(fold_train, fold_val, out_dir=fold_dir, **train_kw)
            tuned_wer = evaluate_model_wer(saved_dir, fold_val)
            delta = round(tuned_wer - base_wer, 2)
            row = {
                "fold": idx + 1,
                "n_train": len(fold_train),
                "n_val": len(fold_val),
                "baseline_wer_pct": round(base_wer, 2),
                "tuned_wer_pct": round(tuned_wer, 2),
                "delta_pts": delta,
            }
            fold_rows.append(row)
            print(
                f"Fold {idx + 1}/{args.k_folds}: baseline={row['baseline_wer_pct']:.2f}% "
                f"tuned={row['tuned_wer_pct']:.2f}% (delta={delta:+.2f} pts)"
            )
        base_vals = [float(r["baseline_wer_pct"]) for r in fold_rows]
        tuned_vals = [float(r["tuned_wer_pct"]) for r in fold_rows]
        report = {
            "base_model": args.base_model,
            "dataset": args.dataset,
            "k_folds": args.k_folds,
            "mean_baseline_wer_pct": round(float(np.mean(base_vals)), 2),
            "std_baseline_wer_pct": round(float(np.std(base_vals)), 2),
            "mean_tuned_wer_pct": round(float(np.mean(tuned_vals)), 2),
            "std_tuned_wer_pct": round(float(np.std(tuned_vals)), 2),
            "folds": fold_rows,
        }
        report_path = args.cv_report or (args.out_dir / "cv_results.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"CV summary ({args.k_folds} folds): "
            f"baseline={report['mean_baseline_wer_pct']:.2f}%±{report['std_baseline_wer_pct']:.2f} "
            f"-> tuned={report['mean_tuned_wer_pct']:.2f}%±{report['std_tuned_wer_pct']:.2f} "
            f"(saved to {report_path})"
        )
        return 0

    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    eval_samples = list(
        iter_afrispeech_samples(
            stream_hf_afrispeech(args.dataset, args.dataset_config, "test", seed=eval_seed),
            limit=args.eval_limit,
            kenyan_only=kenyan_only,
            max_duration_s=max_dur,
        )
    )
    eval_base_wer = (
        evaluate_model_wer(args.base_model, eval_samples)
        if eval_samples and not args.skip_baseline_eval
        else None
    )
    saved_dir = train_lora(
        train_samples,
        eval_samples,
        out_dir=args.out_dir,
        cache_dir=cache_dir,
        train_indices=train_indices,
        **train_kw,
    )
    eval_tuned_wer = evaluate_model_wer(saved_dir, eval_samples) if eval_samples else None
    if eval_tuned_wer is not None:
        base_str = f"baseline={eval_base_wer:.2f}% -> " if eval_base_wer is not None else ""
        delta_str = (
            f" (delta={eval_tuned_wer - eval_base_wer:+.2f} pts)"
            if eval_base_wer is not None
            else ""
        )
        print(
            f"Eval WER ({len(eval_samples)} clips): "
            f"{base_str}tuned={eval_tuned_wer:.2f}%{delta_str}"
        )
        if args.cv_report is not None:
            args.cv_report.parent.mkdir(parents=True, exist_ok=True)
            args.cv_report.write_text(
                json.dumps(
                    {
                        "base_model": args.base_model,
                        "dataset": args.dataset,
                        "n_train": n_train_count,
                        "n_eval": len(eval_samples),
                        "baseline_wer_pct": round(eval_base_wer, 2)
                        if eval_base_wer is not None
                        else None,
                        "tuned_wer_pct": round(eval_tuned_wer, 2),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    print(f"Saved merged LoRA Whisper model to {saved_dir} ({n_train_count} train samples)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
