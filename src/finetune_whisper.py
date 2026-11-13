"""LoRA fine-tuning for Whisper on Kenyan/African-accented English speech.

Streams samples from ``intronhealth/afrispeech-200`` (filtered to Kenyan accents by default,
or all accents with ``--all-accents``) and optionally mixes in a local audio manifest
(``audio_manifest.json`` format), trains a PEFT LoRA adapter on ``WhisperForConditionalGeneration``,
and merges + saves the resulting model + processor so ``src/transcribe.py --model <out-dir>``
can load it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import soundfile

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "openai/whisper-small"
DEFAULT_DATASET = "robello2/mini_afrispeech"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "whisper-african-lora"
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
        yield TrainingSample(audio=wave, text=text)
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


class LazyWhisperDataset:
    """On-the-fly Whisper feature dataset storing compact int16 PCM waveforms in RAM."""

    def __init__(self, samples: Sequence[TrainingSample], processor: Any) -> None:
        self._pcm16 = [
            (np.clip(s.audio, -1.0, 1.0) * 32767.0).astype(np.int16) for s in samples
        ]
        self._texts = [s.text for s in samples]
        self.processor = processor

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wave = self._pcm16[idx].astype(np.float32) / 32767.0
        sample = TrainingSample(audio=wave, text=self._texts[idx])
        return prepare_feature_dict(sample, self.processor)


def prepare_feature_dict(sample: TrainingSample, processor: Any) -> dict[str, Any]:
    """Convert one TrainingSample into Whisper input_features and label token IDs."""
    input_features = processor.feature_extractor(
        sample.audio, sampling_rate=ASR_SAMPLE_RATE
    ).input_features[0]
    labels = processor.tokenizer(sample.text).input_ids
    return {"input_features": input_features, "labels": labels}


def stream_hf_afrispeech(
    dataset_name: str, config: str, split: str, *, seed: int | None = None
) -> Iterable[dict[str, Any]]:  # pragma: no cover - network
    """Stream AfriSpeech rows from Parquet shards on Hugging Face without torchcodec.

    When ``seed`` is provided, shard order is deterministically shuffled so limited runs
    mix diverse accents across the dataset.
    """
    import importlib

    from huggingface_hub import HfApi, hf_hub_download

    pa: Any = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")
    all_files = HfApi().list_repo_files(dataset_name, repo_type="dataset")
    split_prefix = f"data/{split}-"
    shards = sorted(f for f in all_files if f.startswith(split_prefix) and f.endswith(".parquet"))
    if not shards:
        shards = sorted(f for f in all_files if split in f and f.endswith(".parquet"))
    if not shards:
        raise ValueError(f"no parquet shards found for split {split!r} in {dataset_name}")

    if seed is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(shards)).tolist()
        shards = [shards[i] for i in order]

    for shard in shards:
        local_path = hf_hub_download(dataset_name, repo_type="dataset", filename=shard)
        table = pq.read_table(local_path)
        rows = table.to_pylist()
        del table
        pa.default_memory_pool().release_unused()
        if seed is not None:
            row_order = rng.permutation(len(rows)).tolist()
            rows = [rows[i] for i in row_order]
        yield from rows


DEFAULT_LORA_TARGETS: tuple[str, ...] = ("q_proj", "v_proj")


def train_lora(
    train_samples: Sequence[TrainingSample],
    eval_samples: Sequence[TrainingSample],
    *,
    base_model: str,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    save_best: bool = False,
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

    if not train_samples:
        raise ValueError("train_samples must not be empty")

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

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.05,
        bias="none",
    )
    peft_model: Any = get_peft_model(model, lora_config)

    train_ds: Any = LazyWhisperDataset(train_samples, processor)
    eval_ds: Any = LazyWhisperDataset(eval_samples, processor) if eval_samples else None

    use_save_best = save_best and eval_ds is not None
    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, (len(train_samples) + eff_batch - 1) // eff_batch)
    warmup_steps = max(1, int(steps_per_epoch * epochs * warmup_ratio)) if warmup_ratio > 0 else 0
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "_checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
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
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if use_save_best else "no",
        save_total_limit=2 if use_save_best else None,
        load_best_model_at_end=use_save_best,
        metric_for_best_model="eval_loss" if use_save_best else None,
        greater_is_better=False if use_save_best else None,
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
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
        merged_model.save_pretrained(out_dir)
    else:
        merged: Any = peft_model.merge_and_unload()
        merged.save_pretrained(out_dir)
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
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument(
        "--kenyan-only", action="store_true", help="filter rows to Kenyan accents only"
    )
    parser.add_argument("--local-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = parse_args(argv)
    kenyan_only = bool(args.kenyan_only)
    grad_ckpt = not args.no_gradient_checkpointing
    max_dur = float(args.max_duration) if args.max_duration > 0 else None
    train_samples = list(
        iter_afrispeech_samples(
            stream_hf_afrispeech(args.dataset, args.dataset_config, "train", seed=args.seed),
            limit=args.train_limit,
            kenyan_only=kenyan_only,
            max_duration_s=max_dur,
            clean_transcripts=bool(args.clean_transcripts),
        )
    )
    if args.local_manifest is not None:
        train_samples.extend(load_manifest_samples(args.local_manifest))
    if not train_samples:
        print("error: no training samples collected", file=sys.stderr)
        return 1

    train_kw: dict[str, Any] = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.lora_targets,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "save_best": bool(args.save_best),
        "load_in_8bit": args.load_in_8bit,
        "gradient_checkpointing": grad_ckpt,
    }

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
    saved_dir = train_lora(train_samples, eval_samples, out_dir=args.out_dir, **train_kw)
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
                        "n_train": len(train_samples),
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
    print(f"Saved merged LoRA Whisper model to {saved_dir} ({len(train_samples)} train samples)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
