"""Evaluate Whisper ASR checkpoints across AfriSpeech, Kenyan/EA, and FLEURS benchmarks.

Computes ``regex_wer``, ``whisper_norm_wer``, and ``fair_norm_wer`` (with ``domain`` and
``accent`` breakdowns) for both greedy (``num_beams=1``) and beam-3 (``num_beams=3``) decoding.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import numpy.typing as npt
import soundfile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr import transcribe  # noqa: E402
from src.training import finetune_whisper  # noqa: E402

KENYAN_EA_TEST_SHARDS: tuple[str, ...] = (
    "data/test-kikuyu.parquet",
    "data/test-luhya.parquet",
    "data/test-luo.parquet",
    "data/test-meru.parquet",
    "data/test-kiswahili.parquet",
    "data/test-luganda.parquet",
    "data/test-kinyarwanda.parquet",
)

_CONTRACTION_FIXES: dict[str, str] = {
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "heres": "here's",
    "im": "i'm",
    "isnt": "isn't",
    "ive": "i've",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "weve": "we've",
    "whats": "what's",
    "whos": "who's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
    "youve": "you've",
}
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTION_FIXES)) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkItem:
    """One held-out audio clip with reference text and optional slice labels."""

    audio: npt.NDArray[np.float32]
    text: str
    domain: str = ""
    accent: str = ""


def regex_normalize(text: str) -> str:
    """Match ``evaluation/eval_wer.py`` normalization (lowercase, strip apostrophes)."""
    cleaned = text.lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-z0-9]+", " ", cleaned).strip()


def fix_missing_apostrophes(text: str) -> str:
    """Restore common missing apostrophes in crowd-sourced AfriSpeech references."""
    return _CONTRACTION_RE.sub(lambda m: _CONTRACTION_FIXES[m.group(0).lower()], text)


def _decode_row_audio(row: dict[str, Any]) -> tuple[npt.NDArray[np.float32], str] | None:
    raw_text = str(row.get("transcript") or row.get("transcription") or "")
    text = " ".join(raw_text.split())
    audio_info = row.get("audio")
    if not text or not isinstance(audio_info, dict):
        return None
    if "array" in audio_info and "sampling_rate" in audio_info:
        arr = np.asarray(audio_info["array"])
        sr = int(audio_info["sampling_rate"])
    elif audio_info.get("bytes"):
        raw, sr = soundfile.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
        arr = np.asarray(raw)
    elif audio_info.get("path"):
        raw, sr = soundfile.read(audio_info["path"], dtype="float32")
        arr = np.asarray(raw)
    else:
        return None
    wave = finetune_whisper.resample_to_16k(arr, sr)
    if len(wave) > 30 * transcribe.ASR_SAMPLE_RATE:
        return None
    return wave, text


def _rows_to_items(
    rows: Iterable[dict[str, Any]], *, limit: int | None = None
) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for row in rows:
        decoded = _decode_row_audio(row)
        if decoded is None:
            continue
        wave, text = decoded
        domain = str(row.get("domain") or "").strip().lower()
        accent = str(row.get("accent") or "").strip().lower()
        items.append(BenchmarkItem(audio=wave, text=text, domain=domain, accent=accent))
        if limit is not None and len(items) >= limit:
            break
    return items


def _load_local_suite(name: str) -> list[BenchmarkItem]:
    """Load a prepared public-speech suite without network access."""
    path = REPO_ROOT / "evaluation" / "datasets" / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"offline benchmark dataset missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        audio = data["audio"]
        offsets = data["offsets"]
        texts, domains, accents = data["text"], data["domain"], data["accent"]
        if (
            len(offsets) != len(texts) + 1
            or len(domains) != len(texts)
            or len(accents) != len(texts)
            or offsets[0] != 0
            or offsets[-1] != len(audio)
            or np.any(np.diff(offsets) < 0)
        ):
            raise ValueError(f"invalid offline benchmark dataset: {path}")
        return [
            BenchmarkItem(
                audio=audio[offsets[i] : offsets[i + 1]],
                text=str(texts[i]),
                domain=str(domains[i]),
                accent=str(accents[i]),
            )
            for i in range(len(texts))
        ]


def load_test_300_seed1234() -> list[BenchmarkItem]:
    """Load the prepared 300-clip seeded AfriSpeech test benchmark."""
    return _load_local_suite("test_300_seed1234")


def load_test_330_balanced() -> list[BenchmarkItem]:
    """Load the prepared 330-clip balanced AfriSpeech test benchmark."""
    return _load_local_suite("test_330_balanced")


def load_kenyan_ea_test() -> list[BenchmarkItem]:
    """Load the prepared Kenyan/East-African AfriSpeech-200 benchmark."""
    return _load_local_suite("kenyan_ea_test")


def load_fleurs_all(
    manifest_path: Path = REPO_ROOT / "examples" / "fleurs_manifest.json",
) -> list[BenchmarkItem]:
    """Load all clips from ``examples/fleurs_manifest.json``."""
    manifest = transcribe.load_manifest(manifest_path)
    items: list[BenchmarkItem] = []
    for clip in manifest["clips"]:
        ref = str(clip.get("reference_text") or "").strip()
        if not ref:
            continue
        wav_path = transcribe.clip_path(clip, REPO_ROOT)
        wave, sr = transcribe.read_wav(wav_path)
        items.append(BenchmarkItem(audio=transcribe.load_for_asr(wave, sr), text=ref))
    return items


def compute_corpus_wer(
    refs: Sequence[str],
    hyps: Sequence[str],
    normalizer: Callable[[str], str],
) -> float:
    """Compute corpus WER (0-100%, rounded to 2 decimals) after applying *normalizer*."""
    norm_refs: list[str] = []
    norm_hyps: list[str] = []
    for ref, hyp in zip(refs, hyps, strict=True):
        rn = normalizer(ref).strip()
        hn = normalizer(hyp).strip()
        if not rn:
            continue
        norm_refs.append(rn)
        norm_hyps.append(hn)
    if not norm_refs:
        return 0.0
    return round(100.0 * float(jiwer.wer(norm_refs, norm_hyps)), 2)


def score_predictions(
    items: Sequence[BenchmarkItem],
    hyps: Sequence[str],
    whisper_norm: Callable[[str], str],
) -> dict[str, Any]:
    """Score hypotheses against benchmark items with overall, domain, and accent WER."""
    refs = [item.text for item in items]

    def fair_ref_norm(text: str) -> str:
        return str(whisper_norm(fix_missing_apostrophes(text)))

    def fair_hyp_norm(text: str) -> str:
        return str(whisper_norm(text))

    # Build paired fair-normalized lists so slice stats match corpus jiwer computation.
    def _pair_wer(sub_indices: Sequence[int], *, use_fair: bool) -> float:
        r_list: list[str] = []
        h_list: list[str] = []
        for idx in sub_indices:
            rn = fair_ref_norm(refs[idx]) if use_fair else str(whisper_norm(refs[idx]))
            hn = fair_hyp_norm(hyps[idx]) if use_fair else str(whisper_norm(hyps[idx]))
            rn = rn.strip()
            hn = hn.strip()
            if not rn:
                continue
            r_list.append(rn)
            h_list.append(hn)
        return round(100.0 * float(jiwer.wer(r_list, h_list)), 2) if r_list else 0.0

    all_idx = list(range(len(items)))
    domain_groups: dict[str, list[int]] = defaultdict(list)
    accent_groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        if item.domain:
            domain_groups[item.domain].append(idx)
        if item.accent:
            accent_groups[item.accent].append(idx)

    return {
        "n": len(items),
        "regex_wer": compute_corpus_wer(refs, hyps, regex_normalize),
        "whisper_norm_wer": _pair_wer(all_idx, use_fair=False),
        "fair_norm_wer": _pair_wer(all_idx, use_fair=True),
        "domain": {
            key: {"n": len(idxs), "fair_wer": _pair_wer(idxs, use_fair=True)}
            for key, idxs in domain_groups.items()
        },
        "accent": {
            key: {"n": len(idxs), "fair_wer": _pair_wer(idxs, use_fair=True)}
            for key, idxs in sorted(accent_groups.items(), key=lambda kv: -len(kv[1]))
        },
    }


def _load_whisper_normalizer() -> Callable[[str], str]:
    from transformers import WhisperTokenizer

    tok_dir = (
        transcribe.BASE_MODEL_DIR
        if transcribe.BASE_MODEL_DIR.is_dir()
        else transcribe.FINETUNED_INT8_MODEL_DIR
    )
    tok = WhisperTokenizer.from_pretrained(str(tok_dir), local_files_only=True)

    def _norm(text: str) -> str:
        return str(tok.normalize(text))

    return _norm


def run_benchmarks(
    model_id: str,
    *,
    suites: Sequence[str] = (
        "test_300_seed1234",
        "test_330_balanced",
        "kenyan_ea_test",
        "fleurs_all",
    ),
    modes: Sequence[str] = ("greedy", "beam3"),
    batch_size: int = 8,
    device: str = "auto",
    initial_prompt: str | None = None,
) -> dict[str, Any]:  # pragma: no cover - heavy GPU evaluation
    """Evaluate *model_id* across the requested real-speech suites and decoding modes."""
    import torch

    loaders: dict[str, Callable[[], list[BenchmarkItem]]] = {
        "test_300_seed1234": load_test_300_seed1234,
        "test_330_balanced": load_test_330_balanced,
        "kenyan_ea_test": load_kenyan_ea_test,
        "fleurs_all": load_fleurs_all,
    }
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        from evaluation import fetch_public_datasets

        for name in suites:
            if name in loaders:
                fetch_public_datasets.prepare_suite(name)
    loaded_suites = {name: loaders[name]() for name in suites}
    whisper_norm = _load_whisper_normalizer()
    resolved_device = transcribe.resolve_device(device)
    beam_map = {"greedy": 1, "beam3": 3, "beam5": 5}

    results: dict[str, Any] = {"model": model_id}
    for mode in modes:
        num_beams = beam_map[mode]
        asr = transcribe.make_whisper(
            model_id,
            resolved_device,
            batch_size=batch_size,
            initial_prompt=initial_prompt,
            num_beams=num_beams,
        )
        asr.load()
        mode_out: dict[str, Any] = {}
        for name, items in loaded_suites.items():
            print(f"[{mode}] Evaluating {name} ({len(items)} clips)...", flush=True)
            hyps = asr([item.audio for item in items])
            mode_out[name] = score_predictions(items, hyps, whisper_norm)
            s = mode_out[name]
            print(
                f"  -> {name}: regex={s['regex_wer']:.2f}% "
                f"whisper={s['whisper_norm_wer']:.2f}% fair={s['fair_norm_wer']:.2f}%",
                flush=True,
            )
        results[mode] = mode_out
        del asr
        gc.collect()
        if resolved_device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-benchmark Whisper WER evaluations (AfriSpeech, Kenyan/EA, FLEURS)."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["test_300_seed1234", "test_330_balanced", "kenyan_ea_test", "fleurs_all"],
        choices=["test_300_seed1234", "test_330_balanced", "kenyan_ea_test", "fleurs_all"],
    )
    parser.add_argument(
        "--modes", nargs="+", default=["greedy", "beam3"], choices=["greedy", "beam3", "beam5"]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initial-prompt", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - GPU CLI
    args = parse_args(argv)
    report = run_benchmarks(
        args.model,
        suites=args.suites,
        modes=args.modes,
        batch_size=args.batch_size,
        device=args.device,
        initial_prompt=args.initial_prompt,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved benchmark evaluation report -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
