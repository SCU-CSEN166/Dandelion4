"""Offline unit tests for src/finetune_whisper.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile
import torch

import finetune_whisper
from finetune_whisper import (
    ASR_SAMPLE_RATE,
    LazyWhisperDataset,
    TrainingSample,
    WhisperDataCollator,
    is_kenyan_accent,
    iter_afrispeech_samples,
    load_manifest_samples,
    make_k_folds,
    normalize_all_caps_transcript,
    parse_args,
    prepare_feature_dict,
    resample_to_16k,
)


def test_is_kenyan_accent() -> None:
    assert is_kenyan_accent("Kenyan English")
    assert is_kenyan_accent("Kikuyu")
    assert is_kenyan_accent("kiswahili")
    assert not is_kenyan_accent("Yoruba")
    assert not is_kenyan_accent(None)
    assert not is_kenyan_accent("")


def test_resample_to_16k_stereo_and_8k() -> None:
    stereo_8k = np.ones((800, 2), dtype=np.float32) * 0.25
    out = resample_to_16k(stereo_8k, 8000)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert len(out) == 1600

    with pytest.raises(ValueError, match="1-D or 2-D"):
        resample_to_16k(np.zeros((2, 2, 2), dtype=np.float32), 16000)


def test_iter_afrispeech_samples_filters_and_limits() -> None:
    rows = [
        {
            "accent": "Yoruba",
            "transcript": "skip me",
            "audio": {"array": [0.1], "sampling_rate": 16000},
        },
        {"accent": "Kikuyu", "transcript": "", "audio": {"array": [0.1], "sampling_rate": 16000}},
        {
            "accent": "Kenyan English",
            "transcript": "We visited Narok.",
            "audio": {"array": [0.0, 0.1, -0.1], "sampling_rate": 16000},
        },
        {
            "accent": "Luo",
            "transcript": "Screened 12 patients.",
            "audio": {"array": [0.0, 0.2], "sampling_rate": 16000},
        },
    ]
    kenyan = list(iter_afrispeech_samples(rows, limit=1, kenyan_only=True))
    assert len(kenyan) == 1
    assert kenyan[0].text == "We visited Narok."
    assert kenyan[0].audio.dtype == np.float32

    all_accents = list(iter_afrispeech_samples(rows, kenyan_only=False))
    assert [s.text for s in all_accents] == [
        "skip me",
        "We visited Narok.",
        "Screened 12 patients.",
    ]

    with pytest.raises(ValueError, match="limit must be >= 1"):
        list(iter_afrispeech_samples(rows, limit=0))


def test_load_manifest_samples(tmp_path: Path) -> None:
    wav_path = tmp_path / "examples" / "clip.wav"
    wav_path.parent.mkdir(parents=True)
    soundfile.write(wav_path, np.zeros(1600, dtype=np.float32), ASR_SAMPLE_RATE)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clips": [
                    {"path": "examples/clip.wav", "reference_text": "Visited Kisumu."},
                    {"path": "examples/clip.wav", "reference_text": ""},
                ]
            }
        ),
        encoding="utf-8",
    )
    samples = load_manifest_samples(manifest_path, root=tmp_path)
    assert len(samples) == 1
    assert samples[0].text == "Visited Kisumu."
    assert len(samples[0].audio) == 1600

    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'clips' list"):
        load_manifest_samples(bad_manifest, root=tmp_path)


class _FakeFeatureExtractor:
    def __call__(self, audio: Any, sampling_rate: int) -> Any:
        assert sampling_rate == ASR_SAMPLE_RATE
        return type("FE", (), {"input_features": [np.zeros((80, 10), dtype=np.float32)]})()

    def pad(self, items: list[dict[str, Any]], return_tensors: str) -> dict[str, Any]:
        assert return_tensors == "pt"
        return {"input_features": torch.stack([torch.tensor(i["input_features"]) for i in items])}


class _FakeTokenizer:
    def __call__(self, text: str) -> Any:
        return type("Tok", (), {"input_ids": [50258, 10, 20, len(text)]})()

    def pad(self, items: list[dict[str, Any]], return_tensors: str) -> Any:
        assert return_tensors == "pt"
        max_len = max(len(i["input_ids"]) for i in items)
        padded = [i["input_ids"] + [0] * (max_len - len(i["input_ids"])) for i in items]
        mask = [[1] * len(i["input_ids"]) + [0] * (max_len - len(i["input_ids"])) for i in items]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)}


class _FakeProcessor:
    feature_extractor = _FakeFeatureExtractor()
    tokenizer = _FakeTokenizer()


def test_prepare_feature_dict_and_collator() -> None:
    proc = _FakeProcessor()
    samples = [
        TrainingSample(np.zeros(100, dtype=np.float32), "hi"),
        TrainingSample(np.zeros(100, dtype=np.float32), "hello"),
    ]
    ds = LazyWhisperDataset(samples, proc)
    assert len(ds) == 2
    f1 = prepare_feature_dict(samples[0], proc)
    f2 = ds[1]
    collator = WhisperDataCollator(proc, decoder_start_token_id=50258)
    batch = collator([f1, f2])
    assert batch["input_features"].shape == (2, 80, 10)
    assert batch["labels"].shape == (2, 3)


def test_iter_afrispeech_max_duration_and_whitespace() -> None:
    rows = [
        {
            "accent": "swahili",
            "transcript": "too long\n",
            "audio": {"array": np.zeros(32000, dtype=np.float32), "sampling_rate": 16000},
        },
        {
            "accent": "swahili",
            "transcript": "short  clip\n",
            "audio": {"array": np.zeros(8000, dtype=np.float32), "sampling_rate": 16000},
        },
    ]
    out = list(iter_afrispeech_samples(rows, kenyan_only=False, max_duration_s=1.0))
    assert [s.text for s in out] == ["short clip"]
    assert normalize_all_caps_transcript("PAIN IS DIFFUSED\n  06-05") == "pain is diffused 06-05"
    assert normalize_all_caps_transcript("Patient has BP 120/80.") == "Patient has BP 120/80."


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.base_model == finetune_whisper.DEFAULT_BASE_MODEL
    assert args.dataset == finetune_whisper.DEFAULT_DATASET
    assert args.train_limit == 500
    assert args.kenyan_only is False
    assert args.k_folds == 1


def test_make_k_folds_disjoint_and_deterministic() -> None:
    samples = [TrainingSample(np.zeros(10, dtype=np.float32), f"s{i}") for i in range(10)]
    folds_a = make_k_folds(samples, k=5, seed=42)
    folds_b = make_k_folds(samples, k=5, seed=42)
    assert len(folds_a) == 5
    assert [v for _, v in folds_a] == [v for _, v in folds_b]
    all_val_texts = [s.text for _, val in folds_a for s in val]
    assert sorted(all_val_texts) == sorted(s.text for s in samples)
    for train, val in folds_a:
        assert len(train) == 8
        assert len(val) == 2
        assert not ({s.text for s in train} & {s.text for s in val})

    with pytest.raises(ValueError, match="k must be >= 2"):
        make_k_folds(samples, k=1)
    with pytest.raises(ValueError, match="need at least 5 samples"):
        make_k_folds(samples[:3], k=5)
