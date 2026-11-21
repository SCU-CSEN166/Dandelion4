"""Bootstrap and package runtime models for CPU/GPU deployment from GitHub Releases."""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR = REPO_ROOT / ".models"
GITHUB_REPO = "SCU-CSEN166/Dandelion4"
DEFAULT_RELEASE_TAG = "models-v5-int8"
WHISPER_ASSET_NAME = "whisper-large-v3-tuned-int8.tar.gz"
WHISPER_HF_ARCHIVE_NAME = "whisper-large-v3-tuned.tar.gz"
WHISPER_HF_ASSET_PARTS: tuple[str, str] = (
    "whisper-large-v3-tuned.tar.gz.part0",
    "whisper-large-v3-tuned.tar.gz.part1",
)
MAX_RELEASE_PART_BYTES = 1_800 * 1024 * 1024
NLP_ASSET_NAME = "nlp-models.tar.gz"


def _cuda_is_available() -> bool:
    try:
        import torch

        cuda_mod = getattr(torch, "cuda", None)
        return bool(cuda_mod is not None and cuda_mod.is_available())
    except Exception:
        return False


def is_whisper_int8_ready(models_dir: Path = MODELS_DIR) -> bool:
    """Return True if the tuned Whisper INT8 CTranslate2 model is present."""
    return (models_dir / "whisper-large-v3-tuned-int8" / "model.bin").is_file()


def is_whisper_hf_ready(models_dir: Path = MODELS_DIR) -> bool:
    """Return True if the tuned Hugging Face Whisper model is present for GPU."""
    hf_dir = models_dir / "whisper-large-v3-tuned"
    return (hf_dir / "config.json").is_file() and (
        (hf_dir / "model.safetensors").is_file() or (hf_dir / "pytorch_model.bin").is_file()
    )


def are_nlp_models_ready(models_dir: Path = MODELS_DIR) -> bool:
    """Return True if local GLiNER and DeBERTa directories are present."""
    gliner_dir = models_dir / "gliner_medium-v2.1"
    deberta_dir = models_dir / "deberta-v3-base"
    has_gliner_weights = (gliner_dir / "pytorch_model.bin").is_file() or (
        gliner_dir / "model.safetensors"
    ).is_file()
    return has_gliner_weights and (deberta_dir / "config.json").is_file()


def are_models_ready(
    models_dir: Path = MODELS_DIR,
    *,
    include_gpu_hf: bool | None = None,
) -> bool:
    """Return True if all required runtime models for this machine are present locally."""
    need_gpu_hf = _cuda_is_available() if include_gpu_hf is None else include_gpu_hf
    asr_ok = is_whisper_int8_ready(models_dir) and (
        not need_gpu_hf or is_whisper_hf_ready(models_dir)
    )
    return asr_ok and are_nlp_models_ready(models_dir)


def _safe_extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into *dest_dir*, rejecting path traversal."""
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            target = (dest_resolved / member.name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"Unsafe archive member path: {member.name}")
        try:
            tqdm_cls = importlib.import_module("tqdm").tqdm
            iterator: object = tqdm_cls(
                members, desc=f"Extracting {archive_path.name}", unit="file"
            )
        except ImportError:
            iterator = members
        tf.extractall(dest_resolved, members=iterator, filter="data")  # type: ignore[arg-type]


def _download_release_asset(repo: str, tag: str, asset_name: str, dest_file: Path) -> None:
    """Download *asset_name* from a GitHub Release with a progress bar, falling back to `gh`."""
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, dest_file.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            try:
                tqdm_cls = importlib.import_module("tqdm").tqdm
                bar: Any = tqdm_cls(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=asset_name,
                )
            except ImportError:
                bar = None
            try:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()
        return
    except Exception as url_exc:
        gh = shutil.which("gh")
        if gh is None:
            raise RuntimeError(
                f"Failed to download {url} and `gh` CLI is not available"
            ) from url_exc
        subprocess.run(
            [
                gh,
                "release",
                "download",
                tag,
                "--repo",
                repo,
                "--pattern",
                asset_name,
                "--dir",
                str(dest_file.parent),
                "--clobber",
            ],
            check=True,
        )


def ensure_runtime_models(
    models_dir: Path = MODELS_DIR,
    *,
    repo: str = GITHUB_REPO,
    tag: str = DEFAULT_RELEASE_TAG,
    include_gpu_hf: bool | None = None,
) -> Path:
    """Ensure required Whisper (HF on CUDA, INT8 on CPU), GLiNER, and DeBERTa exist locally."""
    need_gpu_hf = _cuda_is_available() if include_gpu_hf is None else include_gpu_hf
    missing_single: list[str] = []
    if not is_whisper_int8_ready(models_dir):
        missing_single.append(WHISPER_ASSET_NAME)
    if not are_nlp_models_ready(models_dir):
        missing_single.append(NLP_ASSET_NAME)
    need_hf_download = need_gpu_hf and not is_whisper_hf_ready(models_dir)

    if not missing_single and not need_hf_download:
        return models_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for asset in missing_single:
            archive = tmp_dir / asset
            print(f"Downloading {asset} from {repo} ({tag})...", flush=True)
            _download_release_asset(repo, tag, asset, archive)
            print(f"Extracting {asset} into {models_dir}...", flush=True)
            _safe_extract_tar_gz(archive, models_dir)
            archive.unlink(missing_ok=True)

        if need_hf_download:
            combined = tmp_dir / WHISPER_HF_ARCHIVE_NAME
            with combined.open("wb") as out_fh:
                for part_name in WHISPER_HF_ASSET_PARTS:
                    part_path = tmp_dir / part_name
                    print(f"Downloading {part_name} from {repo} ({tag})...", flush=True)
                    _download_release_asset(repo, tag, part_name, part_path)
                    with part_path.open("rb") as in_fh:
                        shutil.copyfileobj(in_fh, out_fh)
                    part_path.unlink(missing_ok=True)
            print(f"Extracting {WHISPER_HF_ARCHIVE_NAME} into {models_dir}...", flush=True)
            _safe_extract_tar_gz(combined, models_dir)
            combined.unlink(missing_ok=True)
    return models_dir


def package_gpu_model(
    models_dir: Path = MODELS_DIR,
    out_dir: Path = REPO_ROOT / ".temp" / "release_assets",
    *,
    max_part_bytes: int = MAX_RELEASE_PART_BYTES,
) -> list[Path]:
    """Package ``whisper-large-v3-tuned`` into <=2 GB split .tar.gz parts for GitHub Releases."""
    if not is_whisper_hf_ready(models_dir):
        raise FileNotFoundError(f"whisper-large-v3-tuned not found in {models_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw_archive = Path(tmp) / WHISPER_HF_ARCHIVE_NAME
        with tarfile.open(raw_archive, "w:gz") as tf:
            tf.add(models_dir / "whisper-large-v3-tuned", arcname="whisper-large-v3-tuned")
        total_size = raw_archive.stat().st_size
        part_size = (
            max(1, (total_size + 1) // 2)
            if total_size <= 2 * max_part_bytes
            else max_part_bytes
        )
        parts: list[Path] = []
        with raw_archive.open("rb") as src:
            idx = 0
            while True:
                chunk = src.read(part_size)
                if not chunk:
                    break
                part_path = out_dir / f"{WHISPER_HF_ARCHIVE_NAME}.part{idx}"
                part_path.write_bytes(chunk)
                parts.append(part_path)
                idx += 1
    return parts


def package_models(
    models_dir: Path = MODELS_DIR,
    out_dir: Path = REPO_ROOT / ".temp" / "release_assets",
) -> list[Path]:
    """Create compressed .tar.gz archives for GitHub Releases (each < 2 GB)."""
    if not (is_whisper_int8_ready(models_dir) and are_nlp_models_ready(models_dir)):
        raise FileNotFoundError(f"Required runtime models not found in {models_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    whisper_archive = out_dir / WHISPER_ASSET_NAME
    with tarfile.open(whisper_archive, "w:gz") as tf:
        tf.add(models_dir / "whisper-large-v3-tuned-int8", arcname="whisper-large-v3-tuned-int8")

    nlp_archive = out_dir / NLP_ASSET_NAME
    with tarfile.open(nlp_archive, "w:gz") as tf:
        tf.add(models_dir / "gliner_medium-v2.1", arcname="gliner_medium-v2.1")
        tf.add(models_dir / "deberta-v3-base", arcname="deberta-v3-base")

    archives = [whisper_archive, nlp_archive]
    if is_whisper_hf_ready(models_dir):
        archives.extend(package_gpu_model(models_dir, out_dir))
    return archives


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for model setup and release packaging."""
    parser = argparse.ArgumentParser(
        description="Check/download runtime models from GitHub Releases or package them."
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--repo", default=GITHUB_REPO)
    parser.add_argument("--tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument(
        "--package",
        action="store_true",
        help="Package local .models/ into .tar.gz archives for a GitHub Release.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / ".temp" / "release_assets",
        help="Output directory for --package.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """Run the runtime model setup or release-packaging CLI."""
    args = parse_args(argv)
    if args.package:
        archives = package_models(args.models_dir, args.out_dir)
        for path in archives:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"Created {path} ({size_mb:.1f} MB)", flush=True)
        return 0
    ensure_runtime_models(args.models_dir, repo=args.repo, tag=args.tag)
    print(f"All runtime models ready in {args.models_dir}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
