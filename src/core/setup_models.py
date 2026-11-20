"""Bootstrap and package runtime models for CPU-only edge deployment from GitHub Releases."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR = REPO_ROOT / "models"
GITHUB_REPO = "SCU-CSEN166/Dandelion4"
DEFAULT_RELEASE_TAG = "models-v5-int8"
WHISPER_ASSET_NAME = "whisper-large-v3-tuned-int8.tar.gz"
NLP_ASSET_NAME = "nlp-models.tar.gz"


def is_whisper_int8_ready(models_dir: Path = MODELS_DIR) -> bool:
    """Return True if the tuned Whisper INT8 CTranslate2 model is present."""
    return (models_dir / "whisper-large-v3-tuned-int8" / "model.bin").is_file()


def are_nlp_models_ready(models_dir: Path = MODELS_DIR) -> bool:
    """Return True if local GLiNER and DeBERTa directories are present."""
    gliner_dir = models_dir / "gliner_medium-v2.1"
    deberta_dir = models_dir / "deberta-v3-base"
    has_gliner_weights = (gliner_dir / "pytorch_model.bin").is_file() or (
        gliner_dir / "model.safetensors"
    ).is_file()
    return has_gliner_weights and (deberta_dir / "config.json").is_file()


def are_models_ready(models_dir: Path = MODELS_DIR) -> bool:
    """Return True if all required runtime models are present locally."""
    return is_whisper_int8_ready(models_dir) and are_nlp_models_ready(models_dir)


def _safe_extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into *dest_dir*, rejecting path traversal."""
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest_resolved / member.name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"Unsafe archive member path: {member.name}")
        tf.extractall(dest_resolved, filter="data")


def _download_release_asset(repo: str, tag: str, asset_name: str, dest_file: Path) -> None:
    """Download *asset_name* from a GitHub Release, falling back to `gh` if needed."""
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, dest_file.open("wb") as out:
            shutil.copyfileobj(resp, out)
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
) -> Path:
    """Ensure Whisper INT8, GLiNER, and DeBERTa exist under *models_dir*."""
    missing_assets: list[str] = []
    if not is_whisper_int8_ready(models_dir):
        missing_assets.append(WHISPER_ASSET_NAME)
    if not are_nlp_models_ready(models_dir):
        missing_assets.append(NLP_ASSET_NAME)

    if not missing_assets:
        return models_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for asset in missing_assets:
            archive = tmp_dir / asset
            print(f"Downloading {asset} from {repo} ({tag})...", flush=True)
            _download_release_asset(repo, tag, asset, archive)
            print(f"Extracting {asset} into {models_dir}...", flush=True)
            _safe_extract_tar_gz(archive, models_dir)
    return models_dir


def package_models(
    models_dir: Path = MODELS_DIR,
    out_dir: Path = REPO_ROOT / ".temp" / "release_assets",
) -> list[Path]:
    """Create compressed .tar.gz archives for GitHub Releases (each < 2 GB)."""
    if not are_models_ready(models_dir):
        raise FileNotFoundError(f"Required runtime models not found in {models_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    whisper_archive = out_dir / WHISPER_ASSET_NAME
    with tarfile.open(whisper_archive, "w:gz") as tf:
        tf.add(models_dir / "whisper-large-v3-tuned-int8", arcname="whisper-large-v3-tuned-int8")

    nlp_archive = out_dir / NLP_ASSET_NAME
    with tarfile.open(nlp_archive, "w:gz") as tf:
        tf.add(models_dir / "gliner_medium-v2.1", arcname="gliner_medium-v2.1")
        tf.add(models_dir / "deberta-v3-base", arcname="deberta-v3-base")

    return [whisper_archive, nlp_archive]


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
        help="Package local models/ into .tar.gz archives for a GitHub Release.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / ".temp" / "release_assets",
        help="Output directory for --package.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
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
