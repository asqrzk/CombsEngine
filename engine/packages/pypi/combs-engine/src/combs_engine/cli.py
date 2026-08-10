"""`combs` console-script shim.

Resolution order:
  1. ``COMBS_BIN`` env var (explicit path to a combs binary)
  2. A binary bundled in the wheel under ``combs_engine/vendor/``
     (CI builds per-platform platlib wheels this way)
  3. Download the matching GitHub Release asset into
     ``~/.cache/combs/bin`` (first run only; cached afterwards)

Asset names (produced by .github/workflows/release.yml):
  combs-<version>-macos-arm64.tar.gz / combs-<version>-macos-x86_64.tar.gz
  combs-<version>-linux-x86_64.tar.gz / combs-<version>-windows-x86_64.zip
"""

from __future__ import annotations

import io
import os
import platform
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from combs_engine import __version__

REPO = os.environ.get("COMBS_RELEASE_REPO", "asqrzk/CombsEngine")


def _asset_name() -> str:
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64", "AMD64": "x86_64"}.get(
        platform.machine()
    )
    if not system or not arch:
        raise RuntimeError(f"unsupported platform: {platform.system()}/{platform.machine()}")
    ext = "zip" if system == "windows" else "tar.gz"
    return f"combs-{__version__}-{system}-{arch}.{ext}"


def _cache_dir() -> Path:
    home = os.environ.get("COMBS_HOME") or str(Path.home() / ".cache" / "combs")
    return Path(home) / "bin"


def _bundled_binary() -> Path | None:
    vendor = Path(__file__).parent / "vendor"
    name = "combs.exe" if platform.system() == "Windows" else "combs"
    if vendor.is_dir():
        for candidate in vendor.rglob(name):
            return candidate
    return None


def _download_binary() -> Path:
    asset = _asset_name()
    url = f"https://github.com/{REPO}/releases/download/v{__version__}/{asset}"
    dest = _cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    inner = asset.removesuffix(".tar.gz").removesuffix(".zip")
    out_dir = dest / inner
    name = "combs.exe" if asset.endswith(".zip") else "combs"
    binary = out_dir / name
    if binary.exists():
        return binary

    print(f"[combs-engine] downloading {asset}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 (release URL)
            payload = resp.read()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"could not download {url}: {e}\n"
            "no prebuilt binary for this platform/version yet — build from source:\n"
            "  cargo install --path Engine/Core/combs-cli"
        ) from e

    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
            tf.extractall(dest, filter="data")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return binary


def _find_binary() -> Path:
    if env := os.environ.get("COMBS_BIN"):
        return Path(env)
    if bundled := _bundled_binary():
        return bundled
    return _download_binary()


def main() -> None:
    binary = _find_binary()
    sys.exit(subprocess.call([str(binary), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
