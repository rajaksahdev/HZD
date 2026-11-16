"""Raw zone (immutable downloads) and clean zone (harmonised files), plus the download log."""

import hashlib
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from api.config import settings

MANIFEST = "manifest.jsonl"
USER_AGENT = "hazardlens-student-project"


def raw_dir(source: str, day: date) -> Path:
    """raw/{source}/YYYY/MM/DD"""
    return settings.raw_dir / source / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"


def clean_path(source: str, day: date) -> Path:
    """processed/{source}/YYYY/MM/DD.nc"""
    return settings.processed_dir / source / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.nc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_download(source: str, day: date, path: Path, url: str) -> None:
    """Append one line per raw file: what was downloaded, from where, when, and its checksum."""
    entry = {
        "source": source,
        "day": day.isoformat(),
        "file": path.relative_to(settings.raw_dir).as_posix(),
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    with (settings.raw_dir / MANIFEST).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def http_get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 60):
    """GET with retries and backoff. Retries on network errors, 429 and 5xx."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} from {url}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status != 429 and status < 500:
                raise  # a 4xx will not fix itself
            last = exc
            if attempt < retries - 1:
                time.sleep(5 * 2**attempt)
    raise RuntimeError(f"Giving up on {url} after {retries} tries: {last}")


def write_raw(source: str, day: date, name: str, content: bytes, url: str) -> Path:
    """Write a new raw file. Refuses to overwrite: raw data is never edited."""
    path = raw_dir(source, day) / name
    if path.exists():
        raise FileExistsError(f"Raw file already exists, not overwriting: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)
    log_download(source, day, path, url)
    return path


def latest_raw(source: str, day: date, pattern: str) -> Path | None:
    files = sorted(raw_dir(source, day).glob(pattern))
    return files[-1] if files else None
