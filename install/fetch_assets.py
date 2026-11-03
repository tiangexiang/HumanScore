#!/usr/bin/env python3
"""HumanScore model files: manifest, downloader, SMPL-X placement and Hugging Face access check.

The manifest ``install/assets.json`` lists every model file: its official URL (and mirrors), size,
SHA-256, destination and license. Groups: ``core`` (needed for scoring; ``bash install.sh``),
``sam3`` (``--with-sam3``), ``selftest`` (``--self-test``) and ``manual`` (the licensed SMPL-X
model, which is never downloaded). Files go to ``outputs/assets/``; symbolic links at the paths
the code reads (``links`` in the manifest) point there. This script uses the Python standard
library only (Python 3.8 or newer), so ``install.sh`` can run it before the HumanScore environment
exists. Run it from any directory:

  python3 install/fetch_assets.py plan  [--groups core,sam3,selftest] [--json]
  python3 install/fetch_assets.py fetch [--groups core] [--only NAME ...] [--jobs 4] [--verify]
  python3 install/fetch_assets.py check [--groups core] [--verified]   # no network, no hashing
  python3 install/fetch_assets.py smplx [--source PATH]            # find / place SMPLX_NEUTRAL.npz
  python3 install/fetch_assets.py hf-access [--name sam3.1]        # gated Hugging Face access
  python3 install/fetch_assets.py hosts [--groups core]            # download hosts (preflight)
  python3 install/fetch_assets.py remove [--groups core,sam3,selftest]

Downloads go to ``DEST.part`` first. An interrupted download resumes from the partial file (HTTP
range requests; a server that ignores the range restarts it). Each attempt tries the URLs in order;
failed attempts are retried with exponential backoff. The finished file must have the manifest's
size and SHA-256, otherwise it is deleted and the next URL is tried. Verified files are recorded in
``outputs/assets/.humanscore-verified.json`` (size, mtime, SHA-256), so later runs check them
without hashing again (``--verify`` re-hashes).

Exit status: 0 ok, 1 error, 2 usage. ``smplx``: 3 when the model is missing. ``hf-access``: 0 access
granted, 4 no Hugging Face token, 5 token rejected, 6 access not granted (not requested or pending),
7 network error.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import random
import re
import shutil
import signal
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "install" / "assets.json"
RECORDS = "outputs/assets/.humanscore-verified.json"
MARKER = ".humanscore-asset.json"
GROUPS = ("core", "sam3", "selftest", "manual")
USER_AGENT = "HumanScore-installer (+https://cs.stanford.edu/~xtiange/projects/humanscore/; python-urllib)"
CHUNK = 4 * 1024 * 1024          # read/write/hash block size (bytes)
HF_HOST = "huggingface.co"

EXIT_MISSING_MANUAL = 3          # exit status of ``smplx`` when the model is missing
#: Set on Ctrl-C / SIGTERM: download threads stop after the current block (the .part file stays).
CANCEL = threading.Event()
#: Exit status of ``hf-access`` per state (install.sh tells the user what to do for each).
HF_EXIT = {"granted": 0, "no-token": 4, "invalid-token": 5, "not-granted": 6, "network": 7}


class AssetError(RuntimeError):
    """A download or verification failed."""


class ChecksumError(AssetError):
    """The downloaded bytes do not have the expected size or SHA-256."""


class ManifestError(ValueError):
    """install/assets.json is malformed."""


class Cancelled(Exception):
    """The download was interrupted (Ctrl-C or SIGTERM); partial files are kept for a resume."""


def _sleep(seconds: float) -> None:
    """Backoff that returns early, with :class:`Cancelled`, when the run is interrupted."""
    if CANCEL.wait(seconds):
        raise Cancelled("interrupted")


def log(message: str) -> None:
    """Print a progress line (prefixed ``[assets]``, flushed for the installer's log)."""
    print(f"[assets] {message}", flush=True)


def human(size: float) -> str:
    """A byte count in decimal units, e.g. ``5.7 GB``."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000.0
    return f"{size:.1f} TB"


# ----------------------------------------------------------------------------- manifest
def load_manifest(path: Path = MANIFEST) -> dict:
    """Read and validate the manifest (raises :class:`ManifestError` when it is malformed)."""
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping, require_https: bool = True) -> None:
    """Check names, groups, sizes, hashes, URLs and that every path stays inside the repository."""
    if manifest.get("version") != 1 or not isinstance(manifest.get("assets"), list):
        raise ManifestError("manifest must have version 1 and a list of assets")
    names, dests = set(), set()
    for asset in manifest["assets"]:
        name = asset.get("name")
        if not name or name in names:
            raise ManifestError(f"missing or duplicate asset name: {name!r}")
        names.add(name)
        for key in ("description", "group", "size", "sha256", "dest", "source", "license"):
            if key not in asset:
                raise ManifestError(f"{name}: missing field {key!r}")
        if asset["group"] not in GROUPS:
            raise ManifestError(f"{name}: unknown group {asset['group']!r}")
        if not isinstance(asset["size"], int) or asset["size"] <= 0:
            raise ManifestError(f"{name}: size must be a positive integer")
        if not re.fullmatch(r"[0-9a-f]{64}", str(asset["sha256"])):
            raise ManifestError(f"{name}: sha256 must be 64 lowercase hex digits")
        for rel in [asset["dest"], *asset.get("links", [])]:
            if not _safe_relative(rel):
                raise ManifestError(f"{name}: path must be relative and inside the repository: {rel!r}")
        if asset["dest"] in dests:
            raise ManifestError(f"{name}: duplicate destination {asset['dest']}")
        dests.add(asset["dest"])
        urls = asset.get("urls", [])
        if asset["group"] == "manual":
            if urls or "manual" not in asset:
                raise ManifestError(f"{name}: a manual asset has no URLs and a 'manual' section")
            continue
        if not urls:
            raise ManifestError(f"{name}: no URL")
        for url in urls:
            scheme = urllib.parse.urlsplit(url).scheme
            if scheme != "https" and (require_https or scheme not in ("http", "file")):
                raise ManifestError(f"{name}: URL must use https: {url}")
        extract = asset.get("extract")
        if extract is not None:
            if extract.get("format") != "tar.gz" or not extract.get("files"):
                raise ManifestError(f"{name}: extract needs format tar.gz and the expected files")
            for rel in extract["files"]:
                if not _safe_relative(rel):
                    raise ManifestError(f"{name}: unsafe extracted path {rel!r}")
        if asset.get("auth") not in (None, "huggingface"):
            raise ManifestError(f"{name}: unknown auth {asset.get('auth')!r}")


def _safe_relative(rel: str) -> bool:
    """A non-empty relative path without ``..``."""
    path = Path(rel)
    return bool(rel) and not path.is_absolute() and ".." not in path.parts


def select(manifest: Mapping, groups: Iterable[str] = ("core",), only: Sequence[str] = ()) -> List[dict]:
    """The assets of ``groups``, or the assets named in ``only`` (which takes precedence)."""
    assets = manifest["assets"]
    if only:
        known = {asset["name"] for asset in assets}
        unknown = [name for name in only if name not in known]
        if unknown:
            raise ManifestError(f"unknown asset(s): {', '.join(unknown)} (known: {', '.join(sorted(known))})")
        return [asset for asset in assets if asset["name"] in only]
    groups = set(groups)
    return [asset for asset in assets if asset["group"] in groups]


# ----------------------------------------------------------------------------- verification records
class Records:
    """``dest -> {size, mtime_ns, sha256}`` of verified files (thread-safe, written atomically)."""

    def __init__(self, root: Path):
        self.path = Path(root) / RECORDS
        self.lock = threading.Lock()
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def matches(self, rel: str, path: Path, sha256: str) -> bool:
        entry = self.data.get(rel)
        try:
            stat = path.stat()
        except OSError:
            return False
        return bool(entry) and entry.get("sha256") == sha256 and entry.get("size") == stat.st_size \
            and entry.get("mtime_ns") == stat.st_mtime_ns

    def add(self, rel: str, path: Path, sha256: str) -> None:
        stat = path.stat()
        with self.lock:
            self.data[rel] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256}
            self._write()

    def drop(self, rel: str) -> None:
        with self.lock:
            if self.data.pop(rel, None) is not None:
                self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        os.replace(tmp, self.path)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_status(asset: Mapping, root: Path, records: Optional[Records] = None, full: bool = False) -> str:
    """``ok`` (complete and verified), ``unverified`` (right size, not hashed yet), ``partial``
    (a ``.part`` download exists), ``mismatch`` (wrong size/content) or ``missing``."""
    dest = Path(root) / asset["dest"]
    if asset.get("extract"):
        state = extracted_state(asset, dest, full)
        return state or ("partial" if _part(dest, asset).exists() else "missing")
    if dest.is_file():
        if dest.stat().st_size != asset["size"]:
            return "mismatch"
        if not full and records is not None and records.matches(asset["dest"], dest, asset["sha256"]):
            return "ok"
        return "unverified"
    if _part(dest, asset).exists():
        return "partial"
    return "missing"


def _part(dest: Path, asset: Mapping) -> Path:
    suffix = ".tar.gz.part" if asset.get("extract") else ".part"
    return dest.with_name(dest.name + suffix)


def _read_marker(dest: Path) -> dict:
    try:
        marker = json.loads((dest / MARKER).read_text())
    except (OSError, ValueError):
        return {}
    return marker if isinstance(marker, dict) else {}


def extracted_state(asset: Mapping, dest: Path, full: bool = False) -> Optional[str]:
    """``ok`` (extracted from the verified archive and unchanged since: sizes and modification
    times as recorded in the marker), ``unverified`` (right sizes, but a file was modified since the
    extraction, or ``full``: to be compared with the SHA-256 recorded at extraction), or ``None``."""
    marker = _read_marker(dest)
    if marker.get("sha256") != asset["sha256"]:
        return None
    if not all((dest / rel).is_file() and (dest / rel).stat().st_size == size
               for rel, size in asset["extract"]["files"].items()):
        return None
    recorded = marker.get("files")
    if not isinstance(recorded, dict):
        return "ok"          # a marker without per-file records: the sizes are all it can check
    if full:
        return "unverified"
    for rel in asset["extract"]["files"]:
        entry = recorded.get(rel) or {}
        if entry.get("mtime_ns") != (dest / rel).stat().st_mtime_ns:
            return "unverified"
    return "ok"


def _file_records(asset: Mapping, directory: Path) -> Dict[str, dict]:
    """Size, modification time and SHA-256 of each extracted file (stored in the marker)."""
    files = {}
    for rel in asset["extract"]["files"]:
        stat = (directory / rel).stat()
        files[rel] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(directory / rel)}
    return files


def reverify_extracted(asset: Mapping, dest: Path) -> bool:
    """Compare the extracted files with the SHA-256 recorded at extraction; when they all match,
    record their current modification times (the next check is quick again)."""
    marker = _read_marker(dest)
    recorded = marker.get("files")
    if not isinstance(recorded, dict):
        return True
    for rel in asset["extract"]["files"]:
        if file_sha256(dest / rel) != (recorded.get(rel) or {}).get("sha256"):
            return False
    for rel in asset["extract"]["files"]:
        recorded[rel]["mtime_ns"] = (dest / rel).stat().st_mtime_ns
    (dest / MARKER).write_text(json.dumps(marker))
    return True


# ----------------------------------------------------------------------------- HTTP
class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects; the Authorization header is added unredirected, so it never leaves the
    host it was meant for (urllib copies only redirectable headers)."""


def build_opener() -> urllib.request.OpenerDirector:
    """The URL opener of the downloads (redirects followed, tokens not forwarded)."""
    return urllib.request.build_opener(_NoAuthOnRedirect())


def hf_token(environ: Mapping[str, str] = os.environ) -> Optional[str]:
    """The Hugging Face token, looked up like huggingface_hub does."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if environ.get(key, "").strip():
            return environ[key].strip()
    candidates = []
    if environ.get("HF_TOKEN_PATH"):
        candidates.append(Path(environ["HF_TOKEN_PATH"]).expanduser())
    if environ.get("HF_HOME"):
        candidates.append(Path(environ["HF_HOME"]).expanduser() / "token")
    cache = Path(environ.get("XDG_CACHE_HOME") or Path(environ.get("HOME", "~")) / ".cache").expanduser()
    candidates.append(cache / "huggingface" / "token")
    for path in candidates:
        try:
            token = path.read_text().strip()
        except OSError:
            continue
        if token:
            return token
    return None


def _request(url: str, offset: int = 0, token: Optional[str] = None, method: str = "GET") -> urllib.request.Request:
    """A request for ``url`` from byte ``offset`` on; the token is sent to Hugging Face only."""
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    if token and urllib.parse.urlsplit(url).hostname == HF_HOST:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def download(urls: Sequence[str], part: Path, size: int, sha256: str, *, token: Optional[str] = None,
             opener: Optional[urllib.request.OpenerDirector] = None, retries: int = 6, backoff: float = 2.0,
             max_backoff: float = 60.0, timeout: float = 60.0, sleep: Callable[[float], None] = _sleep,
             progress: Optional[Callable[[int, int, str], None]] = None, label: str = "") -> str:
    """Download into ``part`` (resuming it) until it has ``size`` bytes and ``sha256``; returns the URL
    used. Raises :class:`ChecksumError` when no URL delivers the right bytes, :class:`AssetError` when
    every attempt failed."""
    opener = opener or build_opener()
    part.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    bad_content = set()
    checksum_failed = False
    for attempt in range(retries + 1):
        for url in urls:
            if url in bad_content:
                continue
            resumed = part.exists() and part.stat().st_size > 0
            try:
                _transfer(url, part, size, token, opener, timeout, progress, label)
            except ChecksumError as exc:   # this source serves another file: do not use it again
                errors.append(f"{_short(url)}: {exc}")
                if part.exists():
                    part.unlink()
                bad_content.add(url)
                checksum_failed = True
                continue
            except (OSError, urllib.error.URLError, http.client.HTTPException, AssetError) as exc:
                errors.append(f"{_short(url)}: {_describe(exc)}")
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404, 410):
                    bad_content.add(url)   # permanent for this URL: try the next one, do not retry it
                continue
            digest = file_sha256(part)
            if digest == sha256:
                return url
            errors.append(f"{_short(url)}: SHA-256 {digest[:16]}... does not match {sha256[:16]}...")
            part.unlink()
            if resumed:
                continue      # the kept partial file may be the damaged part: download it once more in full
            bad_content.add(url)
            checksum_failed = True
        if len(bad_content) == len(urls):
            break
        if attempt < retries:
            delay = min(max_backoff, backoff * (2 ** attempt)) * (0.75 + 0.5 * random.random())
            log(f"{label}: attempt {attempt + 1} failed ({errors[-1]}); retrying in {delay:.0f} s")
            sleep(delay)
    detail = "; ".join(errors[-4:])
    if checksum_failed and all(url in bad_content for url in urls):
        raise ChecksumError(f"{label}: no source delivered the expected file ({detail})")
    raise AssetError(f"{label}: download failed after {retries + 1} attempts ({detail})")


def _transfer(url, part: Path, size: int, token, opener, timeout, progress, label) -> None:
    """One attempt: append the missing bytes of ``url`` to ``part`` until it has ``size`` bytes.

    Raises :class:`ChecksumError` when the server's file has another size (this source serves
    something else) and :class:`AssetError` or a network error when the attempt fails.
    """
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        part.unlink()
        offset = 0
    if offset == size:
        return
    with opener.open(_request(url, offset, token), timeout=timeout) as response:
        status = getattr(response, "status", None) or response.getcode()
        ctype = (response.headers.get("Content-Type") or "").lower()
        if ctype.startswith("text/html"):
            raise AssetError("got an HTML page instead of the file (quota, login or error page)")
        mode = "ab"
        if offset and status == 206:
            match = re.match(r"bytes (\d+)-", response.headers.get("Content-Range", ""))
            if not match or int(match.group(1)) != offset:
                raise AssetError(f"unexpected Content-Range {response.headers.get('Content-Range')!r}")
        elif offset:
            mode, offset = "wb", 0   # the server ignored the range request: start again
        length = response.headers.get("Content-Length")
        if length is not None and offset + int(length) != size:
            raise ChecksumError(f"{label}: server reports {offset + int(length)} bytes, expected {size}")
        with open(part, mode) as output:
            done = offset
            while True:
                if CANCEL.is_set():
                    raise Cancelled("interrupted")
                block = response.read(CHUNK)
                if not block:
                    break
                output.write(block)
                done += len(block)
                if done > size:
                    raise ChecksumError(f"{label}: more than the expected {size} bytes")
                if progress:
                    progress(done, size, url)
    final = part.stat().st_size
    if final != size:
        raise AssetError(f"connection closed after {final} of {size} bytes")


def _short(url: str) -> str:
    """The host of a URL, for messages."""
    return urllib.parse.urlsplit(url).netloc or url


def _describe(exc: BaseException) -> str:
    """A one-line description of a download error."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"{type(exc.reason).__name__}: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


class Progress:
    """Prints a line per asset every ``interval`` seconds or 10 % (log-friendly, no carriage returns)."""

    def __init__(self, label: str, interval: float = 15.0, clock: Callable[[], float] = time.monotonic):
        self.label, self.interval, self.clock = label, interval, clock
        self.start = clock()
        self.last_time, self.last_fraction, self.first_done = self.start, 0.0, None

    def __call__(self, done: int, total: int, url: str) -> None:
        now = self.clock()
        fraction = done / total if total else 1.0
        if self.first_done is None:   # first block: remember where this transfer started
            self.first_done, self.start, self.last_time, self.last_fraction = done, now, now, fraction
            return
        if done < total and now - self.last_time < self.interval and fraction - self.last_fraction < 0.1:
            return
        rate = (done - self.first_done) / max(now - self.start, 1e-6)
        log(f"{self.label}: {fraction * 100:5.1f} % of {human(total)} ({human(rate)}/s from {_short(url)})")
        self.last_time, self.last_fraction = now, fraction


# ----------------------------------------------------------------------------- archives and links
def safe_extract_tar(archive: Path, target: Path) -> None:
    """Extract a .tar.gz into ``target``: regular files and directories only, no absolute or parent
    paths (a leading ``/`` or ``./`` is stripped, as MeTRAbs archives start with ``/``)."""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name.lstrip("/")
            while name.startswith("./"):
                name = name[2:]
            if not name or name == ".":
                continue
            parts = Path(name).parts
            if ".." in parts:
                raise AssetError(f"unsafe path in archive: {member.name}")
            out = target.joinpath(*parts)
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                out.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                with source, open(out, "wb") as handle:
                    shutil.copyfileobj(source, handle, CHUNK)
            else:
                raise AssetError(f"unsupported archive member type: {member.name}")


def finish_extract(asset: Mapping, archive: Path, dest: Path) -> None:
    """Extract a verified archive next to ``dest``, check the files, then move it into place.

    The staging directory is renamed to ``dest`` only when every expected file has its expected
    size, so an interrupted or wrong extraction never leaves a half-filled model folder.
    """
    staging = dest.with_name(dest.name + ".extracting")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    safe_extract_tar(archive, staging)
    for rel, size in asset["extract"]["files"].items():
        path = staging / rel
        if not path.is_file() or path.stat().st_size != size:
            shutil.rmtree(staging, ignore_errors=True)
            raise ChecksumError(f"{asset['name']}: archive content differs from the manifest ({rel})")
    # Sizes, times and SHA-256 of the extracted files: a file changed later is detected (and, if its
    # content differs, extracted again from a new download).
    (staging / MARKER).write_text(json.dumps({"name": asset["name"], "sha256": asset["sha256"],
                                              "source": "install/fetch_assets.py",
                                              "files": _file_records(asset, staging)}))
    if dest.exists():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    os.replace(staging, dest)
    archive.unlink()


def links_ok(asset: Mapping, root: Path) -> bool:
    """Every link of ``links`` exists and leads to ``dest`` (or is a regular file of the right size,
    which :func:`ensure_links` leaves alone)."""
    target = (Path(root) / asset["dest"]).resolve()
    for rel in asset.get("links", []):
        link = Path(root) / rel
        if link.is_symlink():
            if link.resolve() != target:
                return False
        elif not (link.is_file() and link.stat().st_size == asset["size"]):
            return False
    return True


def ensure_links(asset: Mapping, root: Path) -> List[str]:
    """Symbolic links (relative) from ``links`` to ``dest``. A regular file already at a link path is
    left alone (a user's copy); a link that points elsewhere is replaced. Returns notes."""
    notes = []
    target = Path(root) / asset["dest"]
    for rel in asset.get("links", []):
        link = Path(root) / rel
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.exists():
            if link.is_file() and link.stat().st_size == asset["size"]:
                notes.append(f"{rel} is a regular file of the right size; left as it is")
            else:
                notes.append(f"{rel} exists and is not the expected file; move it away and re-run")
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
    return notes


# ----------------------------------------------------------------------------- disk
def _existing_parent(path: Path) -> Path:
    """The nearest existing directory at or above ``path`` (where its file system is measured)."""
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return path


def space_needed(assets: Iterable[Mapping], root: Path, records: Optional[Records] = None) -> Dict[str, dict]:
    """Bytes still to be written, by file system (device of the nearest existing parent): the
    download plus, for archives, the extracted files while the archive still exists."""
    needs: Dict[str, dict] = {}
    for asset in assets:
        if asset.get("group") == "manual":
            continue
        status = asset_status(asset, root, records)
        if status in ("ok", "unverified"):
            continue
        dest = Path(root) / asset["dest"]
        part = _part(dest, asset)
        have = part.stat().st_size if part.exists() else 0
        need = asset["size"] - have
        if asset.get("extract"):
            need += sum(asset["extract"]["files"].values())
        parent = _existing_parent(dest.parent)
        device = str(parent.stat().st_dev)
        entry = needs.setdefault(device, {"path": str(parent), "need": 0,
                                          "free": shutil.disk_usage(parent).free, "assets": []})
        entry["need"] += max(need, 0)
        entry["assets"].append(asset["name"])
    return needs


def check_space(needs: Mapping[str, dict], margin: int = 1_000_000_000) -> List[str]:
    """Problems (one line per file system) where the free space is below the need plus ``margin``."""
    return [f"{entry['path']}: {human(entry['need'] + margin)} needed for {', '.join(entry['assets'])}, "
            f"{human(entry['free'])} free" for entry in needs.values() if entry["need"] + margin > entry["free"]]


# ----------------------------------------------------------------------------- fetch
def fetch_one(asset: Mapping, root: Path, records: Records, *, verify: bool = False, token: Optional[str] = None,
              opener=None, retries: int = 6, timeout: float = 60.0, sleep=_sleep) -> str:
    """Make one asset present and verified; returns ``present``, ``verified`` or ``downloaded``."""
    name, root = asset["name"], Path(root)
    dest = root / asset["dest"]
    status = asset_status(asset, root, records, full=verify)
    if status == "ok":
        for note in ensure_links(asset, root):
            log(f"{name}: {note}")
        return "present"
    if status == "unverified" and asset.get("extract"):
        log(f"{name}: checking SHA-256 of the files in {asset['dest']}")
        if reverify_extracted(asset, dest):
            for note in ensure_links(asset, root):
                log(f"{name}: {note}")
            return "verified"
        log(f"{name}: files in {asset['dest']} differ from the verified archive; downloading it again")
    elif status == "unverified":
        log(f"{name}: checking SHA-256 of {asset['dest']} ({human(asset['size'])})")
        if file_sha256(dest) == asset["sha256"]:
            records.add(asset["dest"], dest, asset["sha256"])
            for note in ensure_links(asset, root):
                log(f"{name}: {note}")
            return "verified"
        log(f"{name}: {asset['dest']} has the right size but other content; downloading it again")
    if status in ("unverified", "mismatch"):
        records.drop(asset["dest"])
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
    part = _part(dest, asset)
    if asset.get("auth") == "huggingface" and not token:
        raise AssetError(f"{name}: needs a Hugging Face token (hf auth login) with access to {asset.get('gated_page')}")
    resumed = part.stat().st_size if part.exists() else 0
    log(f"{name}: downloading {human(asset['size'])}" + (f", resuming at {human(resumed)}" if resumed else "")
        + f" -> {asset['dest']}")
    started = time.monotonic()
    url = download(asset["urls"], part, asset["size"], asset["sha256"], token=token, opener=opener,
                   retries=retries, timeout=timeout, sleep=sleep, progress=Progress(name), label=name)
    if asset.get("extract"):
        log(f"{name}: extracting into {asset['dest']}")
        finish_extract(asset, part, dest)
    else:
        os.replace(part, dest)
        records.add(asset["dest"], dest, asset["sha256"])
    for note in ensure_links(asset, root):
        log(f"{name}: {note}")
    log(f"{name}: done in {time.monotonic() - started:.0f} s (from {_short(url)}, SHA-256 verified)")
    return "downloaded"


def fetch(assets: Sequence[Mapping], root: Path, *, jobs: int = 4, verify: bool = False,
          environ: Mapping[str, str] = os.environ, opener=None, retries: int = 6, timeout: float = 60.0,
          sleep=_sleep, check_disk: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Fetch several assets concurrently. Returns ``(results, failures)`` keyed by asset name."""
    root = Path(root)
    records = Records(root)
    assets = [asset for asset in assets if asset["group"] != "manual"]
    if check_disk:
        problems = check_space(space_needed(assets, root, records))
        if problems:
            raise AssetError("not enough free disk space: " + "; ".join(problems))
    token = hf_token(environ) if any(asset.get("auth") == "huggingface" for asset in assets) else None
    results, failures, futures = {}, {}, {}
    CANCEL.clear()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs))
    try:
        futures = {pool.submit(fetch_one, asset, root, records, verify=verify, token=token, opener=opener,
                               retries=retries, timeout=timeout, sleep=sleep): asset for asset in assets}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]["name"]
            try:
                results[name] = future.result()
            except Cancelled:
                raise KeyboardInterrupt from None
            except Exception as exc:  # noqa: BLE001 - every failure is reported with its asset
                failures[name] = f"{type(exc).__name__}: {exc}"
                log(f"{name}: FAILED: {exc}")
    except KeyboardInterrupt:
        CANCEL.set()          # running downloads stop after their current block; .part files stay
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True)
        raise
    pool.shutdown(wait=True)
    return results, failures


# ----------------------------------------------------------------------------- SMPL-X
SMPLX_NAME = "SMPLX_NEUTRAL.npz"


def smplx_candidates(source: Optional[str], environ: Mapping[str, str], root: Path) -> List[Path]:
    """Where to look for the SMPL-X model, in order: ``source``, ``$HMB_SMPLX_MODEL``, the
    installer's destination and GVHMR's own body-model folder."""
    paths = []
    if source:
        paths.append(Path(source).expanduser())
    if environ.get("HMB_SMPLX_MODEL"):
        paths.append(Path(environ["HMB_SMPLX_MODEL"]).expanduser())
    paths += [root / "outputs/assets/smplx" / SMPLX_NAME,
              root / "GVHMR/inputs/checkpoints/body_models/smplx" / SMPLX_NAME]
    return paths


def _resolve_source(path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """``(file, zip member)`` for a file, a folder (``SMPLX_NEUTRAL.npz`` in it, in ``smplx/`` or in
    ``models/smplx/``) or the SMPL-X zip."""
    if path.is_dir():
        for rel in (SMPLX_NAME, f"smplx/{SMPLX_NAME}", f"models/smplx/{SMPLX_NAME}",
                    f"body_models/smplx/{SMPLX_NAME}"):
            if (path / rel).is_file():
                return path / rel, None
        for zipped in sorted(path.glob("*.zip")):
            member = _zip_member(zipped)
            if member:
                return zipped, member
        return None, None
    if path.is_file() and zipfile.is_zipfile(path) and path.suffix == ".zip":
        member = _zip_member(path)
        return (path, member) if member else (None, None)
    if path.is_file():
        return path, None
    return None, None


def _zip_member(path: Path) -> Optional[str]:
    """The ``SMPLX_NEUTRAL.npz`` member of a zip, preferring one in a ``smplx/`` folder."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == SMPLX_NAME]
    except (OSError, zipfile.BadZipFile):
        return None
    names.sort(key=lambda n: (0 if "/smplx/" in f"/{n}" else 1, len(n)))
    return names[0] if names else None


def validate_smplx(path: Path) -> Tuple[bool, str]:
    """A SMPL-X ``.npz`` (zip container with the expected arrays when NumPy is available)."""
    try:
        if not zipfile.is_zipfile(path):
            return False, "not an .npz file"
        with zipfile.ZipFile(path) as archive:
            names = {n[:-4] for n in archive.namelist() if n.endswith(".npy")}
    except OSError as exc:
        return False, f"unreadable ({exc})"
    missing = {"v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f", "kintree_table"} - names
    if missing:
        return False, f"not a SMPL-X model (missing arrays: {', '.join(sorted(missing))})"
    try:
        import numpy as np  # optional: the installer may run before the environment exists
    except ImportError:
        return True, "arrays present"
    with np.load(path, allow_pickle=False) as data:
        if data["v_template"].shape != (10475, 3) or data["f"].shape != (20908, 3):
            return False, f"unexpected mesh size {data['v_template'].shape} (SMPL-X has 10475 vertices)"
    return True, "SMPL-X mesh (10475 vertices)"


def smplx_place(asset: Mapping, root: Path, source: Optional[str] = None,
                environ: Mapping[str, str] = os.environ) -> dict:
    """Find SMPLX_NEUTRAL.npz and make it available at the manifest destination (a symbolic link to
    a file you keep elsewhere, or the file extracted from the SMPL-X zip). Never downloads."""
    root = Path(root)
    dest = root / asset["dest"]
    for candidate in smplx_candidates(source, environ, root):
        path, member = _resolve_source(candidate)
        if path is None:
            if source and candidate == Path(source).expanduser():
                return {"status": "invalid", "detail": f"{candidate}: no {SMPLX_NAME} found there"}
            continue
        if member:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            with zipfile.ZipFile(path) as archive, archive.open(member) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            ok, detail = validate_smplx(tmp)
            if not ok:
                tmp.unlink()
                return {"status": "invalid", "detail": f"{path}:{member}: {detail}"}
            if (dest.is_file() and not dest.is_symlink() and dest.stat().st_size == tmp.stat().st_size
                    and file_sha256(dest) == file_sha256(tmp)):
                # The same file from an earlier run: keep it, with its times. The result caches identify
                # large model files by size and times, so rewriting it would recompute every body mesh.
                tmp.unlink()
            else:
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                os.replace(tmp, dest)
            origin = f"{path} ({member})"
        else:
            ok, detail = validate_smplx(path)
            if not ok:
                if source and candidate == Path(source).expanduser():
                    return {"status": "invalid", "detail": f"{path}: {detail}"}
                continue
            if path.absolute() != dest.absolute() and not (dest.exists() and dest.resolve() == path.resolve()):
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                dest.symlink_to(path.resolve())
            origin = str(path)
        digest = file_sha256(dest)
        same = digest == asset["sha256"]
        note = ("the SMPL-X v1.1 file HumanScore expects" if same else
                f"SHA-256 {digest[:12]}... differs from the expected SMPL-X v1.1 file; scores can differ")
        return {"status": "ok", "path": str(dest), "source": origin, "validated_version": same, "detail": note}
    return {"status": "missing", "detail": smplx_instructions(asset, root)}


def smplx_instructions(asset: Mapping, root: Path) -> str:
    """What to do when the SMPL-X model is missing (printed by ``smplx`` and ``install.sh``)."""
    manual = asset["manual"]
    dest = Path(root) / asset["dest"]
    return "\n".join([
        "The SMPL-X body model is licensed and cannot be downloaded automatically.",
        f"  1. Register at {manual['website']} and accept the license.",
        f"  2. Download {manual['download']}.",
        f"  3. Give it to the installer, which takes {manual['member']} out of the zip:",
        "       bash install.sh --smplx-dir /path/to/models_smplx_v1_1.zip",
        "     or place the file yourself (a symbolic link is fine):",
        f"       mkdir -p {dest.parent} && cp SMPLX_NEUTRAL.npz {dest}",
        f"     or set {manual['env']}=/path/to/SMPLX_NEUTRAL.npz when running HumanScore.",
        "  Only SMPLX_NEUTRAL.npz is used (SMPL and the male/female models are not needed).",
    ])


# ----------------------------------------------------------------------------- Hugging Face access
def hf_access(asset: Mapping, environ: Mapping[str, str] = os.environ, opener=None,
              base: str = f"https://{HF_HOST}", timeout: float = 30.0) -> Tuple[str, str]:
    """Whether the token may download a gated file: ``(status, message)`` with status one of
    ``granted``, ``no-token``, ``invalid-token``, ``not-granted``, ``network``."""
    token = hf_token(environ)
    page = asset.get("gated_page", base)
    if not token:
        return "no-token", f"not logged in to Hugging Face (no token); run `hf auth login` after access to {page} was granted"
    opener = opener or urllib.request.build_opener(_NoRedirect())
    who = urllib.request.Request(f"{base}/api/whoami-v2", headers={"User-Agent": USER_AGENT})
    who.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with opener.open(who, timeout=timeout) as response:
            user = json.loads(response.read().decode() or "{}").get("name", "?")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return "invalid-token", "the Hugging Face token was rejected (expired or revoked); run `hf auth login` again"
        return "network", f"Hugging Face whoami failed (HTTP {exc.code})"
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return "network", f"cannot reach {base} ({_describe(exc)})"
    url = asset["urls"][0]
    if base != f"https://{HF_HOST}":
        url = base + urllib.parse.urlsplit(url).path
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    head.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with opener.open(head, timeout=timeout):
            return "granted", f"logged in as {user}; access to {page} granted"
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            return "granted", f"logged in as {user}; access to {page} granted"
        message = exc.headers.get("X-Error-Message", "") if exc.headers else ""
        if exc.code in (401, 403):
            pending = "awaiting" in message.lower() or "pending" in message.lower()
            state = "your access request is pending review" if pending else "access not granted yet"
            return "not-granted", (f"logged in as {user}, but {state}: request access at {page} (approval by the "
                                   f"authors can take a while), then re-run the installer. Server: {message or exc.code}")
        return "network", f"HEAD {url} failed (HTTP {exc.code})"
    except (OSError, urllib.error.URLError) as exc:
        return "network", f"cannot reach {base} ({_describe(exc)})"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: for the access check, a redirect to the file store means access."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


# ----------------------------------------------------------------------------- remove / plan
def remove(assets: Sequence[Mapping], root: Path) -> List[str]:
    """Delete downloaded files (never manual ones), their partial downloads and the links to them."""
    root = Path(root)
    records = Records(root)
    removed = []
    for asset in assets:
        if asset["group"] == "manual":
            continue
        dest = root / asset["dest"]
        for rel in asset.get("links", []):
            link = root / rel
            if link.is_symlink() and link.resolve() == dest.resolve():
                link.unlink()
                removed.append(rel)
        for path in (dest, _part(dest, asset), dest.with_name(dest.name + ".extracting")):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                removed.append(str(path.relative_to(root)))
            elif path.exists() or path.is_symlink():
                path.unlink()
                removed.append(str(path.relative_to(root)))
        records.drop(asset["dest"])
    return removed


def plan(assets: Sequence[Mapping], root: Path) -> dict:
    """Status, size and bytes still to download of each asset, and the disk space they need."""
    root = Path(root)
    records = Records(root)
    rows = []
    for asset in assets:
        if asset["group"] == "manual":
            path, _ = next(((p, m) for p, m in (_resolve_source(c) for c in
                                                smplx_candidates(None, os.environ, root)) if p), (None, None))
            rows.append({"name": asset["name"], "group": "manual", "status": "ok" if path else "missing",
                         "size": asset["size"], "download": 0, "dest": asset["dest"]})
            continue
        status = asset_status(asset, root, records)
        dest = root / asset["dest"]
        part = _part(dest, asset)
        have = part.stat().st_size if part.exists() else 0
        download_bytes = 0 if status in ("ok", "unverified") else asset["size"] - have
        final = sum(asset["extract"]["files"].values()) if asset.get("extract") else asset["size"]
        rows.append({"name": asset["name"], "group": asset["group"], "status": status, "size": final,
                     "download": download_bytes, "dest": asset["dest"], "required": asset.get("required", False)})
    needs = space_needed(assets, root, records)
    return {"assets": rows, "download_bytes": sum(r["download"] for r in rows),
            "disk": list(needs.values()), "disk_problems": check_space(needs)}


def download_hosts(assets: Sequence[Mapping]) -> List[Tuple[str, str]]:
    """``(host, "primary" | "mirror")``: hosts of the first URL of each asset, then the others."""
    primary, mirrors = [], []
    for asset in assets:
        for index, url in enumerate(asset.get("urls", [])):
            host = urllib.parse.urlsplit(url).netloc
            (primary if index == 0 else mirrors).append(host)
    seen, result = set(), []
    for host, kind in [(h, "primary") for h in primary] + [(h, "mirror") for h in mirrors]:
        if host not in seen:
            seen.add(host)
            result.append((host, kind))
    return result


def format_plan(report: Mapping) -> str:
    """The :func:`plan` report as a table (``plan`` without ``--json``)."""
    lines = [f"  {'asset':14s} {'group':9s} {'status':10s} {'size':>9s} {'to download':>12s}  destination"]
    for row in report["assets"]:
        lines.append(f"  {row['name']:14s} {row['group']:9s} {row['status']:10s} {human(row['size']):>9s} "
                     f"{human(row['download']) if row['download'] else '-':>12s}  {row['dest']}")
    lines.append(f"  total to download: {human(report['download_bytes'])}")
    for problem in report["disk_problems"]:
        lines.append(f"  NOT ENOUGH DISK: {problem}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the exit status of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root (default: this checkout)")
    parser.add_argument("--manifest", default=None, help="manifest (default: ROOT/install/assets.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fetch", "check", "remove", "hosts"):
        command = sub.add_parser(name)
        command.add_argument("--groups", default="core", help="comma-separated: core, sam3, selftest, manual")
        command.add_argument("--only", nargs="*", default=[], help="asset names (overrides --groups)")
        if name == "plan":
            command.add_argument("--json", action="store_true")
            command.add_argument("--disk-tsv", action="store_true",
                                 help="only the disk needs: PATH<TAB>BYTES_NEEDED<TAB>BYTES_FREE per file system")
        if name == "check":
            command.add_argument("--verified", action="store_true",
                                 help="also require each file to be SHA-256 verified and unchanged since")
        if name == "fetch":
            command.add_argument("--jobs", type=int, default=4, help="parallel downloads (default 4)")
            command.add_argument("--verify", action="store_true", help="re-hash files verified before")
            command.add_argument("--retries", type=int, default=6)
            command.add_argument("--timeout", type=float, default=60.0, help="seconds without data before a retry")
    smplx = sub.add_parser("smplx", help="find SMPLX_NEUTRAL.npz and place it where HumanScore reads it")
    smplx.add_argument("--source", default=None, help="SMPLX_NEUTRAL.npz, a folder containing it, or the SMPL-X zip")
    smplx.add_argument("--json", action="store_true")
    access = sub.add_parser("hf-access", help="check Hugging Face login and access to a gated asset")
    access.add_argument("--name", default="sam3.1")
    access.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    previous = None
    if threading.current_thread() is threading.main_thread():
        # SIGTERM (install.sh stopping a background download) ends like Ctrl-C: cleanly, .part kept.
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("[assets] interrupted; partial downloads (*.part) are kept: run again to resume", file=sys.stderr,
              flush=True)
        return 130
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _raise_interrupt(signum, frame):
    """SIGTERM handler: stop like Ctrl-C."""
    raise KeyboardInterrupt


def _run(args) -> int:
    """Run one command of the parsed arguments."""
    root = Path(args.root).resolve()
    try:
        manifest = load_manifest(Path(args.manifest) if args.manifest else root / "install/assets.json")
    except (OSError, ValueError) as exc:
        print(f"[assets] ERROR: cannot read the manifest: {exc}", file=sys.stderr)
        return 1
    by_name = {asset["name"]: asset for asset in manifest["assets"]}

    if args.command == "smplx":
        result = smplx_place(by_name["smplx"], root, args.source)
        if args.json:
            print(json.dumps(result))
        elif result["status"] == "ok":
            log(f"SMPL-X: {result['path']} <- {result['source']} ({result['detail']})")
        else:
            print(result["detail"], flush=True)
        return 0 if result["status"] == "ok" else (EXIT_MISSING_MANUAL if result["status"] == "missing" else 1)

    if args.command == "hf-access":
        status, message = hf_access(by_name[args.name])
        print(json.dumps({"status": status, "message": message}) if args.json else f"[assets] {message}", flush=True)
        return HF_EXIT[status]

    try:
        assets = select(manifest, [g for g in args.groups.split(",") if g], args.only)
    except ManifestError as exc:
        print(f"[assets] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.command == "plan":
        report = plan(assets, root)
        if args.disk_tsv:
            for entry in report["disk"]:
                print(f"{entry['path']}\t{entry['need']}\t{entry['free']}")
        else:
            print(json.dumps(report, indent=1) if args.json else format_plan(report))
        return 0
    if args.command == "hosts":
        for host, kind in download_hosts(assets):
            print(f"{host}\t{kind}")
        return 0
    if args.command == "check":
        records = Records(root)
        # Default: complete files (sizes). --verified: also verified by SHA-256 and unchanged since.
        accepted = ("ok",) if args.verified else ("ok", "unverified")
        bad = [f"{a['name']} ({asset_status(a, root, records)})" for a in assets
               if a["group"] != "manual" and asset_status(a, root, records) not in accepted]
        bad += [f"{a['name']} (link missing: {', '.join(a['links'])})" for a in assets
                if a["group"] != "manual" and a.get("links") and asset_status(a, root, records) in accepted
                and not links_ok(a, root)]
        if bad:
            print(f"[assets] incomplete: {', '.join(bad)}", flush=True)
            return 1
        return 0
    if args.command == "remove":
        for path in remove(assets, root):
            log(f"removed {path}")
        return 0
    try:
        results, failures = fetch(assets, root, jobs=args.jobs, verify=args.verify, retries=args.retries,
                                  timeout=args.timeout)
    except AssetError as exc:
        print(f"[assets] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    downloaded = [name for name, result in results.items() if result == "downloaded"]
    log(f"{len(results)} of {len(results) + len(failures)} present and verified"
        + (f" ({len(downloaded)} downloaded)" if downloaded else ""))
    if failures:
        for name, error in sorted(failures.items()):
            print(f"[assets] ERROR: {name}: {error}", file=sys.stderr, flush=True)
        print("[assets] Re-run to resume: partial downloads (*.part) are kept.", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
