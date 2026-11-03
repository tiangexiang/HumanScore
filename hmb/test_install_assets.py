"""install/fetch_assets.py (model-file manifest and downloader) and install/self_test.py.

A local HTTP server with range support and failure injection stands in for the model hosts.
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import io
import json
import os
import re
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fa = _load("hmb_fetch_assets", REPO / "install/fetch_assets.py")
self_test = _load("hmb_install_self_test", REPO / "install/self_test.py")


# ----------------------------------------------------------------------------- local HTTP server
class Server:
    """Serves ``files`` (path -> bytes). Per path: ``fail[path]`` = number of initial requests that
    get HTTP 500, ``truncate[path]`` = byte count after which the first response is cut,
    ``ignore_range`` = paths that always answer 200, ``status[path]`` = fixed status code,
    ``html`` = paths served as an HTML page, ``redirect[path]`` = Location for a 302."""

    def __init__(self):
        self.files, self.fail, self.truncate, self.status, self.redirect = {}, {}, {}, {}, {}
        self.ignore_range, self.html = set(), set()
        self.requests, self.served_bytes, self.headers = [], {}, []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                pass

            def do_HEAD(self):
                self._respond(head=True)

            def do_GET(self):
                self._respond(head=False)

            def _respond(self, head):
                path = self.path.split("?")[0]
                outer.requests.append((self.command, path, self.headers.get("Range")))
                outer.headers.append(dict(self.headers))
                if path in outer.redirect:
                    self.send_response(302)
                    self.send_header("Location", outer.redirect[path])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path in outer.status:
                    code, extra = outer.status[path]
                    self.send_response(code)
                    for key, value in extra.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if outer.fail.get(path, 0) > 0:
                    outer.fail[path] -= 1
                    self.send_response(500)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = outer.files.get(path)
                if data is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                match = re.match(r"bytes=(\d+)-", self.headers.get("Range") or "")
                if match and path not in outer.ignore_range:
                    start = int(match.group(1))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Type", "text/html" if path in outer.html else "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if head:
                    return
                cut = outer.truncate.pop(path, None)
                if cut is not None:
                    body = body[:cut]
                self.wfile.write(body)
                outer.served_bytes[path] = outer.served_bytes.get(path, 0) + len(body)
                if cut is not None:
                    self.wfile.flush()
                    self.close_connection = True

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = Server()
    yield srv
    srv.close()


def payload(size: int, seed: int = 0) -> bytes:
    return bytes((i * 31 + seed) % 251 for i in range(size))


def asset_for(name, data, urls, dest=None, **extra):
    entry = {"name": name, "description": name, "group": "core", "required": True, "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest(), "urls": urls, "source": "test", "license": "test",
             "dest": dest or f"outputs/assets/{name}.bin"}
    entry.update(extra)
    return entry


NO_SLEEP = []


def no_sleep(seconds):
    NO_SLEEP.append(seconds)


# ----------------------------------------------------------------------------- manifest
def test_repository_manifest_is_valid_and_matches_the_code():
    manifest = fa.load_manifest(REPO / "install/assets.json")
    by_name = {a["name"]: a for a in manifest["assets"]}
    from hmb import doctor, model_paths
    from hmb.gvhmr_prepare import CHECKPOINTS
    from hmb.hadm_prepare import DEFAULT_CHECKPOINT

    # every GVHMR checkpoint, at the size prepare_assets expects, where the code links it
    for relative, (size, _) in CHECKPOINTS.items():
        entry = next(a for a in manifest["assets"] if a["dest"] == f"{doctor.GVHMR_ASSET_DIR}/{relative}")
        assert entry["size"] == size and entry["links"] == [f"GVHMR/inputs/checkpoints/{relative}"]
    assert by_name["hadm-g"]["links"] == [f"metrics/multi-limbs/HADM/{DEFAULT_CHECKPOINT}"]
    assert REPO / by_name["metrabs"]["dest"] == model_paths.METRABS_LOCAL
    assert REPO / by_name["sam3.1"]["dest"] == model_paths.SAM3_LOCAL
    assert by_name["yolov8x-seg"]["dest"] == f"{doctor.YOLO_SEG_DIR}/{doctor.YOLO_SEG_DEFAULT}"
    assert by_name["smplx"]["dest"] == "outputs/assets/smplx/SMPLX_NEUTRAL.npz"
    assert by_name["sam3.1"]["auth"] == "huggingface"
    # pinned revisions, not moving branches
    for asset in manifest["assets"]:
        for url in asset["urls"]:
            assert "/resolve/main/" not in url and "/latest/" not in url, url
    # the SAM 3.1 worker derives the same location without importing hmb
    worker = (REPO / "hmb/tracking/sam3_worker.py").read_text()
    assert '"outputs", "assets", "sam3"' in worker
    # the MeTRAbs loaders know the installer's location
    for loader in ("baseline/monocular_demos/utils.py", "metrics/bone_length/monocular_demos/utils.py"):
        assert by_name["metrabs"]["dest"] in (REPO / loader).read_text()


@pytest.mark.parametrize("mutate, message", [
    (lambda a: a.update(sha256="abc"), "sha256"),
    (lambda a: a.update(size=0), "size"),
    (lambda a: a.update(dest="../outside.bin"), "inside the repository"),
    (lambda a: a.update(dest="/abs/path.bin"), "inside the repository"),
    (lambda a: a.update(urls=["http://insecure.example/x"]), "https"),
    (lambda a: a.update(group="other"), "group"),
    (lambda a: a.pop("license"), "license"),
    (lambda a: a.update(extract={"format": "zip", "files": {}}), "extract"),
])
def test_manifest_validation_rejects(mutate, message):
    asset = asset_for("x", b"data", ["https://example.org/x"])
    mutate(asset)
    with pytest.raises(fa.ManifestError, match=message):
        fa.validate_manifest({"version": 1, "assets": [asset]})


def test_manifest_validation_rejects_duplicates():
    a = asset_for("x", b"data", ["https://example.org/x"])
    with pytest.raises(fa.ManifestError, match="duplicate"):
        fa.validate_manifest({"version": 1, "assets": [a, dict(a)]})
    b = dict(a, name="y")
    with pytest.raises(fa.ManifestError, match="duplicate destination"):
        fa.validate_manifest({"version": 1, "assets": [a, b]})


def test_select_groups_and_names():
    manifest = fa.load_manifest(REPO / "install/assets.json")
    core = {a["name"] for a in fa.select(manifest, ["core"])}
    assert {"gvhmr", "hmr2", "vitpose", "yolov8x", "hadm-g", "metrabs"} <= core and "sam3.1" not in core
    assert [a["name"] for a in fa.select(manifest, ["core"], ["sam3.1"])] == ["sam3.1"]
    with pytest.raises(fa.ManifestError, match="unknown asset"):
        fa.select(manifest, ["core"], ["nope"])


# ----------------------------------------------------------------------------- downloads
def test_download_verifies_and_links(server, tmp_path):
    data = payload(300_000)
    server.files["/a.bin"] = data
    asset = asset_for("a", data, [server.url + "/a.bin"], links=["GVHMR/inputs/a.bin"])
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"a": "downloaded"} and not failures
    dest = tmp_path / asset["dest"]
    assert dest.read_bytes() == data and not dest.with_name("a.bin.part").exists()
    link = tmp_path / "GVHMR/inputs/a.bin"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link)) and link.read_bytes() == data
    # a second run neither downloads nor hashes again
    server.requests.clear()
    results, _ = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"a": "present"} and server.requests == []
    records = json.loads((tmp_path / fa.RECORDS).read_text())
    assert records[asset["dest"]]["sha256"] == asset["sha256"]


def test_resume_requests_only_the_missing_bytes(server, tmp_path):
    data = payload(1_000_000)
    server.files["/big.bin"] = data
    asset = asset_for("big", data, [server.url + "/big.bin"])
    dest = tmp_path / asset["dest"]
    dest.parent.mkdir(parents=True)
    dest.with_name("big.bin.part").write_bytes(data[:600_000])      # an interrupted earlier run
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"big": "downloaded"} and not failures
    assert server.requests[-1][2] == "bytes=600000-"
    assert server.served_bytes["/big.bin"] == 400_000
    assert dest.read_bytes() == data


def test_connection_drop_is_resumed_on_retry(server, tmp_path):
    data = payload(800_000)
    server.files["/drop.bin"] = data
    server.truncate["/drop.bin"] = 250_000          # the first response stops after 250 kB
    asset = asset_for("drop", data, [server.url + "/drop.bin"])
    sleeps = []
    results, failures = fa.fetch([asset], tmp_path, sleep=sleeps.append, check_disk=False)
    assert results == {"drop": "downloaded"} and not failures
    ranges = [r[2] for r in server.requests if r[1] == "/drop.bin"]
    assert ranges[0] is None and ranges[1] == "bytes=250000-"
    assert server.served_bytes["/drop.bin"] == len(data)
    assert len(sleeps) == 1


def test_interrupt_keeps_the_partial_file_and_resumes(server, tmp_path, monkeypatch):
    data = payload(3 * fa.CHUNK)
    server.files["/int.bin"] = data
    asset = asset_for("int", data, [server.url + "/int.bin"])

    class StopAfterFirstBlock:
        def __init__(self, label):
            pass

        def __call__(self, done, total, url):
            fa.CANCEL.set()                    # what Ctrl-C / SIGTERM does in the main thread

    monkeypatch.setattr(fa, "Progress", StopAfterFirstBlock)
    with pytest.raises(KeyboardInterrupt):
        fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    part = tmp_path / "outputs/assets/int.bin.part"
    assert part.stat().st_size == fa.CHUNK and not (tmp_path / asset["dest"]).exists()
    monkeypatch.setattr(fa, "Progress", lambda label: (lambda done, total, url: None))
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"int": "downloaded"} and server.requests[-1][2] == f"bytes={fa.CHUNK}-"
    assert (tmp_path / asset["dest"]).read_bytes() == data


def test_cli_reports_an_interrupt(tmp_path, monkeypatch, capsys):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(fa, "fetch", interrupted)
    before = fa.signal.getsignal(fa.signal.SIGTERM)
    assert fa.main(["fetch", "--groups", "selftest"]) == 130
    assert "partial downloads" in capsys.readouterr().err
    assert fa.signal.getsignal(fa.signal.SIGTERM) == before      # the handler is restored


def test_damaged_partial_file_is_downloaded_again_in_full(server, tmp_path):
    data = payload(200_000)
    server.files["/dmg.bin"] = data
    asset = asset_for("dmg", data, [server.url + "/dmg.bin"])
    dest = tmp_path / asset["dest"]
    dest.parent.mkdir(parents=True)
    dest.with_name("dmg.bin.part").write_bytes(b"\0" * 70_000)       # garbage from an earlier crash
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"dmg": "downloaded"} and not failures and dest.read_bytes() == data
    assert [r[2] for r in server.requests] == ["bytes=70000-", None]


def test_server_ignoring_range_restarts_from_zero(server, tmp_path):
    data = payload(200_000)
    server.files["/norange.bin"] = data
    server.ignore_range.add("/norange.bin")
    asset = asset_for("norange", data, [server.url + "/norange.bin"])
    dest = tmp_path / asset["dest"]
    dest.parent.mkdir(parents=True)
    dest.with_name("norange.bin.part").write_bytes(data[:50_000])
    results, _ = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"norange": "downloaded"} and dest.read_bytes() == data


def test_retries_with_exponential_backoff(server, tmp_path, monkeypatch):
    data = payload(10_000)
    server.files["/flaky.bin"] = data
    server.fail["/flaky.bin"] = 3                    # three HTTP 500 first
    monkeypatch.setattr(fa.random, "random", lambda: 0.5)   # no jitter
    sleeps = []
    path = tmp_path / "flaky.part"
    url = fa.download([server.url + "/flaky.bin"], path, len(data), hashlib.sha256(data).hexdigest(),
                      sleep=sleeps.append, backoff=2.0, retries=5)
    assert url.endswith("/flaky.bin") and path.read_bytes() == data
    assert sleeps == [2.0, 4.0, 8.0]


def test_gives_up_after_the_retries(server, tmp_path):
    server.files["/down.bin"] = payload(100)
    server.fail["/down.bin"] = 100
    sleeps = []
    with pytest.raises(fa.AssetError, match="after 3 attempts"):
        fa.download([server.url + "/down.bin"], tmp_path / "down.part", 100, "0" * 64, sleep=sleeps.append,
                    retries=2)
    assert len(sleeps) == 2


def test_checksum_mismatch_deletes_the_file_and_fails(server, tmp_path):
    data = payload(50_000)
    server.files["/wrong.bin"] = payload(50_000, seed=7)   # same size, other bytes
    asset = asset_for("wrong", data, [server.url + "/wrong.bin"])
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert not results and "ChecksumError" in failures["wrong"] and "SHA-256" in failures["wrong"]
    dest = tmp_path / asset["dest"]
    assert not dest.exists() and not dest.with_name("wrong.bin.part").exists()
    assert len([r for r in server.requests if r[1] == "/wrong.bin"]) == 1   # not retried


def test_checksum_mismatch_falls_back_to_the_mirror(server, tmp_path):
    data = payload(50_000)
    server.files["/bad.bin"] = payload(50_000, seed=3)
    server.files["/good.bin"] = data
    asset = asset_for("m", data, [server.url + "/bad.bin", server.url + "/good.bin"])
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"m": "downloaded"} and not failures
    assert (tmp_path / asset["dest"]).read_bytes() == data


def test_missing_url_and_html_pages_fall_back_to_the_mirror(server, tmp_path):
    data = payload(20_000)
    server.files["/quota.html"] = b"<html>Quota exceeded</html>"
    server.html.add("/quota.html")
    server.files["/ok.bin"] = data
    asset = asset_for("h", data, [server.url + "/gone.bin", server.url + "/quota.html", server.url + "/ok.bin"])
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"h": "downloaded"} and not failures


def test_wrong_size_on_disk_is_downloaded_again(server, tmp_path):
    data = payload(30_000)
    server.files["/s.bin"] = data
    asset = asset_for("s", data, [server.url + "/s.bin"])
    dest = tmp_path / asset["dest"]
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"truncated")
    assert fa.asset_status(asset, tmp_path) == "mismatch"
    results, _ = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"s": "downloaded"} and dest.read_bytes() == data


def test_existing_copy_is_verified_once_then_trusted(server, tmp_path):
    data = payload(30_000)
    asset = asset_for("c", data, [server.url + "/c.bin"])
    dest = tmp_path / asset["dest"]
    dest.parent.mkdir(parents=True)
    dest.write_bytes(data)                          # placed by the user
    assert fa.asset_status(asset, tmp_path, fa.Records(tmp_path)) == "unverified"
    results, _ = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"c": "verified"} and server.requests == []
    assert fa.asset_status(asset, tmp_path, fa.Records(tmp_path)) == "ok"
    # same size, other content (e.g. replaced by hand): the record no longer matches, so it is re-hashed
    dest.write_bytes(payload(30_000, seed=9))
    os.utime(dest, ns=(1, 1))
    server.files["/c.bin"] = data
    results, _ = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"c": "downloaded"} and dest.read_bytes() == data


def test_not_enough_disk_space_stops_before_downloading(server, tmp_path, monkeypatch):
    data = payload(1000)
    asset = asset_for("d", data, [server.url + "/d.bin"])
    real = fa.shutil.disk_usage
    monkeypatch.setattr(fa.shutil, "disk_usage", lambda p: real(p)._replace(free=10))
    with pytest.raises(fa.AssetError, match="not enough free disk space"):
        fa.fetch([asset], tmp_path, sleep=no_sleep)
    assert server.requests == []


def test_parallel_fetch_reports_each_failure(server, tmp_path):
    good, bad = payload(5000), payload(5000, seed=1)
    server.files["/g.bin"] = good
    assets = [asset_for("g", good, [server.url + "/g.bin"]), asset_for("x", bad, [server.url + "/x.bin"])]
    results, failures = fa.fetch(assets, tmp_path, jobs=2, sleep=no_sleep, retries=1, check_disk=False)
    assert results == {"g": "downloaded"} and set(failures) == {"x"} and "HTTP 404" in failures["x"]


def test_cli_fetch_and_check_exit_status(server, tmp_path):
    data = payload(4000)
    server.files["/cli.bin"] = data
    manifest = {"version": 1, "assets": [asset_for("cli", data, ["https://placeholder/cli.bin"])]}
    manifest["assets"][0]["urls"] = [server.url + "/cli.bin"]
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(manifest))
    # load_manifest insists on https: the CLI is exercised with a validated manifest swapped in
    original = fa.validate_manifest
    try:
        fa.validate_manifest = lambda m, require_https=True: original(m, require_https=False)
        args = ["--root", str(tmp_path), "--manifest", str(path)]
        assert fa.main(args + ["check"]) == 1
        assert fa.main(args + ["fetch", "--retries", "0"]) == 0
        assert fa.main(args + ["check"]) == 0
        assert fa.main(args + ["remove"]) == 0
        assert not (tmp_path / "outputs/assets/cli.bin").exists()
    finally:
        fa.validate_manifest = original


# ----------------------------------------------------------------------------- archives
def make_tar(entries) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_archive_is_extracted_verified_and_removed(server, tmp_path):
    files = {"saved_model.pb": b"graph" * 100, "variables/variables.index": b"index"}
    archive = make_tar([("/", None), ("/saved_model.pb", files["saved_model.pb"]), ("/variables", None),
                        ("/variables/variables.index", files["variables/variables.index"]), ("/assets", None)])
    server.files["/model.tar.gz"] = archive
    asset = asset_for("model", archive, [server.url + "/model.tar.gz"], dest="outputs/assets/model/m1",
                      extract={"format": "tar.gz", "files": {k: len(v) for k, v in files.items()}})
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert results == {"model": "downloaded"} and not failures
    dest = tmp_path / "outputs/assets/model/m1"
    assert (dest / "saved_model.pb").read_bytes() == files["saved_model.pb"] and (dest / "assets").is_dir()
    assert not dest.with_name("m1.tar.gz.part").exists() and not dest.with_name("m1.extracting").exists()
    assert fa.asset_status(asset, tmp_path) == "ok"
    server.requests.clear()
    assert fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)[0] == {"model": "present"}
    assert server.requests == []


def test_archive_with_parent_path_is_rejected(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(make_tar([("../escape.txt", b"x")]))
    with pytest.raises(fa.AssetError, match="unsafe path"):
        fa.safe_extract_tar(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_archive_with_other_content_is_rejected(server, tmp_path):
    archive = make_tar([("saved_model.pb", b"short")])
    server.files["/m.tar.gz"] = archive
    asset = asset_for("m", archive, [server.url + "/m.tar.gz"], dest="outputs/assets/m",
                      extract={"format": "tar.gz", "files": {"saved_model.pb": 999}})
    results, failures = fa.fetch([asset], tmp_path, sleep=no_sleep, check_disk=False)
    assert not results and "archive content differs" in failures["m"]
    assert not (tmp_path / "outputs/assets/m").exists()


# ----------------------------------------------------------------------------- links and removal
def test_links_respect_user_files_and_replace_stale_links(tmp_path):
    data = payload(100)
    asset = asset_for("l", data, ["https://example.org/l"], links=["a/link.bin", "b/link.bin", "c/link.bin"])
    dest = tmp_path / asset["dest"]
    dest.parent.mkdir(parents=True)
    dest.write_bytes(data)
    (tmp_path / "a").mkdir()
    (tmp_path / "a/link.bin").write_bytes(data)                 # a user's copy: kept
    (tmp_path / "b").mkdir()
    (tmp_path / "b/link.bin").symlink_to(tmp_path / "elsewhere")  # stale link: replaced
    notes = fa.ensure_links(asset, tmp_path)
    assert not (tmp_path / "a/link.bin").is_symlink() and any("regular file" in n for n in notes)
    assert (tmp_path / "b/link.bin").resolve() == dest.resolve()
    assert (tmp_path / "c/link.bin").resolve() == dest.resolve()
    removed = fa.remove([asset], tmp_path)
    assert not dest.exists() and not (tmp_path / "b/link.bin").is_symlink()
    assert (tmp_path / "a/link.bin").read_bytes() == data      # never deletes a user's file
    assert asset["dest"] in removed


def test_remove_never_touches_manual_assets(tmp_path):
    manifest = fa.load_manifest(REPO / "install/assets.json")
    smplx = next(a for a in manifest["assets"] if a["name"] == "smplx")
    target = tmp_path / smplx["dest"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"licensed")
    fa.remove([smplx], tmp_path)
    assert target.read_bytes() == b"licensed"


# ----------------------------------------------------------------------------- SMPL-X
def fake_smplx(path: Path, vertices: int = 10475) -> bytes:
    np = pytest.importorskip("numpy")
    arrays = {"v_template": np.zeros((vertices, 3)), "shapedirs": np.zeros((vertices, 3, 2)),
              "posedirs": np.zeros((vertices, 3, 2)), "J_regressor": np.zeros((55, vertices)),
              "weights": np.zeros((vertices, 55)), "f": np.zeros((20908, 3), dtype=np.uint32),
              "kintree_table": np.zeros((2, 55), dtype=np.int64)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path.read_bytes()


@pytest.fixture
def smplx_asset():
    manifest = fa.load_manifest(REPO / "install/assets.json")
    return next(a for a in manifest["assets"] if a["name"] == "smplx")


def test_smplx_missing_prints_where_to_get_it(tmp_path, smplx_asset):
    result = fa.smplx_place(smplx_asset, tmp_path, None, environ={})
    assert result["status"] == "missing"
    for text in ("https://smpl-x.is.tue.mpg.de", "models_smplx_v1_1.zip", "--smplx-dir",
                 str(tmp_path / "outputs/assets/smplx/SMPLX_NEUTRAL.npz"), "HMB_SMPLX_MODEL"):
        assert text in result["detail"]


@pytest.mark.parametrize("layout", ["file", "folder", "smplx-subfolder", "models-subfolder"])
def test_smplx_from_a_folder_or_file_is_linked(tmp_path, smplx_asset, layout):
    user = tmp_path / "user"
    rel = {"file": "SMPLX_NEUTRAL.npz", "folder": "SMPLX_NEUTRAL.npz", "smplx-subfolder": "smplx/SMPLX_NEUTRAL.npz",
           "models-subfolder": "models/smplx/SMPLX_NEUTRAL.npz"}[layout]
    fake_smplx(user / rel)
    source = user / rel if layout == "file" else user
    repo = tmp_path / "repo"
    result = fa.smplx_place(smplx_asset, repo, str(source), environ={})
    dest = repo / "outputs/assets/smplx/SMPLX_NEUTRAL.npz"
    assert result["status"] == "ok" and dest.is_symlink() and dest.resolve() == (user / rel).resolve()
    assert result["validated_version"] is False and "differs" in result["detail"]


def test_smplx_from_the_zip_is_extracted(tmp_path, smplx_asset):
    model = tmp_path / "SMPLX_NEUTRAL.npz"
    data = fake_smplx(model)
    archive = tmp_path / "models_smplx_v1_1.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("models/smplx/SMPLX_MALE.npz", b"other")
        handle.writestr("models/smplx/SMPLX_NEUTRAL.npz", data)
    repo = tmp_path / "repo"
    result = fa.smplx_place(smplx_asset, repo, str(archive), environ={})
    dest = repo / "outputs/assets/smplx/SMPLX_NEUTRAL.npz"
    assert result["status"] == "ok" and not dest.is_symlink() and dest.read_bytes() == data
    # Placing it again from the zip (every `bash install.sh --smplx-dir ZIP`) leaves the identical
    # file untouched: hmb/result_cache.py identifies large model files by size, mtime and ctime.
    before = dest.stat()
    assert fa.smplx_place(smplx_asset, repo, str(archive), environ={})["status"] == "ok"
    after = dest.stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_ctime_ns) == (before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
    assert not dest.with_name(dest.name + ".part").exists()


def test_smplx_already_placed_or_from_environment(tmp_path, smplx_asset):
    repo = tmp_path / "repo"
    fake_smplx(repo / "outputs/assets/smplx/SMPLX_NEUTRAL.npz")
    assert fa.smplx_place(smplx_asset, repo, None, environ={})["status"] == "ok"
    other = tmp_path / "other"
    fake_smplx(other / "SMPLX_NEUTRAL.npz")
    repo2 = tmp_path / "repo2"
    result = fa.smplx_place(smplx_asset, repo2, None, environ={"HMB_SMPLX_MODEL": str(other / "SMPLX_NEUTRAL.npz")})
    assert result["status"] == "ok"


def test_smplx_invalid_source_is_rejected(tmp_path, smplx_asset):
    bad = tmp_path / "SMPLX_NEUTRAL.npz"
    bad.write_bytes(b"not a model")
    assert fa.smplx_place(smplx_asset, tmp_path / "repo", str(bad), environ={})["status"] == "invalid"
    wrong = tmp_path / "smpl" / "SMPLX_NEUTRAL.npz"
    fake_smplx(wrong, vertices=6890)                      # an SMPL-sized mesh
    result = fa.smplx_place(smplx_asset, tmp_path / "repo", str(wrong), environ={})
    assert result["status"] == "invalid" and "10475" in result["detail"]
    assert fa.smplx_place(smplx_asset, tmp_path / "repo", str(tmp_path / "nowhere"), environ={})["status"] == "invalid"


# ----------------------------------------------------------------------------- Hugging Face access
def hf_asset(server):
    return {"name": "gated", "urls": ["https://huggingface.co/org/model/resolve/abc/file.pt"],
            "gated_page": "https://huggingface.co/org/model"}


def test_hf_token_lookup_order(tmp_path):
    (tmp_path / "home/.cache/huggingface").mkdir(parents=True)
    (tmp_path / "home/.cache/huggingface/token").write_text("hf_home_default\n")
    (tmp_path / "hfhome").mkdir()
    (tmp_path / "hfhome/token").write_text("hf_from_hf_home")
    (tmp_path / "path_token").write_text("hf_from_path")
    home = {"HOME": str(tmp_path / "home")}
    assert fa.hf_token(home) == "hf_home_default"
    assert fa.hf_token({**home, "HF_HOME": str(tmp_path / "hfhome")}) == "hf_from_hf_home"
    assert fa.hf_token({**home, "HF_TOKEN_PATH": str(tmp_path / "path_token")}) == "hf_from_path"
    assert fa.hf_token({**home, "HF_TOKEN": "hf_env"}) == "hf_env"
    assert fa.hf_token({"HOME": str(tmp_path / "nobody")}) is None


def test_hf_access_states(server, tmp_path):
    asset = hf_asset(server)
    environ = {"HOME": str(tmp_path), "HF_TOKEN": "hf_test"}
    assert fa.hf_access(asset, {"HOME": str(tmp_path)}, base=server.url)[0] == "no-token"
    server.status["/api/whoami-v2"] = (401, {})
    assert fa.hf_access(asset, environ, base=server.url)[0] == "invalid-token"
    del server.status["/api/whoami-v2"]
    server.files["/api/whoami-v2"] = b'{"name": "tester"}'
    server.status["/org/model/resolve/abc/file.pt"] = (403, {"X-Error-Message": "Your request to access model "
                                                             "org/model is awaiting a review from the repo authors."})
    status, message = fa.hf_access(asset, environ, base=server.url)
    assert status == "not-granted" and "pending" in message and "tester" in message
    server.status["/org/model/resolve/abc/file.pt"] = (302, {"Location": "https://cdn.example/x"})
    status, message = fa.hf_access(asset, environ, base=server.url)
    assert status == "granted"
    # the token went to the Hugging Face host only
    sent = [h.get("Authorization") for h in server.headers]
    assert "Bearer hf_test" in sent


def test_authorization_is_not_forwarded_on_redirect(server, tmp_path, monkeypatch):
    data = payload(1000)
    server.files["/cdn/file.bin"] = data
    server.redirect["/resolve/file.bin"] = server.url + "/cdn/file.bin"
    monkeypatch.setattr(fa, "HF_HOST", "127.0.0.1")       # treat the test server as huggingface.co
    path = tmp_path / "f.part"
    fa.download([server.url + "/resolve/file.bin"], path, len(data), hashlib.sha256(data).hexdigest(),
                token="hf_secret", sleep=no_sleep)
    first, second = server.headers[-2], server.headers[-1]
    assert first.get("Authorization") == "Bearer hf_secret" and "Authorization" not in second


# ----------------------------------------------------------------------------- plan
def test_plan_counts_partial_downloads_and_extraction_space(tmp_path):
    data = payload(10_000)
    plain = asset_for("p", data, ["https://example.org/p"])
    archive = asset_for("t", data, ["https://example.org/t"], dest="outputs/assets/t",
                        extract={"format": "tar.gz", "files": {"a": 30_000}})
    (tmp_path / "outputs/assets").mkdir(parents=True)
    (tmp_path / "outputs/assets/p.bin.part").write_bytes(data[:4000])
    report = fa.plan([plain, archive], tmp_path)
    rows = {r["name"]: r for r in report["assets"]}
    assert rows["p"]["status"] == "partial" and rows["p"]["download"] == 6000
    assert rows["t"]["download"] == 10_000 and rows["t"]["size"] == 30_000
    assert report["download_bytes"] == 16_000 and report["disk"][0]["need"] == 6000 + 10_000 + 30_000
    hosts = fa.download_hosts([dict(plain, urls=["https://a.org/x", "https://m.org/y"])])
    assert hosts == [("a.org", "primary"), ("m.org", "mirror")]


# ----------------------------------------------------------------------------- self-test schema
def good_result():
    scores = {m: 90.0 + i for i, m in enumerate(self_test.METRICS)}
    return {"video": "/x/clip.mp4", "scores": scores, "tiers": {t: 95.0 for t in self_test.TIERS},
            "overall": sum(scores.values()) / 6, "status": {m: "ok" for m in self_test.METRICS},
            "reasons": {}, "person": {}, "files": {"session": "/x"}, "timings": {},
            "paper_scores": {"scores": {m: 80.0 for m in self_test.METRICS}, "overall": 80.0}}


def test_self_test_schema_check():
    assert self_test.check_result(good_result()) == []
    bad = good_result()
    bad["scores"]["joint_range"] = 101.0
    assert any("joint_range" in p for p in self_test.check_result(bad))
    bad = good_result()
    bad["status"]["self_collision"] = "missing_input"
    bad["reasons"]["self_collision"] = "SMPL-X model missing"
    assert any("SMPL-X model missing" in p for p in self_test.check_result(bad))
    bad = good_result()
    bad["overall"] = 50.0
    assert any("mean" in p for p in self_test.check_result(bad))
    bad = good_result()
    del bad["paper_scores"]
    assert self_test.check_result(bad, paper_scores=False) == [] and self_test.check_result(bad)
    bad = good_result()
    bad["paper_scores"]["overall"] = None
    assert any("paper score overall" in p for p in self_test.check_result(bad))
    bad = good_result()
    del bad["tiers"]
    assert self_test.check_result(bad) == ["missing key 'tiers'"]
