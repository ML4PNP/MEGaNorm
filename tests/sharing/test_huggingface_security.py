"""Offline regressions at the real SDK and streaming HTTP boundaries."""

import importlib
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("huggingface_hub")
import httpx
import huggingface_hub
from huggingface_hub import CommitOperationAdd

from meganorm.sharing.errors import AuthorizationError, IntegrityError
from meganorm.sharing.huggingface import HuggingFaceTransport, PayloadReceipt
from .test_huggingface import REPO, TOKEN, PREFIX, HubService

HEAD = "a" * 40
SIGNED = "PRIVATE_SIGNED_QUERY"


def receipt():
    return PayloadReceipt(
        repo_id=REPO,
        version="1.0.0",
        bundle_sha256="b" * 64,
        size_bytes=10,
        commit=HEAD,
    )


@pytest.mark.parametrize("write", [False, True])
def test_canonical_repository_must_match_before_authorization(tmp_path, write):
    client = HuggingFaceTransport(token=False)
    service = HubService(tmp_path)
    client._api = service
    original = service.repo_info

    def moved(*args, **kwargs):
        info = original(*args, **kwargs)
        info.id = "different-owner/example"
        return info

    service.repo_info = moved
    with pytest.raises(AuthorizationError):
        client.preflight(REPO, write=write)
    assert not any(c[0] == "auth" for c in service.calls)


def test_output_parent_symlink_loop_is_sanitized_before_network(tmp_path):
    client = HuggingFaceTransport(token=False)
    client._api = object()
    link = tmp_path / "PRIVATE_DIRECTORY"
    link.symlink_to(link.name)
    with pytest.raises(IntegrityError) as error:
        client.download_bundle(receipt(), output_dir=link / "new")
    assert "PRIVATE_DIRECTORY" not in str(error.value)
    assert error.value.__context__ is None


def test_download_stops_before_writing_more_than_declared_size(tmp_path, monkeypatch):
    module = importlib.import_module("meganorm.sharing.huggingface")
    client = HuggingFaceTransport(token=False)
    body = io.BytesIO(b"x" * 1024**2)
    reads = []

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            body.close()

        def read(self, size):
            reads.append(size)
            assert size <= 11
            return body.read(size)

    monkeypatch.setattr(module, "_open_public", lambda url: Response(), raising=False)

    # The old SDK download writes the whole response before checking the budget.
    def unbounded(*args, **kwargs):
        out = Path(kwargs["cache_dir"]) / "blob"
        out.parent.mkdir()
        out.write_bytes(b"x" * 1024**2)
        return out

    client._api = SimpleNamespace(hf_hub_download=unbounded)
    with pytest.raises(IntegrityError):
        client._fetch(REPO, PREFIX + "release.json", HEAD, 10, tmp_path)
    assert reads == [11]
    assert body.closed
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def sdk_boundary(monkeypatch):
    """Keep real SDK request construction; replace only HTTP and process launch."""
    import huggingface_hub.utils._http as sdk_http
    import subprocess

    requests = []
    state = {"ignored": None, "verify_host": "huggingface.co", "fail_upload": False}
    monkeypatch.setattr(sdk_http.time, "sleep", lambda seconds: None)

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/preupload/main"):
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "path": f["path"],
                            "uploadMode": (
                                "lfs" if f["path"].endswith(".zip") else "regular"
                            ),
                            "shouldIgnore": f["path"] == state["ignored"],
                        }
                        for f in json.loads(request.content)["files"]
                    ]
                },
            )
        if request.url.path.endswith("/info/lfs/objects/batch"):
            objects = json.loads(request.content)["objects"]
            return httpx.Response(
                200,
                json={
                    "objects": [
                        {
                            **o,
                            "actions": {
                                "upload": {
                                    "href": f"https://storage.example/blob?sig={SIGNED}"
                                },
                                "verify": {
                                    "href": f"https://{state['verify_host']}/verify"
                                },
                            },
                        }
                        for o in objects
                    ]
                },
            )
        if request.url.host == "storage.example":
            if state["fail_upload"]:
                state["fail_upload"] = False
                return httpx.Response(503)
            return httpx.Response(200)
        if request.url.path == "/verify":
            return httpx.Response(200)
        if request.url.path.endswith("/commit/main"):
            return httpx.Response(
                200,
                json={
                    "commitOid": "c" * 40,
                    "commitUrl": f"https://huggingface.co/{REPO}/commit/" + "c" * 40,
                },
            )
        if request.url.path.endswith("/revision/main"):
            return httpx.Response(
                200, json={"id": REPO, "sha": HEAD, "private": False, "gated": False}
            )
        if request.url.path == "/retry":
            return httpx.Response(503)
        raise AssertionError(f"Unexpected HTTP path: {request.url.path}")

    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    monkeypatch.setattr(sdk_http, "_GLOBAL_CLIENT", http)
    monkeypatch.setattr(sdk_http, "_GLOBAL_CLIENT_FACTORY", lambda: http)

    # New upload worker executes in isolation in production; run its entry point
    # here so the same offline HTTP transport can inspect every actual request.
    def run_worker(args, *, input, **kwargs):
        worker = importlib.import_module("meganorm.sharing._hf_upload")
        result = worker.perform_upload(json.loads(input))
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(subprocess, "run", run_worker)
    yield state, requests, http
    http.close()


def operations():
    return [
        CommitOperationAdd(
            path_in_repo=PREFIX + "model.zip", path_or_fileobj=io.BytesIO(b"ZIP")
        ),
        CommitOperationAdd(
            path_in_repo=PREFIX + "payload.json", path_or_fileobj=io.BytesIO(b"{}")
        ),
    ]


@pytest.mark.parametrize("name", ["model.zip", "payload.json"])
def test_ignored_addition_never_commits_partial_version(sdk_boundary, name):
    state, requests, _ = sdk_boundary
    state["ignored"] = PREFIX + name
    client = HuggingFaceTransport(token=TOKEN)
    with pytest.raises(IntegrityError):
        client._commit(REPO, HEAD, operations(), lambda commit: commit)
    assert not any(r.url.path.endswith("/commit/main") for r in requests)


def test_lfs_verification_never_receives_hub_token_off_origin(sdk_boundary):
    state, requests, _ = sdk_boundary
    state["verify_host"] = "untrusted.example"
    client = HuggingFaceTransport(token=TOKEN)
    try:
        client._commit(REPO, HEAD, operations(), lambda commit: None)
    except Exception:
        pass
    assert not any(r.url.host == "untrusted.example" for r in requests)


def test_upload_signed_urls_are_absent_from_logs(sdk_boundary, caplog):
    _, requests, _ = sdk_boundary
    caplog.set_level(logging.DEBUG)
    client = HuggingFaceTransport(token=TOKEN)
    assert client._commit(REPO, HEAD, operations(), lambda commit: commit) == "c" * 40
    assert any(r.url.host == "storage.example" for r in requests)
    assert SIGNED not in caplog.text
    assert TOKEN not in caplog.text


def test_retry_signed_urls_are_absent_from_logs(sdk_boundary, caplog):
    state, requests, _ = sdk_boundary
    state["fail_upload"] = True
    caplog.set_level(logging.DEBUG)
    client = HuggingFaceTransport(token=TOKEN)
    assert client._commit(REPO, HEAD, operations(), lambda commit: commit) == "c" * 40
    assert sum(r.url.host == "storage.example" for r in requests) == 2
    assert SIGNED not in caplog.text and TOKEN not in caplog.text
    logging.getLogger("application").warning("Logging still works after upload")
    assert "Logging still works after upload" in caplog.text


def test_upload_worker_uses_validated_local_file_and_keeps_credentials_off_argv(
    sdk_boundary, tmp_path, monkeypatch
):
    _, requests, _ = sdk_boundary
    original = __import__("subprocess").run
    source = tmp_path / "snapshot.zip"
    source.write_bytes(b"validated zip snapshot")

    def inspect(args, **kwargs):
        assert TOKEN not in str(args)
        assert json.loads(kwargs["input"])["token"] == TOKEN
        return original(args, **kwargs)

    monkeypatch.setattr("subprocess.run", inspect)
    op = CommitOperationAdd(path_in_repo=PREFIX + "model.zip", path_or_fileobj=source)
    client = HuggingFaceTransport(token=TOKEN)
    assert client._commit(REPO, HEAD, [op], lambda commit: commit) == "c" * 40
    request = next(r for r in requests if r.url.host == "storage.example")
    assert request.content == source.read_bytes()


def test_worker_entrypoint_is_offline_and_sanitized():
    import os
    import subprocess
    import meganorm.sharing._hf_upload as worker

    env = dict(os.environ, HF_HUB_OFFLINE="1")
    # The isolated entry point must honor offline mode even with its own client.
    result = subprocess.run(
        [__import__("sys").executable, worker.__file__],
        input=json.dumps(dict(repo_id=REPO, token=False, parent_commit=HEAD, files=[])),
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"error": "transport"}
    assert not result.stderr
