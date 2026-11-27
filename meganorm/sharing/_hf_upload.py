"""Private SDK upload worker, executed in a separate interpreter.

The SDK owns a process-global HTTP client and transfer logging. Isolating uploads
lets us enforce request credentials and silence signed URLs without modifying an
application's clients, hooks, logging configuration or Xet environment settings.
Only JSON status leaves the worker; credentials enter via stdin, never argv.
"""

from contextlib import ExitStack
import base64
import io
import json
import logging
import sys

ENDPOINT = "https://huggingface.co"


def _guard_request(request):
    from huggingface_hub import constants

    if constants.is_offline_mode():
        raise RuntimeError("Offline mode is enabled.")
    url = request.url
    if url.scheme != "https" or url.username or url.password:
        raise PermissionError("Unsafe upload request.")
    if (url.host != "huggingface.co" or url.port not in (None, 443)) and (
        "authorization" in request.headers or "cookie" in request.headers
    ):
        raise PermissionError("Credentials may only be sent to the Hub origin.")


def _client_factory():
    import httpx

    return httpx.Client(
        event_hooks={"request": [_guard_request]},
        follow_redirects=True,
        timeout=60,
    )


def perform_upload(payload):
    """One conditional SDK commit; return no provider text or response objects."""
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub.utils import get_session

    old_logging = logging.root.manager.disable
    logging.disable(sys.maxsize)
    session = get_session()
    hooks = session.event_hooks["request"]
    installed = _guard_request not in hooks
    if installed:
        hooks.insert(0, _guard_request)
    try:
        api = HfApi(endpoint=ENDPOINT, token=payload["token"], library_name="meganorm")
        repo_id = payload["repo_id"]
        info = api.repo_info(repo_id, repo_type="model", revision="main", timeout=30)
        if info.id != repo_id or info.private is not False or info.gated is not False:
            return {"error": "authorization"}
        if info.sha != payload["parent_commit"]:
            return {"error": "conflict"}
        with ExitStack() as stack:
            operations = []
            for item in payload["files"]:
                # Buffered streams select the SDK's HTTP LFS path, whose every
                # request passes our guard. No native Xet transfer bypasses it.
                stream = (
                    stack.enter_context(open(item["local_path"], "rb"))
                    if "local_path" in item
                    else stack.enter_context(io.BytesIO(base64.b64decode(item["data"])))
                )
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=item["path"], path_or_fileobj=stream
                    )
                )
            api.preupload_lfs_files(
                repo_id,
                operations,
                repo_type="model",
                revision="main",
                num_threads=1,
                free_memory=False,
            )
            # SDK 1.33 mutates these flags and otherwise silently drops ignored
            # additions. Refuse the entire commit, including the identity record.
            if any(op._should_ignore is not False for op in operations):
                return {"error": "ignored"}
            result = api.create_commit(
                repo_id,
                operations,
                commit_message="Store validated MEGaNorm model-sharing artifact",
                repo_type="model",
                revision="main",
                parent_commit=payload["parent_commit"],
                create_pr=False,
                num_threads=1,
            )
            return {"oid": result.oid}
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in (401, 403, 404):
            return {"error": "authorization"}
        if status in (409, 412):
            return {"error": "conflict"}
        return {"error": "transport"}
    finally:
        if installed:
            hooks.remove(_guard_request)
        logging.disable(old_logging)


def main():
    from huggingface_hub import set_client_factory

    set_client_factory(_client_factory)
    try:
        result = perform_upload(json.load(sys.stdin))
    except Exception:
        result = {"error": "transport"}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
