Model-sharing transport development
===================================

Status and scope
----------------

The internal Hugging Face transport supports uploads and exact-commit downloads
of the existing validated bundles. Local verification uses synthetic HBR bundles
and a remote-service double, plus the real SDK behind an offline HTTP transport. The live service test is opt-in and must pass in an
authorized test repository before the Stage 5 remote exit gate is complete.

This is a developer interface. It does not implement ``ModelHub.push`` or
``ModelHub.fetch``, Zenodo archival publication, or the resumable two-provider
coordinator. A successful upload produces a payload receipt, not a completed
publication. Local model export, loading, prediction and adaptation work without
the optional network client.

Install and authenticate
------------------------

.. code-block:: bash

   python -m pip install -e '.[sharing]'
   hf auth login

The transport was checked against ``huggingface_hub==1.33.0`` and
``httpx==0.29.0``. The sharing extra pins ``huggingface_hub==1.33.0`` because
ignore handling and transfer behavior were checked against this SDK version.
Upgrade the pin only after rerunning the SDK-boundary checks.
Use a token scoped to the intended test repository. The SDK uses its local token
store or ``HF_TOKEN``. Keep tokens out of source code, notebooks, reports and
model metadata.

Create a dedicated **public, ungated model repository** under a namespace you
may write to. This transport neither creates repositories nor changes their
visibility. ``preflight(..., write=True)`` asks the service to verify write
permission; the service checks authorization again when committing. A namespace
name does not confer permission. The canonical repository ID must match the
requested ID; renamed or transferred aliases are rejected.

Internal usage
--------------

.. code-block:: python

   from meganorm.sharing.huggingface import HuggingFaceTransport

   transport = HuggingFaceTransport()
   receipt = transport.upload_bundle(
       bundle.path,
       repo_id="your-namespace/synthetic-test-model",
       version="1.0.0",
       metadata_reviewed=True,
   )
   local = transport.download_bundle(receipt, output_dir="downloaded-model")

The bundle's ``model_id`` and ``version`` must match the destination. Export the
bundle with those values first. ``metadata_reviewed=True`` affirms that the
public labels and metadata were deliberately reviewed. It does not replace
payload validation or establish scientific applicability.

The upload uses the same privately staged bytes that passed the existing deep
bundle validation. Changing the input file after validation does not change the
upload. The version's ZIP and small identity/digest record are added in one
conditional commit. A required file excluded by repository ignore rules rejects
the entire commit; adjust those rules before retrying. The SDK may have staged
unreferenced blobs, but no partial release is committed:


.. code-block:: text

   releases/1.0.0/model.zip
   releases/1.0.0/payload.json

The transport never recursively uploads an analysis directory. Existing version
plus the same digest is a verified, non-mutating success; different content is a
conflict. Concurrent changes use the SDK's ``parent_commit`` condition. An
uncertain commit response triggers a read and verification, never an automatic
second mutation. If verification is unavailable, ``IncompletePublicationError``
asks the caller to rerun and reconcile. An unrelated competing commit requires an
explicit retry. Service authorization failures remain authorization failures.

A receipt contains the exact commit, repository, version, digest and size.
Downloads require that receipt and verify remote sizes, the archive digest,
manifest, file inventory and approved model state before exposing local output.
The destination must be new. An idempotent upload retry may return a newer exact
commit containing the same bytes; it does not promise the original creation
commit. Owners can still change repository state outside MEGaNorm, so retain
exact receipts.

``BundleLimits`` bounds declared remote sizes before transfer. Anonymous HTTPS
reads pin the commit and enforce the declared byte budget before every local
write, including an overflow probe. Encoded responses are rejected. Descriptors
are additionally limited to 64 KiB. Partial downloads and private working files
are removed on failure. Persistent offline caching remains a later stage.

SDK uploads run in an isolated Python worker. Credentials enter through a private
stdin pipe, never command-line arguments; only a sanitized status is returned.
The worker uses buffered HTTP LFS transfers, verifies ignore flags before the
conditional commit, and blocks credentials sent outside the HTTPS Hub origin.
Transfer logging is disabled inside that worker so signed storage URLs do not
appear in application logs. The application's SDK client factory and logging
configuration remain unchanged. This adds process startup overhead and requires
an ordinary Python installation with subprocess support. Requests have a
60-second timeout; an upload worker has a ten-minute limit. An expired upload
still requires reconciliation because remote success may be uncertain.

Archival linkage
----------------

``read_descriptor(repo_id, version, revision=None)`` resolves the current branch
once, or uses a supplied full 40-character commit. It returns ``None`` when no
linkage exists. ``write_descriptor(document, metadata_reviewed=True)`` stores a
separate ``releases/<version>/release.json``. It checks both the referenced
payload commit and the current version before writing. Identical linkage may be
retried; different linkage cannot replace an existing descriptor.

``HubDescriptor`` contains schema version 1, repository, version, payload commit,
bundle digest and a strict ``ZenodoReference``. ``DescriptorReceipt`` adds the
commit containing that document. The descriptor excludes its own commit to avoid
a circular identity. These methods validate structure and Hugging Face bytes;
the future publication coordinator must independently confirm that the Zenodo
record is published and has the intended access policy. Do not use an invented
DOI to exercise this method on a live repository.

Testing
-------

Local tests require the sharing extra and do not access the service:

.. code-block:: bash

   python -m pytest tests/sharing/test_huggingface.py \
     tests/sharing/test_huggingface_security.py \
     tests/sharing/test_huggingface_live.py -q

The live test requires an existing dedicated public repository, a write-capable
credential in the normal SDK store or ``HF_TOKEN``, and a separate read-only token
in ``MEGANORM_HF_READ_TOKEN``. Configure credentials locally; do not paste them
into reports or chat. Set the repository and opt-in flag explicitly:

.. code-block:: bash

   export MEGANORM_HF_TEST_REPO='your-namespace/meganorm-synthetic-test'
   MEGANORM_HF_LIVE=1 python -m pytest \
     tests/sharing/test_huggingface_live.py -q -s

That command authorizes synthetic uploads to the named repository. It fits a
small HBR fixture, checks read-only denial before any write, uploads two unique
versions, verifies retries and conflicting-version rejection, and compares the
downloaded bytes. The printed receipts and ``live-receipts.json`` in pytest's
temporary output identify the artifacts. It creates no Zenodo records or
completion descriptor and performs no cleanup of remote versions. Review and
remove the dedicated test repository manually if appropriate. Ordinary CI leaves
this test skipped.

Provider references
-------------------

The implementation follows the official
`HfApi reference <https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api>`_
for write authorization and conditional commits, and the
`download guide <https://huggingface.co/docs/huggingface_hub/en/guides/download>`_
for exact revisions. Installed SDK source was also inspected for commit retry
behavior, repository ignore flags, HTTP LFS verification and transfer logging.
