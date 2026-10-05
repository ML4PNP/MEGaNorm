"""End-to-end release metadata checks without scientific dependencies."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    for name in ("pyproject.toml", "README.md", "CITATION.cff"):
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "meganorm").mkdir()
    shutil.copy(ROOT / "meganorm/_version.py", tmp_path / "meganorm/_version.py")
    return tmp_path


def run(root, *args):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/sync_metadata.py"),
            "--root",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=root.parent,
    )


def test_check_detects_drift_without_writing_and_sync_is_idempotent(checkout):
    readme = checkout / "README.md"
    readme.write_text(readme.read_text().replace("zenodo.15441319", "zenodo.99999"))
    before = {p.name: p.read_bytes() for p in checkout.glob("*") if p.is_file()}
    result = run(checkout, "--check")
    assert result.returncode == 1, result.stderr
    assert before == {p.name: p.read_bytes() for p in checkout.glob("*") if p.is_file()}
    assert run(checkout).returncode == 0
    after = readme.read_bytes()
    assert run(checkout).returncode == 0
    assert readme.read_bytes() == after
    assert run(checkout, "--check").returncode == 0


def test_release_fields_must_belong_to_checkout_version(checkout):
    config = checkout / "pyproject.toml"
    config.write_text(
        config.read_text()
        .replace('version = ""', 'version = "0.2.0"')
        .replace('date = ""', 'date = "2026-07-10"')
    )
    before = (checkout / "CITATION.cff").read_bytes()
    result = run(checkout)
    assert result.returncode == 2
    assert "release" in result.stderr.lower()
    assert (checkout / "CITATION.cff").read_bytes() == before


def test_release_date_and_optional_version_doi(checkout):
    config = checkout / "pyproject.toml"
    config.write_text(
        config.read_text()
        .replace('version = ""', 'version = "0.2.2"')
        .replace('date = ""', 'date = "2026-10-20"')
        .replace('doi = ""', 'doi = "10.5281/zenodo.99999999"')
    )
    assert run(checkout).returncode == 0
    cff = (checkout / "CITATION.cff").read_text()
    assert 'date-released: "2026-10-20"' in cff
    assert '\ndoi: "10.5281/zenodo.99999999"' in cff
    assert "10.5281/zenodo.15441319" in (checkout / "README.md").read_text()
    assert run(checkout, "--check", "--release-tag", "v0.2.2").returncode == 0
    assert run(checkout, "--check", "--release-tag", "v0.2.1").returncode == 2


def test_development_has_no_release_date_or_version_doi(checkout):
    assert run(checkout).returncode == 0
    cff = (checkout / "CITATION.cff").read_text()
    assert "date-released:" not in cff
    assert '\ndoi: "10.5281/zenodo.15441319"' in cff
    assert 'version: "0.2.2"' in cff
    assert run(checkout, "--check", "--release-tag", "v0.2.2").returncode == 2


@pytest.mark.parametrize("replacement", ["missing", "duplicate"])
def test_broken_markers_fail_before_any_write(checkout, replacement):
    readme = checkout / "README.md"
    marker = "<!-- BEGIN MEGANORM METADATA: software-doi -->"
    content = readme.read_text()
    readme.write_text(
        content.replace(
            marker, "" if replacement == "missing" else marker + "\n" + marker
        )
    )
    before = (checkout / "CITATION.cff").read_bytes()
    result = run(checkout)
    assert result.returncode == 2
    assert "marker" in result.stderr.lower()
    assert (checkout / "CITATION.cff").read_bytes() == before


def test_sync_preserves_authors_paper_and_prose(checkout):
    cff = checkout / "CITATION.cff"
    before = cff.read_text()
    assert run(checkout).returncode == 0
    after = cff.read_text()
    assert before.split("authors:\n", 1)[1].split("\nversion:", 1)[0] in after
    assert before.split("references:\n", 1)[1] == after.split("references:\n", 1)[1]
    assert "**MEGaNorm** is a Python package" in (checkout / "README.md").read_text()


@pytest.mark.parametrize("date_value", ["2026-02-30", "2026-1-01"])
def test_invalid_release_dates_fail_without_writing(checkout, date_value):
    config = checkout / "pyproject.toml"
    config.write_text(
        config.read_text()
        .replace('version = ""', 'version = "0.2.2"')
        .replace('date = ""', f'date = "{date_value}"')
    )
    before = (checkout / "README.md").read_bytes()
    assert run(checkout).returncode == 2
    assert (checkout / "README.md").read_bytes() == before


def test_changed_canonical_urls_and_paper_doi_propagate(checkout):
    config = checkout / "pyproject.toml"
    config.write_text(
        config.read_text()
        .replace("github.com/ML4PNP/MEGaNorm", "github.com/example/MEGaNorm")
        .replace("https://meganorm.readthedocs.io/", "https://docs.example.org/")
        .replace("10.1038/s42003-026-09825-2", "10.1038/example-article")
    )
    assert run(checkout, "--check").returncode == 1
    assert run(checkout).returncode == 0
    readme = (checkout / "README.md").read_text()
    cff = (checkout / "CITATION.cff").read_text()
    assert "github.com/ML4PNP/MEGaNorm" not in readme
    assert "git clone https://github.com/example/MEGaNorm.git" in readme
    assert "https://docs.example.org/en/latest/" in readme
    assert "10.1038/example-article" in readme and "10.1038/example-article" in cff
    assert run(checkout, "--check").returncode == 0


def test_editorial_sections_are_not_duplicated():
    readme = (ROOT / "README.md").read_text()
    assert readme.count("## Getting started\n") == 1
    assert readme.count("After installation, cortical reconstruction") == 1


def test_scalar_release_config_returns_input_error(checkout):
    config = checkout / "pyproject.toml"
    config.write_text(
        config.read_text().split("[tool.meganorm.release]")[0] + 'release = "invalid"\n'
    )
    result = run(checkout)
    assert result.returncode == 2
    assert "table" in result.stderr.lower()
    assert "Traceback" not in result.stderr
