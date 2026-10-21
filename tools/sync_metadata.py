#!/usr/bin/env python3
"""Synchronize small README regions and CFF fields using Python 3.12 stdlib.

Exit codes: 0 = consistent/synchronized, 1 = stale (--check), 2 = invalid input.
No package import, network access, or YAML rewriting is required.
"""

import argparse
import ast
from datetime import date
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def load_metadata(root=ROOT):
    """Read canonical metadata without importing MEGaNorm or executing code."""
    tree = ast.parse((root / "meganorm/_version.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if len(versions) != 1 or not isinstance(versions[0], str):
        raise ValueError("Expected one literal __version__ assignment")
    with (root / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)["tool"]["meganorm"]
    for key in ("repository", "documentation"):
        value = metadata[key]
        url = urlsplit(value)
        if url.scheme != "https" or not url.netloc or any(c.isspace() for c in value):
            raise ValueError(f"Invalid HTTPS URL: {key}")
    for key in ("software-concept-doi", "paper-doi"):
        if not re.fullmatch(r"10\.\d{4,9}/[-._;()/A-Za-z0-9]+", metadata[key]):
            raise ValueError(f"Invalid DOI: {key}")
    release = metadata["release"]
    if not isinstance(release, dict):
        raise ValueError("Release metadata must be a TOML table")
    if any(release.get(k) for k in ("version", "date", "doi")):
        if release.get("version") != versions[0] or not release.get("date"):
            raise ValueError(
                "Release metadata must match the package version and include a date"
            )
        if date.fromisoformat(release["date"]).isoformat() != release["date"]:
            raise ValueError("Release date must be YYYY-MM-DD")
        if release.get("doi"):
            if not re.fullmatch(r"10\.\d{4,9}/[-._;()/A-Za-z0-9]+", release["doi"]):
                raise ValueError("Invalid release DOI")
            if release["doi"] == metadata["software-concept-doi"]:
                raise ValueError("Release DOI must differ from concept DOI")
    return {**metadata, "version": versions[0]}


def replace_one(text, pattern, value, label):
    updated, count = re.subn(pattern, lambda _: value, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one {label}; found {count}")
    return updated


def readme(text, metadata):
    repo = metadata["repository"].rstrip("/")
    docs = metadata["documentation"].rstrip("/") + "/"
    software = metadata["software-concept-doi"]
    paper = metadata["paper-doi"]
    slug = urlsplit(repo).path.strip("/")
    values = {
        "documentation": f"The full MEGaNorm documentation, including usage instructions and examples, is available at [meganorm.readthedocs.io]({docs}).",
        "ci-link": f"[GitHub Actions]({repo}/actions/workflows/tests.yml)",
        "issues": f"[GitHub issue tracker]({repo}/issues).",
        "software-doi": f"https://doi.org/{software}",
        "paper-doi": f"https://doi.org/{paper}",
    }
    badges = {
        "Tests": f"[![Tests]({repo}/actions/workflows/tests.yml/badge.svg)]({repo}/actions/workflows/tests.yml)",
        "Documentation": f"[![Documentation](https://readthedocs.org/projects/meganorm/badge/?version=latest)]({docs}en/latest/)",
        "Binder": f"[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{slug}/main?filepath=notebooks%2F)",
        "License": f"[![License](https://img.shields.io/github/license/{slug}?color=blue)]({repo}/blob/main/LICENSE)",
        "Software DOI": f"[![Software DOI](https://zenodo.org/badge/DOI/{software}.svg)](https://doi.org/{software})",
        "Paper DOI": f'[![Paper DOI](https://img.shields.io/badge/DOI-{paper.replace("-", "--").replace("/", "%2F")}-B31B1B?logo=doi\\&logoColor=white)](https://doi.org/{paper})',
        "Last Commit": f"[![Last Commit](https://img.shields.io/github/last-commit/{slug}?logo=github\\&color=informational)]({repo}/commits/main)",
    }
    for name in ("badges", "source-install", *values):
        begin = f"<!-- BEGIN MEGANORM METADATA: {name} -->"
        end = f"<!-- END MEGANORM METADATA: {name} -->"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(f"Missing or duplicate README marker: {name}")
        a = text.index(begin) + len(begin)
        b = text.index(end)
        if b < a:
            raise ValueError(f"Reversed README markers: {name}")
        content = text[a:b].strip("\n")
        if name == "badges":
            for label, value in badges.items():
                content = replace_one(
                    content, rf"^\[!\[{re.escape(label)}\].*$", value, f"{label} badge"
                )
        elif name == "source-install":
            content = replace_one(
                content, r"^git clone .+$", f"git clone {repo}.git", "clone command"
            )
        else:
            content = values[name]
        text = text[:a] + "\n" + content + "\n" + text[b:]
    return text


def citation(text, metadata):
    values = {
        "version": metadata["version"],
        "doi": metadata["release"].get("doi") or metadata["software-concept-doi"],
        "repository-code": metadata["repository"],
        "url": metadata["documentation"],
    }
    for key, value in values.items():
        text = replace_one(
            text, rf"^{key}:.*$", f"{key}: {json.dumps(value)}", f"CFF {key}"
        )
    # Only manage the top-level software release date; keep article dates intact.
    dates = re.findall(r"^date-released:.*\n?", text, flags=re.MULTILINE)
    if len(dates) > 1:
        raise ValueError("Duplicate CFF date-released")
    text = re.sub(r"^date-released:.*\n?", "", text, flags=re.MULTILINE)
    if metadata["release"].get("date"):
        text = text.replace(
            f'version: {json.dumps(metadata["version"])}\n',
            f'version: {json.dumps(metadata["version"])}\ndate-released: {json.dumps(metadata["release"]["date"])}\n',
            1,
        )
    text = replace_one(
        text,
        r"^    doi:.*$",
        f'    doi: {json.dumps(metadata["paper-doi"])}',
        "article DOI",
    )
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Report drift without writing"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument(
        "--release-tag", help="Also require release date and matching vVERSION tag"
    )
    args = parser.parse_args(argv)
    try:
        metadata = load_metadata(args.root)
        if args.release_tag and (
            args.release_tag != "v" + metadata["version"]
            or not metadata["release"].get("date")
        ):
            raise ValueError(
                "Release tag must match package version; release date is required"
            )
        updates = []
        # Render and validate every file before writing any file.
        for name, render in (("README.md", readme), ("CITATION.cff", citation)):
            path = args.root / name
            old = path.read_text(encoding="utf-8")
            new = render(old, metadata)
            if new != old:
                updates.append((path, new))
        if args.check:
            for path, _ in updates:
                print(f"Stale metadata: {path.name}")
            if not updates:
                print("Metadata is synchronized.")
            return int(bool(updates))
        for path, content in updates:
            path.write_text(content, encoding="utf-8")
            print(f"Updated {path.name}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print(f"Metadata error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
