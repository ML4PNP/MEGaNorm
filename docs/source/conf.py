# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import runpy
from pathlib import Path
from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MEGaNorm"
copyright = "2025, Seyed Mostafa Kia"
author = "Seyed Mostafa Kia, Mohammad Zamanzadeh, Ymke Verduyn"
ROOT = Path(__file__).resolve().parents[2]
metadata = runpy.run_path(str(ROOT / "tools/sync_metadata.py"))["load_metadata"](ROOT)
release = get_version("meganorm")
if release != metadata["version"]:
    raise RuntimeError(
        "Installed MEGaNorm version differs from checkout; reinstall with pip install -e ."
    )
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
]

autodoc_typehints = "description"
autodoc_typehints_format = "short"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": metadata["repository"],
    "external_links": [
        {
            "name": "ML4PNP Lab",
            "url": "https://ml4pnp.github.io/",
        },
    ],
    "show_toc_level": 2,
    "navigation_with_keys": True,
}
html_static_path = ["_static"]

# Logo path (relative to html_static_path)
html_logo = "_static/logo.png"

sys.path.insert(0, os.path.abspath("../../meganorm"))

# Inline substitutions include link targets; code blocks do not expand RST
# substitutions. Keep release examples linked to published image tags.
software_doi = metadata["software-concept-doi"]
paper_doi = metadata["paper-doi"]
repository_url = metadata["repository"].rstrip("/")
documentation_url = metadata["documentation"]

rst_epilog = f"""
.. |version| replace:: {release}
.. |software_doi| replace:: {software_doi}
.. |paper_doi| replace:: {paper_doi}
.. |repository_url| replace:: {repository_url}
.. |documentation_url| replace:: {documentation_url}
.. |repository_link| replace:: `GitHub <{repository_url}>`__
.. |documentation_link| replace:: `Documentation <{documentation_url}>`__
.. |paper_link| replace:: `Scientific Paper <https://doi.org/{paper_doi}>`__
.. |software_link| replace:: `Software DOI <https://doi.org/{software_doi}>`__
.. |software_doi_link| replace:: `DOI: {software_doi} <https://doi.org/{software_doi}>`__
.. |paper_doi_link| replace:: `DOI: {paper_doi} <https://doi.org/{paper_doi}>`__
.. |notebooks_link| replace:: `Explore the example notebooks on GitHub <{repository_url}/tree/main/notebooks>`__
"""
