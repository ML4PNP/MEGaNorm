# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MEGaNorm'
copyright = '2025, Seyed Mostafa Kia'
author = 'Seyed Mostafa Kia, Mohammad Zamanzadeh, Ymke Verduyn'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc.typehints",
    "numpydoc"
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# Logo path (relative to html_static_path)
html_logo = '_static/logo.png'

import os
import sys
sys.path.insert(0, os.path.abspath('../../meganorm'))



html_theme = "sphinx_rtd_theme"
