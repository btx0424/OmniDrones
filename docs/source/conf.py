# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = 'OmniDrones'
copyright = '2023, btx0424'
author = 'btx0424'
release = '0.1'
master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
    'myst_parser',
    # "sphinxcontrib.bibtex",
]

# BibTeX configuration
bibtex_bibfiles = ["_static/refs.bib"]

templates_path = ['_templates']
exclude_patterns = []

autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_book_theme

html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/btx0424/OmniDrones",
    "use_repository_button": True,
}

html_static_path = ['_static']

# Mock out modules that are not available on RTD
autodoc_mock_imports = [
    "torch",
    "torchrl",
    "tensordict",
    "functorch",
    "numpy",
    "matplotlib",
    "scipy",
    "carb",
    "warp",
    "pxr",
    "omni",
    "omni.kit",
    "omni.usd",
    "omni.client",
    "pxr.PhysxSchema",
    "pxr.PhysicsSchemaTools",
    "omni.replicator",
    "omni.isaac.core",
    "omni.isaac.core.utils.torch",
    "omni.isaac.kit",
    "omni.isaac.cloner",
    "gym",
    "tqdm",
    "toml",
    "yaml",
]