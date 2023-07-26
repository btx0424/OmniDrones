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

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
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
]