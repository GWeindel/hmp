# ruff: noqa
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "hmp"
copyright = "2025, Gabriel Weindel, Maarten Schermer, Raoul Schram, Leendert van Maanen, Jelmer Borst"
author = "Gabriel Weindel, Maarten Schermer, Raoul Schram, Leendert van Maanen, Jelmer Borst"


# The full version, including alpha/beta/rc tags
release = '1.0.0-b.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "images/logos/logo.png"
html_theme_options = {
    "logo": {
        "image_light": "logo.png",  # adjust if you have dark/light variants
        "image_dark": "logo.png",
        "text": "Hidden Multivariate Pattern",
        "alt_text": "HMP Logo",
    },
    "logo_link": "/",  # optional: makes logo link to home
    "navbar_start": ["navbar-logo"],
    # Increase logo size (height in px, adjust as needed)
    "logo": {
        "image_light": "logo.png",
        "image_dark": "logo.png",
        "text": "HMP",
        "alt_text": "HMP Logo",
        "height": "160px",  # set your desired height
    },
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
napoleon_use_param = Truehtml_logo = "images/logos/logo.png"

