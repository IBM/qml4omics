# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))



project = 'qbiocode'
copyright = '2025 IBM Research, Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida'
author = 'Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
     "nbsphinx",
   "myst_parser",
  
  # "myst_nb"
]

templates_path = ['_templates']
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_static_path = ['_static']
nbsphinx_execute = 'never' # O 'auto', 'always'. 'never'

# Impostazioni di MyST-Parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

def run_apidoc(app):
    """Generate API documentation"""
    import better_apidoc

    better_apidoc.APP = app
    better_apidoc.main(
        [
            "better-apidoc",
            "-t",
            os.path.join(".", "_templates"),
            "--force",
            "--no-toc",
            "--separate",
            "-o",
            os.path.join("source/", "api"),
            os.path.join("..", "qbiocode"),
        ]
    )


# -- Extension configuration -------------------------------------------------
add_module_names = False


napoleon_google_docstring = True
napoleon_include_init_with_doc = True

coverage_ignore_modules = []
coverage_ignore_functions = []
coverage_ignore_classes = []


myst_enable_extensions = [
    "colon_fence",  # Ensures ::: blocks work
  #  "linkify",
    "strikethrough",
    "tasklist",
    # ... any other extensions you want
]


coverage_show_missing_items = True
html_theme = 'pydata_sphinx_theme' #'sphinx_rtd_theme' # 'furo'

html_show_sourcelink = False

html_logo = "_static/QBioCode_logo.png"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/IBM/QBioCode",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ],
    "show_prev_next": False,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "header_links_before_dropdown": 8,
}

html_context = {
    "default_mode": "light",
}

html_sidebars = {
    "index": [],
    "examples/index": [],
    "**": [],
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

def setup(app):
    app.connect("builder-inited", run_apidoc)
