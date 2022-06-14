# project information
author = "DESHIMA software team"
copyright = "2020-2022 DESHIMA software team"


# general configuration
add_module_names = False
autodoc_typehints = "both"
autodoc_typehints_format = "short"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]


# options for HTML output
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/deshima-dev/deshima-sensitivity/",
}
