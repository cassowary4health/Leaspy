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
import warnings
import sphinx_gallery
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# -- Project information -----------------------------------------------------

project = 'Leaspy'
author = 'Igor Koval, Raphael Couronne, Arnaud Valladier, Etienne Maheux, Benoit Martin, Pierre-Emmanuel Poulet, Cecile Di Folco, Juliette Ortholand, Mkrtich Vatinyan, Benoit Sauty De Chalon, Stanley Durrleman'
copyright = '2017-2021, ' + author

# The full version, including alpha/beta/rc tags
release = '1.0.3-dev'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinx.ext.todo',
    # 'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.viewcode',
    'numpydoc',
    # 'rinoh.frontend.sphinx',
    'sphinx_rtd_theme',
    # 'sphinx.ext.napoleon',
    # 'pytsdtwdoc',
    # 'sphinx_gallery.gen_gallery',
]


# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = True

# - From Johann conf.py
# Use svg images for math stuff
imgmath_image_format = 'svg'
# pngmath / imgmath compatibility layer for different sphinx versions
import sphinx
from distutils.version import LooseVersion
if LooseVersion(sphinx.__version__) < LooseVersion('1.4'):
    extensions.append('sphinx.ext.pngmath')
else:
    extensions.append('sphinx.ext.imgmath')

autodoc_default_flags = ['members', 'inherited-members', 'private-members']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
highlight_language = 'python3'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#---sphinx-themes-----
# html_theme = 'neo_rtd_theme'
import sphinx_theme
# html_theme_path = [sphinx_theme.get_html_theme_path()]

# html_theme = 'alabaster'
# html_theme = 'sphinx-theme'
html_theme = 'sphinx_rtd_theme'

add_function_parentheses = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    'display_version': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    # Logo and description
    # 'description': 'LEArning Spatiotemporal Patterns in Python',
    # 'logo': 'leaspy_logo.png',
    # 'logo_name': 'false',
    # 'logo_text_align': 'center',

    # GitHub stuff
    # 'github_banner': 'true',
    # 'github_repo': 'pyts',
    # 'github_type': 'star',
    # 'github_user': 'johannfaouzi',

    # Page and sidebar widths
    # 'page_width': '1300px',
    'body_max_width': '850px',
    # 'sidebar_width': '250px',

    # Related links
    # 'show_related': 'true',
    # 'show_relbar_bottom': 'true',

    # Font sizes
    # 'font_size': '15px',
    # 'code_font_size': '13px'
}

html_context = {
    "display_gitlab": True,  # Integrate Github
    "gitlab_user": "getleaspy",  # Username
    "gitlab_repo": "leaspy",  # Repo name
    "gitlab_version": "master",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

# Custom CSS files
# html_css_files = [
#     'custom.css',
# ]# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "leaspy_logo.png"

html_title = "Leaspy"
# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "Leaspy documentation"

# -- Options for LaTeX output ---------------------------------------------
latex_engine = 'pdflatex'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp'
    # Additional stuff for the LaTeX preamble.
}

# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None)
}
