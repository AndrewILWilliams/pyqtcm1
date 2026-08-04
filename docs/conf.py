import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'pyqtcm1'
author = 'Andrew I. L. Williams'
copyright = '2026, Andrew I. L. Williams'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}
html_theme = 'sphinx_rtd_theme'
html_title = 'pyqtcm1'
exclude_patterns = ['_build']
