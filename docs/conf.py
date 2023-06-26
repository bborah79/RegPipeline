# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
#sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep + '')
sys.path.insert(0,
                str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                + os.sep + 'regression_pipeline')

project = 'Regression pipeline'
copyright = '2023, Bhaskar Borah'
author = 'Bhaskar Borah'
release = '0.00, 2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.coverage',
              'sphinx.ext.todo',
              'numpydoc'
             ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
