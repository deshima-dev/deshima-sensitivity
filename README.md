![DESHIMA logo](http://deshima.ewi.tudelft.nl/image/deshima_logo.png)

# deshima-sensitivity

[![PyPI](https://img.shields.io/pypi/v/deshima-sensitivity.svg?label=PyPI&style=flat-square)](https://pypi.org/pypi/deshima-sensitivity/)
[![Python](https://img.shields.io/pypi/pyversions/deshima-sensitivity.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/pypi/deshima-sensitivity/)
[![Test](https://img.shields.io/github/workflow/status/deshima-dev/deshima-sensitivity/Test?logo=github&label=Test&style=flat-square)](https://github.com/deshima-dev/deshima-sensitivity/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)

Sensitivity calculator for DESHIMA-type spectrometers

## Overview

deshima-sensitivity is a Python package which enables to calculate observation sensitivity of DESHIMA-type spectrometers.
Currently it is mainly used to estimate the observation sensitivity of [DESHIMA](http://deshima.ewi.tudelft.nl) and its successors.

An online Jupyter notebook is available for DESHIMA collaborators to calculate the sensitivity and the mapping speed of the DESHIMA 2.0 by themselves.
Click the budge below to open it in [Google colaboratory](http://colab.research.google.com/) (a Google account is necessary to re-run it).

### Stable version (recommended)

[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deshima-dev/deshima-sensitivity/blob/v0.2.3/sensitivity.ipynb)

### Latest version

[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deshima-dev/deshima-sensitivity/blob/master/sensitivity.ipynb)

In the case of running it in a local Python environment, please follow the requirements and the installation guide below.

## Requirements

- **Python:** 3.6, 3.7, or 3.8 (tested by the authors)
- **Dependencies:** See [pyproject.toml](https://github.com/deshima-dev/deshima-sensitivity/blob/master/pyproject.toml)

## Installation

```shell
$ pip install deshima-sensitivity
``` 
