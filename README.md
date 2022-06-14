# deshima-sensitivity

[![Release](https://img.shields.io/pypi/v/deshima-sensitivity?label=Release&color=cornflowerblue&style=flat-square)](https://pypi.org/pypi/deshima-sensitivity/)
[![Python](https://img.shields.io/pypi/pyversions/deshima-sensitivity?label=Python&color=cornflowerblue&style=flat-square)](https://pypi.org/pypi/deshima-sensitivity/)
[![Downloads](https://img.shields.io/pypi/dm/deshima-sensitivity?label=Downloads&color=cornflowerblue&style=flat-square)](https://pepy.tech/project/deshima-sensitivity)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3966839-cornflowerblue?style=flat-square)](https://doi.org/10.5281/zenodo.3966839)
[![Tests](https://img.shields.io/github/workflow/status/deshima-dev/deshima-sensitivity/Tests?label=Tests&style=flat-square)](https://github.com/deshima-dev/deshima-sensitivity/actions/tests.yml)

Sensitivity calculator for DESHIMA-type spectrometers

## Overview

deshima-sensitivity is a Python package which enables to calculate observation sensitivity of DESHIMA-type spectrometers.
Currently it is mainly used to estimate the observation sensitivity of [DESHIMA](http://deshima.ewi.tudelft.nl) and its successors.

An online Jupyter notebook is available for DESHIMA collaborators to calculate the sensitivity and the mapping speed of the DESHIMA 2.0 by themselves.
Click the budge below to open it in [Google colaboratory](http://colab.research.google.com/) (a Google account is necessary to re-run it).

### Stable version (recommended)

[![open stable version in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deshima-dev/deshima-sensitivity/blob/v0.4.1/sensitivity.ipynb)

### Latest version

[![open latest version in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deshima-dev/deshima-sensitivity/blob/main/sensitivity.ipynb)

In the case of running it in a local Python environment, please follow the requirements and the installation guide below.

## Requirements

- **Python:** 3.7, 3.8, or 3.9 (tested by the authors)
- **Dependencies:** See [pyproject.toml](https://github.com/deshima-dev/deshima-sensitivity/blob/main/pyproject.toml)

## Installation

```shell
$ pip install deshima-sensitivity
```

## Development environment

The following steps can create a standalone development environment (VS Code + Python).

1. Install [VS Code] and [Docker Desktop], and launch them
1. Install the [Remote Containers] extension to VS Code
1. Clone this repository
1. Open the repository by VS Code
1. Choose `Reopen in Container` from the [Command Palette]

[Command Palette]: https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette
[Docker Desktop]: https://www.docker.com/products/docker-desktop
[Remote Containers]: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
[VS Code]: https://code.visualstudio.com

