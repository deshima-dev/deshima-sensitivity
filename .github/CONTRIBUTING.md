# How to contribute

Thank you for contributing deshima-sensitivity!
If you have some ideas to propose, please follow the contribution guide.
We use [GitHub flow] for developing and managing the project.
The first section describes how to contribute with it.
The second and third sections explain how to prepare a local development environment and our automated workflows in GitHub Actions, respectively.


## Contributing with GitHub flow

### Create a branch

First of all, [create an issue] with a simple title and get an issue ID (e.g., `#31`).
For example, if you propose to add functions for plotting something, the title of the issue would be `Add plotting feature`.
Using a simple verb (e.g., add, update, remove, fix, ...) in the present tense is preferable.

Then fork the repository to your account and create a branch (e.g., `#31-plotting-feature`).

### Add commits

After you update something, commit your change with **a message which starts with the issue ID**.
Using a simple verb in the present tense is preferable.

```shell
git commit -m "#31 Add plot functions"
```

Please make sure that your code (1) is formatted by [Black] and (2) passes the tests (`tests/test_*.py`) run by [pytest].
They are necessary to pass the status checks when you create a pull request (see also [GitHub Actions](#github-actions)).

If you add a new feature, please also make sure that you prepare tests for it.
For example, if you add the plotting module (`deshima_sensitivity/plotting.py`), write the series of test functions in `tests/test_plotting.py`.

If you write a Python docstring, follow [the NumPy style] so that it is automatically converted to a part of API docs by [Sphinx].

### Open a Pull Request

When your code is ready, [create a pull request] (PR) to merge with the master branch.
Without special reasons, the title should be the same as that of the issue.
Please specify the issue ID in the comment form so that it is linked to the PR.
For example, writing `Closes #31.` at the beginning of the comment would be nice.

### Discuss and review your code

Your code is reviewed by at least one contributor and checked by the automatic status checks by [GitHub Actions].
After passing them, your code will be merged with the master branch.
That's it!
Thank you for your contribution!

## Development environment

We manage the development environment (i.e., Python and its dependencies) with [Poetry].
After cloning the repository you forked, you can setup the environment by the following command.

```shell
poetry install
```

If you use [VS Code] and [Docker Desktop], the following steps can create a standalone development environment (VS Code + Python + Poetry).

1. Install [VS Code] and [Docker Desktop], and launch them
1. Install the [Remote Containers] extension to VS Code
1. Clone this repository
1. Open the repository by VS Code
1. Choose `Reopen in Container` from the [Command Palette]

## GitHub Actions

### Tests workflow

We have a [tests workflow] for testing and formatting the codes, and docs' build.
It is used for status checks when a pull request is created.
If you would like to check them in local, the following commands are almost equivalent (the difference is that the workflow is run under multiple Python versions).

```shell
poetry run black --check docs tests deshima_sensitivity
poetry run docs/build
poetry run pytest
```

### PyPI workflow

We have a [PyPI workflow] for publishing the package to [PyPI].
When a [GitHub release] is created], the workflow is triggered and the package is automatically built and uploaded to PyPI.

### GitHub Pages workflow

We have a [GitHub Pages workflow] for publishing the HTML docs.
When a [GitHub release] is created, the workflow is triggered and the docs are automatically built and deployed to the gh-pages branch.


[Black]: https://black.readthedocs.io/en/stable/
[Command Palette]: https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette
[create a pull request]: https://github.com/deshima-dev/deshima-sensitivity/compare
[create an issue]: https://github.com/deshima-dev/deshima-sensitivity/issues/new
[Docker Desktop]: https://www.docker.com/products/docker-desktop
[GitHub Actions]: https://github.com/deshima-dev/deshima-sensitivity/actions
[Github flow]: https://guides.github.com/introduction/flow/
[GitHub Pages workflow]: https://github.com/deshima-dev/deshima-sensitivity/blob/master/.github/workflows/gh-pages.yml
[GitHub release]: https://github.com/deshima-dev/deshima-sensitivity/releases
[Poetry]: https://python-poetry.org/
[PyPI]: https://pypi.org/project/deshima-sensitivity/
[PyPI workflow]: https://github.com/deshima-dev/deshima-sensitivity/blob/master/.github/workflows/pypi.yml
[pytest]: https://docs.pytest.org/en/stable/
[Remote Containers]: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
[Sphinx]: https://www.sphinx-doc.org/en/master/
[tests workflow]: https://github.com/deshima-dev/deshima-sensitivity/blob/master/.github/workflows/test.yml
[the NumPy style]: https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html
[VS Code]: https://code.visualstudio.com/
