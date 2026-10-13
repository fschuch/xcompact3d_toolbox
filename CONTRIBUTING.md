# Contributing to Xcompact3d-toolbox

Thank you for your interest in contributing! Contributions of all kinds are welcome: bug reports, feature requests, documentation improvements, and code.

## Reporting issues

Open an issue at <https://github.com/fschuch/xcompact3d_toolbox/issues>. Please include the package version, Python version, operating system, and a minimal example that reproduces the problem.

## Development setup

The development workflow is powered by [Hatch](https://hatch.pypa.io). After installing Hatch, fork and clone the repository, then run:

```bash
hatch run qa
```

This runs the pre-commit checks (ruff, mypy, codespell, and others) and the full test suite with coverage. Useful individual commands:

```bash
hatch run check          # pre-commit hooks on all files
hatch run test           # tests with coverage
hatch run docs:serve     # live preview of the documentation
```

## Pull requests

1. Create a branch from `main`.
1. Add or update tests for your change.
1. Make sure `hatch run qa` passes locally.
1. Open a pull request describing the motivation and the change.

Continuous integration runs the extended test suite across all supported Python versions.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms.

## Full guide

The complete contributor guide, including details on releases, versioning, and documentation, lives in the documentation:
<https://docs.fschuch.com/xcompact3d_toolbox/references/how-to-contribute.html>
