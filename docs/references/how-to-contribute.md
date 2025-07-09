# How to Contribute

Thank you for your interest in contributing to the **Wizard Template** project! Contributions are welcome and greatly appreciated. This guide will help you get started.

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on <https://github.com/fschuch/wizard-template/issues>. Provide as much detail as possible to help us address the issue.

## Setting up the Project

### Prerequisites

The development workflow is powered by [Hatch](https://hatch.pypa.io), which manages Python installations, virtual environments, dependencies, besides builds,
and deploys the project to [PyPI](https://pypi.org).
See [Why Hatch?](https://hatch.pypa.io/latest/why/) for more details.
Refer to [Install Hatch](https://hatch.pypa.io/latest/install/) for instructions on how to install it on your operating system.

Ensure you have Python 3.9 or later installed (this can also be done by hatch, take a look at `hatch python --help`).

````{tip}
Optionally, configure Hatch to keep virtual environments within the project folder:
```bash
hatch config set dirs.env.virtual .venv
```
````

### Clone the repository

Fork the repository and clone it to your local machine:

```bash
git clone https://github.com/<your-username>/wizard-template.git
cd wizard-template
```

Run quality assurance checks to ensure you have green lights on your local copy:

```bash
hatch run qa
```

```{tip}
You can run `hatch env show` at any time to see the available environments, their features and scripts.
```

## Code Quality Standards

To ensure quality standards on the codebase, several tools are configured in the project:

- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
- [ruff](https://github.com/astral-sh/ruff) as the linter and code formatter
- [codespell](https://github.com/codespell-project/codespell) to check spelling
- [pytest](https://docs.pytest.org/en/7.4.x/) as the test engine
- [zizmor](https://github.com/woodruffw/zizmor) for static analysis tool for GitHub Actions workflows

[pre-commit](https://pre-commit.com) manages and runs them on isolated environments.
These tools are not declared as development dependencies on the project to avoid duplication.
Even though it performs checks on the changes for every commit when installed (`hatch run pre-commit-install`),
it is a good practice to run the checks on the whole codebase occasionally (when a new hook is added or on Pull
Requests). You can do so by running `hatch run check <hook-id>`, for instance `hatch run check nbstripout`.
Some of them are available as scripts as a syntax sugar, like `hatch run lint`,
`hatch run format`, or `hatch run type`. They check the whole codebase using ruff, ruff-format, and mypy, respectively.

The file [project.toml](https://github.com/fschuch/wizard-template/blob/main/pyproject.toml) includes configuration for some of the tools, so they can be consumed by your IDE as well.
The file [.pre-commit-config.yaml](https://github.com/fschuch/wizard-template/blob/main/.pre-commit-config.yaml) includes the configuration for the pre-commit hooks.

The [pytest](https://docs.pytest.org/en/stable/index.html) test suite can be run from the default environment with `hatch run test`
or `hatch run test-no-cov` (the latter without coverage check).

Code examples on docstrings and documentation are tested by the `doctest` module (configured on the file [pyproject.toml](https://github.com/fschuch/wizard-template/blob/main/pyproject.toml)).
It is integrated with pytest, so the previous test commands will also run the doctests.

To run all the quality checks, you can use the command `hatch run qa`.

To step up in the game, an extended test environment and the command `hatch run test:extended` are available to
verify the package on different Python versions and under different conditions thanks to the pytest plugins:

- `pytest-randomly` that randomizes the test order;
- `pytest-rerunfailures` that re-runs tests to eliminate intermittent failures;
- `pytest-xdist` that parallelizes the test suite and reduce runtime, to help the previous points that increase the workload;
- The file [pyproject.toml](https://github.com/fschuch/wizard-template/blob/main/pyproject.toml) includes configuration for them.

## Continuous Integration

- The workflow [ci.yaml](https://github.com/fschuch/wizard-template/blob/main/.github/workflows/ci.yaml) performs the verifications on every push and pull request, and deploys the package if running from a valid tag.
- The workflow [update-pre-commits.yaml](https://github.com/fschuch/wizard-template/blob/main/.github/workflows/update-pre-commits.yaml) is scheduled to run weekly to ensure the pre-commit hooks are up-to-date.
- Dependabot is enabled to keep the dependencies up-to-date ([dependabot.yml](https://github.com/fschuch/wizard-template/blob/main/.github/dependabot.yml)).

## Development Workflow

1. **Create a Branch**: Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/<your-feature-name>
   ```

1. **Make Changes**: Implement your changes. Follow the coding standards enforced by the project.

1. **Documentation**: Update or add docstrings and documentation as needed.

1. **Run Quality Checks**: Before committing, ensure your changes pass all checks:

   ```bash
   hatch run qa
   ```

   Optionally, you can install the pre-commit hooks to run these checks automatically before each commit:

   ```bash
   hatch run pre-commit-install
   ```

1. **Commit Changes**: Commit your changes with a meaningful message ([Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) are advised):

   ```bash
   git add .
   git commit -m "change: Describe your changes"
   ```

1. **Push and Open a Pull Request**: Push your branch and open a pull request:

   ```bash
   git push origin feature/<your-feature-name>
   ```

   - Ensure your pull request includes a clear description of the changes.
   - Reference any related issues.

### Pull Request Labels and Changelog

We use GitHub's automated release notes to generate the changelog. Labels on pull requests are critical for organizing the changelog into meaningful categories. Below are the labels we use and their purposes:

- **Security**: For security-related changes (`security`).
- **Removed**: For features or functionality that has been removed (`removed`).
- **Deprecated**: For features that are deprecated but still available (`deprecated`).
- **Added**: For new features or functionality (`added`).
- **Changed**: For updates or modifications to existing features (`changed`).
- **Fixed**: For bug fixes (`fixed`).
- **Documentation**: For changes to documentation (`docs`).
- **Internals**: For internal changes like refactoring, dependency updates, or chores (`chore`, `refactor`, `dependencies`).
- **Other Changes**: For tags that do not fit into the above categories.

A good pull request title is essential for generating a clear and concise changelog. The title should:

1. Be descriptive and summarize the change.
1. Use an imperative tone (e.g., "Add feature X" instead of "Added feature X").
1. Avoid vague terms like "Fix issue" or "Update code."

For example:

- ✅ "Add support for custom configurations"
- ✅ "Fix crash when loading large datasets"
- ❌ "Bug fix"
- ❌ "Miscellaneous updates"

### Creating a New Release

To create a new release, follow these steps:

- Ensure all pull requests for the release are labeled and merged.

- Create a new release from GitHub: <https://github.com/fschuch/wizard-template/releases>

  - Based on previous versions, choose the next version number according to the [EffVer](https://jacobtomlinson.dev/effver/) scheme. The tag matching pattern is set to `v*.*.*`, for instance, `v1.2.3`, , `v2.3.4a0`, , `v2.3.4b0`, `v2.3.4rc0`.
  - Choose to create a new tag on publish based on version from previous step.
  - Click on Generate Release Notes. Modify the release notes as needed.
  - Double-check if the new version number is appropriate for the given set of changes.
  - Verify if it needs to be set to `Pre-release` (i.e., its alpha, beta, or release candidate).
  - Ensure to select the `Save Draft` option to have some extra time to double-check all the points above, or to ask for feedback from collaborators.
  - Click on `Publish Release` when ready.

- The process can also be triggered from the command line using [gh command line tool](https://cli.github.com):

  - Check current version:

  ```bash
  hatch version
  ```

  - Follow previous instruction on how to set the new version number. Start a new release:

  ```bash
  gh release create <new_version> --generate-notes --draft
  ```

- The CI workflow will automatically test, build and publish the package to PyPI if the tag matches the pattern `v*.*.*`.

## Documentation

The project uses [Jupyter Books](https://jupyterbook.org/en/stable/intro.html)
to provide a promising approach for interactive tutorials.
The documentation source is on the `docs` folder and can be
served locally with `hatch run docs:serve`, it will be available on `http://127.0.0.1:8000`.
The documentation is also built automatically on the deployment workflow
[docs.yaml](https://github.com/fschuch/wizard-template/blob/main/.github/workflows/docs.yaml).

Modules and functions docstrings are used to generate the documentation thanks to the [sphinx-autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and [sphinx-napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) packages.

Useful references:

- <https://jupyterbook.org/en/stable/intro.html>
- <https://sphinx-book-theme.readthedocs.io/en/stable/index.html>
- <https://www.sphinx-doc.org/en/master/>

## Miscellaneous

### VSCode Configuration

The project includes a `.vscode` folder with a [extensions.json](https://github.com/fschuch/wizard-template/blob/main/.vscode/extensions.json) file that suggests the extensions to be installed on VSCode. It allows test, debug, auto-format, lint, and a few other functionalities to work directly on your IDE. It also includes a [settings.json](https://github.com/fschuch/wizard-template/blob/main/.vscode/settings.json) file that configures the Python extension to use the virtual environment created by Hatch. Remember to set hatch to use the virtual environment within the project folder `hatch config set dirs.env.virtual .venv`.
