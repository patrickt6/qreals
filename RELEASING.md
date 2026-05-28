# Releasing qreals

The release pipeline is already wired. Distributions build on a tag push and
publish to PyPI through Trusted Publishing (OpenID Connect), so no API token is
stored anywhere. This file records what is in place and the few manual steps
that remain before `pip install qreals` works.

## What is already set up

- `pyproject.toml` builds an sdist and a wheel with hatchling. The version is
  `0.1.0`, matching `src/qreals/__init__.py` (`__version__`) and `CITATION.cff`.
- `.github/workflows/release.yml` builds the distributions, runs `twine check`,
  and publishes:
  - to **TestPyPI** on a manual run (`workflow_dispatch`), for a dry run in a
    clean environment, and
  - to **PyPI** on a tag matching `v*`.
- `.github/workflows/ci.yml` runs the tests, `ruff`, and `mypy --strict` on
  every push and pull request.
- `CHANGELOG.md` describes the 0.1.0 release.

The release job authenticates to PyPI over OIDC, so PyPI has to be told once to
trust this workflow. That one-time setup is the only thing between the current
state and a live package.

## Remaining manual steps

These happen in a browser and on the command line; nothing in the pipeline
needs changing.

### 1. Add the Trusted Publisher on TestPyPI (optional dry run)

On https://test.pypi.org, sign in, open the account Publishing settings, and add
a pending publisher with these exact values:

| Field             | Value         |
|-------------------|---------------|
| PyPI project name | `qreals`      |
| Owner             | `patrickt6`   |
| Repository name   | `qreals`      |
| Workflow name     | `release.yml` |
| Environment name  | `testpypi`    |

Then run the Release workflow by hand (the Actions tab, "Run workflow") to
publish to TestPyPI, and confirm the build installs cleanly:

```bash
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ qreals
```

### 2. Add the Trusted Publisher on PyPI

On https://pypi.org, add the same pending publisher, with the environment name
`pypi`:

| Field             | Value         |
|-------------------|---------------|
| PyPI project name | `qreals`      |
| Owner             | `patrickt6`   |
| Repository name   | `qreals`      |
| Workflow name     | `release.yml` |
| Environment name  | `pypi`        |

### 3. Tag the release

With both publishers in place, tag `v0.1.0` and push the tag. The tag push
triggers the PyPI publish job:

```bash
git tag v0.1.0
git push origin v0.1.0
```

When the workflow finishes, `pip install qreals` works for everyone.

## Until the package is live

Installing from git works today and needs none of the above:

```bash
pip install "git+https://github.com/patrickt6/qreals.git"
```

The "Open in Colab" link and the Binder badge in the README both use this git
install, so they work now and keep working after the PyPI release.

## Cutting later releases

For a future release, bump the version in `pyproject.toml`,
`src/qreals/__init__.py`, and `CITATION.cff`, move the `Unreleased` notes in
`CHANGELOG.md` under the new version, then tag `vX.Y.Z` and push. The Trusted
Publisher setup above is one-time and does not repeat.
