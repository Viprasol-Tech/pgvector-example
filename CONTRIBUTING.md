# Contributing to pgvector Example

Thanks for your interest in improving this project, maintained by
[Viprasol Tech Private Limited](https://viprasol.com)!

## Getting started

```bash
git clone https://github.com/Viprasol-Tech/pgvector-example.git
cd pgvector-example
pip install -e ".[dev]"
```

## Before you open a PR

Please make sure all checks pass locally — these are the same ones CI runs:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
```

The test suite runs fully offline, so you do **not** need a database to develop.

## Guidelines

- Keep changes focused and well-described.
- Add or update tests for any behaviour change.
- Public modules and functions should have docstrings.
- Follow the existing style; `ruff format` is the source of truth.
- Be respectful — see our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs & requesting features

Use the GitHub [issue templates](.github/ISSUE_TEMPLATE). For security reports,
see [SECURITY.md](SECURITY.md).
