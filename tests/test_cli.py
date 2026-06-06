"""Offline tests for the CLI demo."""

from __future__ import annotations

import pytest

from pgvector_example import cli


def test_cli_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["vector search in postgres", "--top-k", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Query:" in out
    # Two ranked results should be printed.
    assert "1." in out
    assert "2." in out


def test_cli_default_query(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Query:" in out


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.top_k == 3
    assert args.dim == 64
    assert args.database_url is None
