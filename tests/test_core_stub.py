"""Check that the `IM._core` type stub matches the compiled extension."""

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from IM import _core

STUB_PATH = Path(_core.__file__).parent / "_core.pyi"


def _stub_functions() -> dict[str, list[str]]:
    tree = ast.parse(STUB_PATH.read_text())
    return {
        node.name: [arg.arg for arg in node.args.posonlyargs + node.args.args]
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _core_functions() -> dict[str, Callable[..., Any]]:
    return {
        name: value
        for name, value in vars(_core).items()
        if callable(value) and not name.startswith("__")
    }


def test_stub_exists() -> None:
    assert STUB_PATH.is_file()


def test_stub_declares_every_core_function() -> None:
    assert set(_stub_functions()) == set(_core_functions())


@pytest.mark.parametrize("name", sorted(_core_functions()))
def test_stub_signature_matches_core(name: str) -> None:
    function = _core_functions()[name]
    runtime_parameters = list(inspect.signature(function).parameters)
    assert _stub_functions().get(name) == runtime_parameters
