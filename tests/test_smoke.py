"""Smoke test: every afdetect module imports."""
import importlib

import pytest

MODULES = ["afdetect", "afdetect.data", "afdetect.features", "afdetect.splits", "afdetect.extract",
           "afdetect.models", "afdetect.metrics", "afdetect.train", "afdetect.evaluate", "afdetect.afdb"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    """Each module can be imported without errors."""
    assert importlib.import_module(name) is not None
