import pytest


def pytest_collection_modifyitems(items):
    """Auto-mark every test in tests/integration/ as slow."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
