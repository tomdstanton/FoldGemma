"""Project skeleton verification test."""

import foldgemma.data as data
import foldgemma.inference as inference
import foldgemma.train as train


def test_packages_importable() -> None:
    """Verify all top-level project packages are importable."""
    assert data is not None
    assert train is not None
    assert inference is not None
