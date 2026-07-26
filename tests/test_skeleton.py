"""Project skeleton verification test."""

import foldgemma.data as data
import foldgemma.models as models

def test_packages_importable() -> None:
    """Verify all top-level project packages are importable."""
    assert data is not None
    assert models is not None
