import pytest

import xcompact3d_toolbox


@pytest.fixture(autouse=True)
def add_x3d(doctest_namespace):
    doctest_namespace["x3d"] = xcompact3d_toolbox
