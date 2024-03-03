import pytest

import xcompact3d_toolbox


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["x3d"] = xcompact3d_toolbox
