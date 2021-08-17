import xarray as xr
import numpy as np
from xcompact3d_toolbox.array import X3dDataset, X3dDataArray
import pytest


@pytest.fixture
def array():
    dims = "x y z n t".split()
    return xr.DataArray(
        1.0, dims="x y z n t".split(), coords={dim: np.arange(11.0) for dim in dims}
    )


@pytest.fixture
def dataset(array):
    return xr.Dataset({key: array for key in "ux uy uz pp phi".split()})


@pytest.mark.parametrize("dims", ["x", "y", "z n t".split()])
def test_array_pencil_decomp(array, dims):
    array.x3d.pencil_decomp(*dims)

@pytest.mark.parametrize("dims", ["x", "y", "z n t".split()])
def test_dataset_pencil_decomp(dataset, dims):
    dataset.x3d.pencil_decomp(*dims)