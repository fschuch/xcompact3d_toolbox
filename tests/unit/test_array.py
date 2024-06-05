import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def array():
    dims = "x y z n t".split()
    return xr.DataArray(1.0, dims="x y z n t".split(), coords={dim: np.arange(11.0) for dim in dims})


@pytest.fixture
def dataset(array):
    return xr.Dataset({key: array for key in "ux uy uz pp phi".split()})


@pytest.mark.parametrize("dims", ["x", "y", "z n t".split()])
def test_data_pencil_decomp(array, dataset, dims):
    array.x3d.pencil_decomp(*dims)
    dataset.x3d.pencil_decomp(*dims)


@pytest.mark.parametrize("dims", ["x", "y", "t"])
def test_data_cumtrapz(array, dataset, dims):
    array.x3d.cumtrapz(*dims)
    dataset.x3d.cumtrapz(*dims)


@pytest.mark.parametrize("dims", ["x", "y", "t"])
def test_data_cumulative_trapezoid(array, dataset, dims):
    array.x3d.cumulative_trapezoid(*dims)
    dataset.x3d.cumulative_trapezoid(*dims)


@pytest.mark.parametrize("dims", ["x", "y", "t"])
def test_data_simps(array, dataset, dims):
    array.x3d.simps(*dims)
    dataset.x3d.simps(*dims)


@pytest.mark.parametrize("dims", ["x", "y", "t"])
def test_data_simpson(array, dataset, dims):
    array.x3d.simpson(*dims)
    dataset.x3d.simpson(*dims)


def test_data_simps__invalid_dims(array, dataset):
    with pytest.raises(ValueError, match='Invalid value for "args", it should be a valid dimension'):
        array.x3d.simpson("not-a-dim")
    with pytest.raises(ValueError, match='Invalid value for "args", it should be a valid dimension'):
        dataset.x3d.simpson("not-a-dim")


@pytest.mark.parametrize("dims", ["x", "y", "z"])
def test_data_first_derivative(array, dims):
    array.x3d.first_derivative(*dims)


@pytest.mark.parametrize("dims", ["x", "y", "z"])
def test_data_second_derivative(array, dims):
    array.x3d.second_derivative(*dims)


def test_data_pencil_decomposition__invalid_dims(array, dataset):
    with pytest.raises(ValueError, match='Invalid value for "args", it should be a valid dimension'):
        array.x3d.pencil_decomp("not-a-dim")
    with pytest.raises(ValueError, match='Invalid value for "args", it should be a valid dimension'):
        dataset.x3d.pencil_decomp("not-a-dim")
