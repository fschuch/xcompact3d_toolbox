import pytest
import xcompact3d_toolbox as x3d
import xcompact3d_toolbox.io
import numpy as np
import xarray as xr


@pytest.fixture
def filename_properties():
    return x3d.io.FilenameProperties()


@pytest.mark.parametrize("prefix", ["ux", "uy", "uz", "pp", "phi1"])
@pytest.mark.parametrize("counter", [0, 10, 100])
@pytest.mark.parametrize("separator", ["", "-"])
@pytest.mark.parametrize("extension", ["", ".bin"])
@pytest.mark.parametrize("number_of_digits", [3, 6])
def test_set_get_filename_from_bin(
    filename_properties, prefix, counter, separator, extension, number_of_digits
):
    filename_properties.set(
        separator=separator, file_extension=extension, number_of_digits=number_of_digits
    )
    assert (counter, prefix) == filename_properties.get_info_from_filename(
        filename_properties.get_filename_for_binary(prefix, counter)
    )


@pytest.fixture
def prm():
    return x3d.Parameters()


def test_read_write_field(prm, numpy_array):
    numpy_array = np.random.random((prm.nx, prm.ny, prm.nz)).astype(x3d.param["mytype"])
    filename = prm.filename_properties.get_filename_for_binary("test_field", 0)
    array_out = xr.DataArray(numpy_array, coords=prm.get_mesh, dims=prm.get_mesh.keys())
    x3d.io.write_field(array_out, prm, filename)
    array_in = x3d.io.read_field(prm, filename)
    xr.testing.assert_equal(array_out, array_in)
