import filecmp

import numpy as np
import pytest
import xarray as xr
import xcompact3d_toolbox as x3d
import xcompact3d_toolbox.io


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


def test_write_read_field(prm):
    numpy_array = np.random.random((prm.nx, prm.ny, prm.nz)).astype(x3d.param["mytype"])
    filename = prm.filename_properties.get_filename_for_binary("test_field", 0)
    array_out = xr.DataArray(
        numpy_array, coords=prm.get_mesh(), dims=prm.get_mesh().keys()
    )
    x3d.io.write_field(array_out, prm, filename)
    array_in = x3d.io.read_field(prm, filename)
    xr.testing.assert_equal(array_out, array_in)


@pytest.fixture
def write_time_series(prm):
    prm.set(numscalar=3, ilast=11, ioutput=1)
    coords = prm.get_mesh()
    coords["t"] = [prm.dt * i for i in range(prm.ilast)]
    coords["n"] = range(prm.numscalar)

    numpy_array = np.random.random([len(i) for i in coords.values()]).astype(
        x3d.param["mytype"]
    )

    array_out = xr.DataArray(
        numpy_array, coords=coords, dims=coords.keys(), attrs=dict(file_name="phi")
    )
    x3d.io.write_field(array_out, prm)

    return prm, array_out


def test_write_read_temporal_series(write_time_series):

    prm, array_out = write_time_series

    for n in array_out.n.data:
        array_in = x3d.io.read_temporal_series(
            prm, filename_pattern=f"phi{n+1}-???.bin"
        )
        xr.testing.assert_equal(array_out.isel(n=n).drop_vars("n"), array_in)


@pytest.mark.parametrize("istret", [0, 1])
def test_write_xdmf(write_time_series, istret):
    prm, _ = write_time_series
    prm.set(istret=istret)
    x3d.io.write_xdmf(prm, filename_pattern="phi?-???.bin")
    assert filecmp.cmp(
        "snapshots.xdmf", f"./tests/unit/data/snapshots_istret_{istret}.xdmf"
    )

