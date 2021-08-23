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
def dataset():
    return x3d.io.Dataset(
        **dict(stack_velocity=True, stack_scalar=True)
    )


def test_write_read_field(dataset):
    coords = dataset._mesh.get()
    shape = [len(x) for x in coords.values()]
    numpy_array = np.random.random(size=shape).astype(x3d.param["mytype"])
    filename = dataset.filename_properties.get_filename_for_binary("ux", 0)
    array_out = xr.DataArray(numpy_array, coords=coords, dims=coords.keys())
    dataset.write_array(array_out, filename)
    array_in = dataset.load_array(dataset.data_path + filename, add_time=False)
    xr.testing.assert_equal(array_out, array_in)


@pytest.fixture
def snapshot(dataset):
    def numpy_array(**kwargs):
        shape = [len(i) for i in kwargs.values()]
        return np.random.random(shape).astype(x3d.param["mytype"])

    def xr_array(file_name, **kwargs):
        return xr.DataArray(
            numpy_array(**kwargs),
            coords=kwargs,
            dims=kwargs.keys(),
            attrs=dict(file_name=file_name),
        )

    coords = dict(dataset._mesh.get())
    coords["t"] = [dataset._time_step * k for k in range(len(dataset))]

    ds = xr.Dataset()

    ds["pp"] = xr_array("pp", **coords)
    ds["u"] = xr_array("u", i="x y z".split(), **coords)
    ds["phi"] = xr_array("phi", n=[n + 1 for n in range(3)], **coords)

    for _, val in ds.items():
        dataset.write_array(val)

    return ds


def test_dataset_getitem_int(dataset, snapshot):

    for k, time in enumerate(snapshot.t.values):
        ds = dataset[k]
        xr.testing.assert_equal(
            snapshot.sel(t=time, drop=True), ds.sel(t=time, drop=True)
        )


def test_dataset_getitem_str(dataset, snapshot):

    xr.testing.assert_equal(snapshot["pp"], dataset["pp"])


@pytest.mark.parametrize(
    "slice",
    [slice(None, None, None), slice(0, -1, 2), slice(-1, 0, -2), slice(0, 9, 3)],
)
def test_dataset_getitem_slice(dataset, snapshot, slice):

    xr.testing.assert_equal(snapshot.isel(t=slice), dataset[slice])


def test_dataset_iter(dataset, snapshot):

    for k, ds in enumerate(dataset):
        xr.testing.assert_equal(snapshot.sel(t=k, drop=True), ds.sel(t=k, drop=True))

@pytest.mark.parametrize("istret", [0, 1])
def test_dataset_write_xdmf(dataset, snapshot, istret):
    ds = snapshot
    dataset._mesh.y.istret = istret

    filename = f"snapshots_istret_{istret}.xdmf"

    dataset.write_xdmf(filename)
    assert filecmp.cmp(
        filename, f"./tests/unit/data/{filename}"
    )
