import filecmp
from textwrap import dedent

import numpy as np
import pytest
import xarray as xr

import xcompact3d_toolbox as x3d
from xcompact3d_toolbox.param import COORDS


@pytest.fixture
def filename_properties():
    return x3d.io.FilenameProperties()


@pytest.fixture(scope="session")
def random_generator():
    return np.random.Generator(np.random.PCG64(1234))


@pytest.mark.parametrize("prefix", ["ux", "uy", "uz", "pp", "phi1"])
@pytest.mark.parametrize("counter", [0, 10, 100])
@pytest.mark.parametrize("separator", ["", "-"])
@pytest.mark.parametrize("extension", ["", ".bin"])
@pytest.mark.parametrize("number_of_digits", [3, 6])
def test_set_get_filename_from_bin(filename_properties, prefix, counter, separator, extension, number_of_digits):
    filename_properties.set(separator=separator, file_extension=extension, number_of_digits=number_of_digits)
    filename = filename_properties.get_filename_for_binary(prefix, counter)
    assert (counter, prefix) == filename_properties.get_info_from_filename(filename)
    assert counter == filename_properties.get_num_from_filename(filename)
    assert prefix == filename_properties.get_name_from_filename(filename)


def test_file_name_properties_set__fail_invalid_name(filename_properties):
    with pytest.raises(KeyError, match=".* is not a valid argument for FilenameProperties"):
        filename_properties.set(invalid_name=None)


@pytest.fixture
def dataset():
    return x3d.io.Dataset(stack_velocity=True, stack_scalar=True)


def test_write_read_field(dataset, random_generator):
    coords = dataset._mesh.get()  # noqa: SLF001
    shape = [len(x) for x in coords.values()]
    numpy_array = random_generator.random(size=shape).astype(x3d.param["mytype"])
    filename = dataset.filename_properties.get_filename_for_binary("ux", 0)
    array_out = xr.DataArray(numpy_array, coords=coords, dims=coords.keys())
    dataset.write(array_out, filename)
    array_in = dataset.load_array(dataset.data_path + filename, add_time=False)
    xr.testing.assert_equal(array_out, array_in)


@pytest.fixture
def snapshot(dataset, random_generator):
    def numpy_array(**kwargs):
        shape = [len(i) for i in kwargs.values()]
        return random_generator.random(shape).astype(x3d.param["mytype"])

    def xr_array(file_name, **kwargs):
        return xr.DataArray(
            numpy_array(**kwargs),
            coords=kwargs,
            dims=kwargs.keys(),
            attrs={"file_name": file_name},
        )

    coords = dict(dataset._mesh.get())  # noqa: SLF001
    coords["t"] = [dataset._time_step * k for k in range(len(dataset))]  # noqa: SLF001

    ds = xr.Dataset()

    ds["pp"] = xr_array("pp", **coords)
    ds["u"] = xr_array("u", i=COORDS, **coords)
    ds["phi"] = xr_array("phi", n=[n + 1 for n in range(3)], **coords)

    dataset.write(ds)

    return ds


def test_dataset_getitem_int(dataset, snapshot):
    for k, time in enumerate(snapshot.t.values):
        ds = dataset[k]
        xr.testing.assert_equal(snapshot.sel(t=time, drop=True), ds.sel(t=time, drop=True))


def test_dataset_getitem_str(dataset, snapshot):
    xr.testing.assert_equal(snapshot["pp"], dataset["pp"])


@pytest.mark.parametrize(
    "slice_value",
    [slice(None, None, None), slice(0, -1, 2), slice(-1, 0, -2), slice(0, 9, 3)],
)
def test_dataset_getitem_slice(dataset, snapshot, slice_value):
    xr.testing.assert_equal(snapshot.isel(t=slice_value), dataset[slice_value])


@pytest.mark.parametrize("slice_value", [None, 3.5])
def test_dataset_getitem_slice__type_error(dataset, slice_value):
    with pytest.raises(TypeError, match="Dataset indices should be integers, string or slices"):
        dataset[slice_value]


def test_dataset_iter(dataset, snapshot):
    for k, ds in enumerate(dataset):
        xr.testing.assert_equal(snapshot.sel(t=k, drop=True), ds.sel(t=k, drop=True))


@pytest.mark.parametrize("istret", [0, 1])
def test_dataset_write_xdmf(dataset, snapshot, istret, tmp_path):  # noqa: ARG001
    dataset._mesh.y.istret = istret  # noqa: SLF001

    filename = f"snapshots_istret_{istret}.xdmf"
    actual_file = tmp_path / filename

    dataset.write_xdmf(actual_file.as_posix(), float_precision=6)
    assert filecmp.cmp(actual_file.as_posix(), f"./tests/unit/data/{filename}")


def test_prm_to_dict(tmp_path):
    prm_content = dedent(
        """
        # Comments
        32 # nx # some explanation
        64 # ny # some explanation
        128 # nz # some explanation
        'bar' # foo
        .false. # flag
        2.5 # float
        1 # my_list(1)
        2 # my_list(2)
        3 # my_list(3)
        # More comments
        """
    )
    prm_file = tmp_path / "test.prm"
    prm_file.write_text(prm_content)

    expected = {
        "nx": 32,
        "ny": 64,
        "nz": 128,
        "foo": "bar",
        "flag": False,
        "float": 2.5,
        "my_list": [1, 2, 3],
    }
    actual = x3d.io.prm_to_dict(prm_file)

    assert expected == actual
