import pytest
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
        filename_properties.get_filename_for_bin(prefix, counter)
    )

