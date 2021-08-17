import pytest
import xcompact3d_toolbox as x3d
import xcompact3d_toolbox.io


@pytest.mark.parametrize("prefix", ["ux", "uy", "uz", "pp", "phi1"])
@pytest.mark.parametrize("counter", [0, 10, 100])
@pytest.mark.parametrize("separator", ["", "-"])
@pytest.mark.parametrize("extension", ["", ".bin"])
@pytest.mark.parametrize("number_of_digits", [3, 6])
def test_set_get_filename_from_bin(
    prefix, counter, separator, extension, number_of_digits
):
    assert (counter, prefix) == x3d.io.get_info_from_filename(
        x3d.io.set_filename_for_bin(prefix, counter, separator, extension, number_of_digits),
        separator,
        extension,
        number_of_digits,
    )

