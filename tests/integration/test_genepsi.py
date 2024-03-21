import pathlib
import sys

import pytest

import xcompact3d_toolbox as x3d


@pytest.fixture(scope="session")
def set_up(tmp_path_factory):
    prm = x3d.Parameters(loadfile="tests/integration/data/input.i3d", raise_warning=True)
    tmp_path = tmp_path_factory.mktemp("data")
    prm.dataset.set(data_path=tmp_path.as_posix())
    epsi = x3d.sandbox.init_epsi(prm, dask=True)
    for key in epsi:
        epsi[key] = epsi[key].geo.cylinder(x=3.0, y=5.0)
    x3d.genepsi.gene_epsi_3d(epsi, prm)
    return tmp_path


@pytest.mark.skipif(sys.platform == "win32", reason="Work in progress to make it platform independent")
@pytest.mark.parametrize("file_ref", pathlib.Path("tests", "integration", "data", "geometry").glob("*.dat"))
def test_dat_files(file_ref, set_up):
    file = set_up / "geometry" / file_ref.name

    expected_content = file_ref.read_text()
    actual_content = file.read_text()

    assert expected_content == actual_content
