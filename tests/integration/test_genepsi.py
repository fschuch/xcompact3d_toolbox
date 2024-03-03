import filecmp
import glob
import os.path

import pytest

import xcompact3d_toolbox as x3d


@pytest.fixture(scope="session")
def set_up(tmpdir_factory):
    prm = x3d.Parameters(loadfile="tests/integration/data/input.i3d", raise_warning=True)
    tmp_path = tmpdir_factory.mktemp("data")
    prm.dataset.set(data_path=tmp_path.strpath)
    epsi = x3d.sandbox.init_epsi(prm, dask=True)
    for key in epsi:
        epsi[key] = epsi[key].geo.cylinder(x=3.0, y=5.0)
    x3d.genepsi.gene_epsi_3d(epsi, prm)
    yield tmp_path
    tmp_path.remove()


@pytest.mark.parametrize("file_ref", glob.glob(os.path.join("tests", "integration", "data", "geometry", "*")))
def test_dat_files(file_ref, set_up):
    file = os.path.join(set_up, "geometry", os.path.basename(file_ref))

    assert filecmp.cmp(file_ref, file)
