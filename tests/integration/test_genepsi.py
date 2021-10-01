import glob
import os.path
import filecmp
import pytest
import xcompact3d_toolbox as x3d
import xcompact3d_toolbox.sandbox
import xcompact3d_toolbox.genepsi

@pytest.fixture
def set_up():
    prm = x3d.Parameters(loadfile="tests/integration/data/input.i3d", raise_warning = True)
    epsi = x3d.sandbox.init_epsi(prm)
    for key in epsi.keys():
        epsi[key] = epsi[key].geo.cylinder(x=3.0, y=5.0)
    x3d.genepsi.gene_epsi_3D(epsi, prm)

@pytest.mark.parametrize(
    "file_ref", glob.glob(os.path.join("tests", "integration", "data", "geometry", "*"))
)
def test_dat_files(file_ref, set_up):
    file = os.path.join("data", "geometry", os.path.basename(file_ref))

    assert filecmp.cmp(file_ref, file)
