import numpy as np
import pytest

import xcompact3d_toolbox as x3d
from xcompact3d_toolbox.binfile import Xcompact3dConvention


@pytest.fixture
def prm():
    return x3d.Parameters(nx=9, ny=9, nz=9, numscalar=2, ilast=100, ioutput=25, dt=0.01)


class TestFromParametersLayout:
    def test_layout_keeps_natural_dims_in_fortran_order(self, prm):
        convention = Xcompact3dConvention.from_parameters(prm)

        assert convention.layout.dims == ("x", "y", "z")
        assert convention.layout.shape == (9, 9, 9)
        assert convention.layout.order == "F"
        np.testing.assert_array_equal(convention.layout.coords["x"], prm.get_mesh()["x"])

    def test_dtype_defaults_to_param_mytype(self, prm, monkeypatch):
        monkeypatch.setitem(x3d.param, "mytype", np.float32)

        assert np.dtype(Xcompact3dConvention.from_parameters(prm).layout.dtype) == np.float32
        assert np.dtype(Xcompact3dConvention.from_parameters(prm, dtype=np.float64).layout.dtype) == np.float64

    @pytest.mark.parametrize("drop_coords", ["x", "y", "z"])
    def test_drop_coords_gives_a_plane_layout(self, prm, drop_coords):
        convention = Xcompact3dConvention.from_parameters(prm, drop_coords=drop_coords)

        assert convention.layout.dims == tuple(d for d in "xyz" if d != drop_coords)

    def test_layout_carries_coordinate_labels(self, prm):
        attrs = Xcompact3dConvention.from_parameters(prm).layout.coord_attrs

        assert attrs["x"]["name"] == "Streamwise coordinate"
        assert attrs["t"]["name"] == "Time"
