from pathlib import Path

import numpy as np
import pytest
import xarray as xr

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


def _field(prm, name, **extra):
    coords = {**extra, **prm.get_mesh()}
    shape = tuple(len(v) for v in coords.values())
    return xr.DataArray(np.zeros(shape, dtype=np.float32), coords=coords, name=name)


class TestFromParametersFiles:
    def test_pattern_follows_filename_properties_with_exact_width(self, prm):
        convention = Xcompact3dConvention.from_parameters(prm)

        assert convention.pattern.template == "{name}-{step:03d}.bin"
        assert convention.pattern.exact_width is True

    def test_pattern_accepts_other_filename_properties(self, prm):
        prm.dataset.filename_properties.set(separator="", file_extension="", number_of_digits=4)
        default = Xcompact3dConvention.from_parameters(prm)
        explicit = Xcompact3dConvention.from_parameters(
            prm, filename_properties={"separator": ".", "file_extension": ".dat", "number_of_digits": 6}
        )

        assert default.pattern.template == "{name}{step:04d}"
        assert explicit.pattern.template == "{name}.{step:06d}.dat"

    def test_reader_decodes_snapshots_with_time_from_step(self, prm, monkeypatch):
        monkeypatch.setitem(x3d.param, "mytype", np.float32)
        specs = Xcompact3dConvention.from_parameters(prm).reader(Path("ux-002.bin"))

        assert specs.name == "ux"
        assert specs.dims == ("x", "y", "z", "t")
        assert specs.order == "F"
        assert specs.coords["t"].dtype == np.float32
        np.testing.assert_allclose(specs.coords["t"], [2 * prm.dt * prm.ioutput])

    def test_snapshot_step_selects_the_output_frequency(self, prm):
        prm.iprocessing = 50
        specs = Xcompact3dConvention.from_parameters(prm, snapshot_step="iprocessing").reader(Path("ux-001.bin"))

        np.testing.assert_allclose(specs.coords["t"], [prm.dt * 50])

    def test_reader_decodes_static_files_in_the_root(self, prm):
        specs = Xcompact3dConvention.from_parameters(prm).reader(Path("epsilon.bin"))

        assert specs.name == "epsilon"
        assert specs.dims == ("x", "y", "z")

    def test_reader_rejects_unrelated_files(self, prm):
        convention = Xcompact3dConvention.from_parameters(prm)

        for name in ("snapshots.xdmf", "notes.txt", "ux-001.bin.bak"):
            with pytest.raises(ValueError, match="No convention accepts"):
                convention.reader(Path(name))

    def test_no_root_static_member_without_extension_unless_named(self, prm):
        prm.dataset.filename_properties.set(separator="", file_extension="", number_of_digits=4)
        bare = Xcompact3dConvention.from_parameters(prm)
        named = Xcompact3dConvention.from_parameters(prm, static_names=("epsilon",))

        assert bare.reader(Path("ux0001")).name == "ux"
        with pytest.raises(ValueError, match="No convention accepts"):
            bare.reader(Path("README"))
        assert named.reader(Path("epsilon")).dims == ("x", "y", "z")
        with pytest.raises(ValueError, match="No convention accepts"):
            named.reader(Path("README"))

    def test_stacks_declare_velocity_and_scalars(self, prm):
        prm.dataset.filename_properties.set(scalar_num_of_digits=2)
        stacks = {s.dim: s for s in Xcompact3dConvention.from_parameters(prm).stacks}

        assert stacks["i"].template == "{name}{i}"
        assert stacks["i"].values == ("x", "y", "z")
        assert stacks["n"].template == "{name}{n:02d}"
        assert stacks["n"].names == ("phi",)
        assert stacks["n"].attrs["name"] == "Scalar fraction"

    def test_writer_splits_stacked_arrays_and_steps(self, prm):
        convention = Xcompact3dConvention.from_parameters(prm)
        t = np.arange(2) * prm.dt * prm.ioutput
        u = _field(prm, "u", i=["x", "y", "z"], t=t)
        phi = _field(prm, "phi", n=[1, 2], t=t)

        assert [s.filename for s in convention.writer(u)] == [f"u{i}-{k:03d}.bin" for i in "xyz" for k in range(2)]
        assert [s.filename for s in convention.writer(phi)] == [f"phi{n}-{k:03d}.bin" for n in (1, 2) for k in range(2)]
        assert all(s.order == "F" and s.sub_array.dims == ("x", "y", "z") for s in convention.writer(u))

    def test_writer_takes_the_name_from_the_file_name_attribute(self, prm):
        convention = Xcompact3dConvention.from_parameters(prm)
        vort = _field(prm, "vorticity", t=[0.0]).assign_attrs(file_name="w3")
        epsi = _field(prm, "epsi").assign_attrs(file_name="geometry/epsilon")

        assert [s.filename for s in convention.writer(vort)] == ["w3-000.bin"]
        assert [s.filename for s in convention.writer(epsi)] == ["geometry/epsilon.bin"]
