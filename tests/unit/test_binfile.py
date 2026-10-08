import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from xarray_binfile.conventions import Layout, StaticFiles

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


@pytest.fixture
def case(prm, tmp_path, monkeypatch):
    """Snapshots written by the on-demand loader, plus a static geometry file."""
    monkeypatch.setitem(x3d.param, "mytype", np.float32)
    prm.dataset.set(data_path=tmp_path.as_posix() + "/", stack_velocity=True, stack_scalar=True)
    rng = np.random.default_rng(0)
    t = np.arange(len(prm.dataset)) * prm.dt * prm.ioutput

    def field(file_name, **extra):
        array = _field(prm, file_name, **extra, t=t).assign_attrs(file_name=file_name)
        array.values[...] = rng.random(array.shape, dtype=np.float32)
        return array

    snapshots = xr.Dataset({"u": field("u", i=["x", "y", "z"]), "phi": field("phi", n=[1, 2]), "pp": field("pp")})
    epsi = _field(prm, "epsi").assign_attrs(file_name="geometry/epsilon")
    epsi.values[...] = rng.random(epsi.shape, dtype=np.float32)
    (tmp_path / "geometry").mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prm.dataset.write(snapshots)
        prm.dataset.write(epsi)
    for extra in ("snapshots.xdmf", "notes.txt", "input.i3d", ".DS_Store"):
        (tmp_path / extra).write_text("not a field")
    return prm, tmp_path, snapshots, epsi


class TestOpen:
    def test_files_lists_only_the_binary_fields(self, case):
        prm, root, *_ = case
        files = Xcompact3dConvention.from_parameters(prm).files(root)

        assert all(p.suffix == ".bin" and p.parent == root for p in files)
        assert len(files) == 6 * 5

    def test_open_matches_the_on_demand_loader(self, case):
        prm, root, snapshots, _ = case
        lazy = Xcompact3dConvention.from_parameters(prm).open(root)
        eager = prm.dataset[:]

        assert sorted(lazy.data_vars) == ["phi", "pp", "u"]
        for name in ("u", "phi", "pp"):
            xr.testing.assert_allclose(lazy[name].transpose(*eager[name].dims).load(), eager[name])
        assert lazy["u"].dims == ("i", "x", "y", "z", "t")
        assert lazy["t"].dtype == np.float32
        assert lazy["x"].attrs["name"] == "Streamwise coordinate"
        assert lazy["i"].attrs["name"] == "Velocity component"
        assert lazy["n"].values.tolist() == [1, 2]

    def test_open_is_lazy_with_one_file_per_chunk_by_default(self, case):
        prm, root, *_ = case
        lazy = Xcompact3dConvention.from_parameters(prm).open(root)

        assert lazy["u"].chunks is not None
        assert lazy["u"].chunks[-1] == (1,) * 5
        spatial = Xcompact3dConvention.from_parameters(prm).open(root, chunks={"x": 3})
        assert spatial["pp"].chunks[0] == (3, 3, 3)

    def test_open_without_stacking_shows_the_files(self, case):
        prm, root, *_ = case
        raw = Xcompact3dConvention.from_parameters(prm).open(root, stack=False)

        assert sorted(raw.data_vars) == ["phi1", "phi2", "pp", "ux", "uy", "uz"]

    def test_stack_rebuilds_arrays_from_any_dataset(self, case):
        prm, root, *_ = case
        convention = Xcompact3dConvention.from_parameters(prm)
        raw = xr.open_mfdataset(convention.files(root), engine="binfile", read_specs_getter=convention.reader)

        assert sorted(convention.stack(raw).data_vars) == ["phi", "pp", "u"]

    @pytest.mark.parametrize(
        ("variables", "expected"),
        [
            (["pp"], ["pp"]),
            (["ux", "uz"], ["u"]),
            (["u"], ["u"]),
            (["phi", "pp"], ["phi", "pp"]),
        ],
    )
    def test_open_selects_variables_by_disk_or_stacked_name(self, case, variables, expected):
        prm, root, *_ = case
        lazy = Xcompact3dConvention.from_parameters(prm).open(root, variables=variables)

        assert sorted(lazy.data_vars) == expected

    def test_open_with_wrong_dtype_fails_early(self, case):
        prm, root, *_ = case

        with pytest.raises(ValueError, match="Size mismatch"):
            Xcompact3dConvention.from_parameters(prm, dtype=np.float64).open(root)


class TestFolders:
    def test_folder_spec_builds_a_sub_convention_from_the_same_parameters(self, case):
        prm, root, _, epsi = case
        convention = Xcompact3dConvention.from_parameters(prm, folders={"geometry": {"static": True}})

        lazy = convention.open(root)

        assert sorted(lazy.data_vars) == ["epsilon", "phi", "pp", "u"]
        np.testing.assert_array_equal(lazy["epsilon"].transpose(*epsi.dims), epsi)

    def test_folder_spec_accepts_from_parameters_overrides(self, case):
        prm, root, *_ = case
        (root / "xy_planes").mkdir()
        plane = _field(prm, "ux", t=[0.0]).isel(z=0, drop=True)
        convention = Xcompact3dConvention.from_parameters(prm, folders={"xy_planes": {"drop_coords": "z"}})
        plane.binary_engine.to_file(convention.writer, root)

        specs = convention.reader(root / "xy_planes" / "ux-000.bin")

        assert specs.dims == ("x", "y", "t")
        assert (root / "xy_planes" / "ux-000.bin").exists()

    def test_folder_accepts_a_ready_convention(self, case):
        prm, root, *_ = case
        custom = StaticFiles(Layout({"x": np.arange(3)}, dtype="<f8"), pattern="{name}.dat")
        convention = Xcompact3dConvention.from_parameters(prm, folders={"probes": custom})

        assert convention.reader(root / "probes" / "p.dat").dims == ("x",)

    def test_writer_sends_folder_prefixed_names_to_the_registered_folder(self, case):
        prm, root, *_ = case
        convention = Xcompact3dConvention.from_parameters(prm, folders={"geometry": {"static": True}})
        epsi = _field(prm, "epsi").assign_attrs(file_name="geometry/epsilon")

        assert [s.filename for s in convention.writer(epsi)] == ["geometry/epsilon.bin"]


class TestParametersOpenDataset:
    def test_open_dataset_defaults_to_the_loader_data_path(self, case):
        prm, root, *_ = case

        lazy = prm.open_dataset()
        eager = prm.dataset[:]

        assert sorted(lazy.data_vars) == ["phi", "pp", "u"]
        xr.testing.assert_allclose(lazy["pp"].transpose(*eager["pp"].dims).load(), eager["pp"])

    def test_open_dataset_forwards_convention_and_open_options(self, case):
        prm, root, *_ = case

        lazy = prm.open_dataset(
            root,
            folders={"geometry": {"static": True}},
            variables=["u", "epsilon"],
            stack=False,
            chunks={"x": 3},
        )

        assert sorted(lazy.data_vars) == ["epsilon", "ux", "uy", "uz"]
        assert lazy["ux"].chunks[0] == (3, 3, 3)

    def test_convention_is_exported_at_top_level(self):
        assert x3d.Xcompact3dConvention is Xcompact3dConvention


class TestWrite:
    def test_write_dataset_only_writes_variables_with_file_name(self, case):
        prm, root, snapshots, _ = case
        convention = Xcompact3dConvention.from_parameters(prm)
        out = root / "out"
        derived = xr.Dataset({
            "w3": snapshots["pp"].drop_attrs().assign_attrs(file_name="w3"),
            "scratch": snapshots["pp"].drop_attrs(),
        })

        with pytest.warns(UserWarning, match="Can't write array scratch"):
            convention.write(derived, out)

        assert sorted(p.name for p in out.glob("*.bin")) == [f"w3-{k:03d}.bin" for k in range(5)]
        loaded = prm.dataset.load_array(str(out / "w3-002.bin"))
        xr.testing.assert_allclose(loaded, derived["w3"].isel(t=[2]).transpose(*loaded.dims))

    def test_write_data_array_with_file_prefix_is_read_by_the_loader(self, case):
        prm, root, snapshots, _ = case
        convention = Xcompact3dConvention.from_parameters(prm)
        vort = snapshots["u"].sel(i="y", drop=True).drop_attrs()

        convention.write(vort, root, file_prefix="w3")

        xr.testing.assert_allclose(prm.dataset["w3"], vort.transpose(*prm.dataset["w3"].dims))

    def test_write_requires_a_name(self, case):
        prm, root, snapshots, _ = case
        convention = Xcompact3dConvention.from_parameters(prm)

        with pytest.raises(ValueError, match="no name"):
            convention.write(snapshots["pp"].drop_attrs().rename(None), root)

    def test_write_creates_the_directory_and_folders(self, case):
        prm, root, _, epsi = case
        convention = Xcompact3dConvention.from_parameters(prm, folders={"geometry": {"static": True}})
        out = root / "fresh"

        convention.write(epsi, out)

        assert (out / "geometry" / "epsilon.bin").exists()

    def test_write_rejects_other_types(self, case):
        prm, root, *_ = case

        with pytest.raises(TypeError, match="xarray.Dataset or xarray.DataArray"):
            Xcompact3dConvention.from_parameters(prm).write([1, 2, 3], root)

    def test_write_reports_progress(self, case):
        prm, root, snapshots, _ = case
        seen = []

        def progress(specs):
            for spec in specs:
                seen.append(spec.filename)
                yield spec

        Xcompact3dConvention.from_parameters(prm).write(snapshots["pp"], root / "p", progress=progress)

        assert seen == [f"pp-{k:03d}.bin" for k in range(5)]

    def test_parameters_write_dataset_mirrors_open_dataset(self, case):
        prm, root, snapshots, _ = case
        vort = snapshots["u"].sel(i="z", drop=True).drop_attrs()

        prm.write_dataset(vort, file_prefix="w3")
        prm.write_dataset(snapshots[["phi"]], root / "copy")

        assert sorted(prm.open_dataset(variables=["w3"]).data_vars) == ["w3"]
        assert sorted(prm.open_dataset(root / "copy").data_vars) == ["phi"]
