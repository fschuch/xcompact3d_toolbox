import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from xarray_binfile.conventions import Layout, LayoutMismatchError, StaticFiles

import xcompact3d_toolbox as x3d
from xcompact3d_toolbox.binfile import Xcompact3dConvention

# These tests exercise the deprecated on-demand loader on purpose.
pytestmark = pytest.mark.filterwarnings("ignore:prm.dataset is deprecated:FutureWarning")


@pytest.fixture
def prm(tmp_path):
    """A small 9x9x9 case with two scalars, five snapshots and a parameters file in tmp_path."""
    return x3d.Parameters(
        filename=(tmp_path / "input.i3d").as_posix(), nx=9, ny=9, nz=9, numscalar=2, ilast=100, ioutput=25, dt=0.01
    )


class TestFromParametersLayout:
    def test_layout_keeps_natural_dims_in_fortran_order(self, prm):
        """Layout keeps natural dims in fortran order."""
        convention = Xcompact3dConvention.from_parameters(prm)

        assert convention.layout.dims == ("x", "y", "z")
        assert convention.layout.shape == (9, 9, 9)
        assert convention.layout.order == "F"
        np.testing.assert_array_equal(convention.layout.coords["x"], prm.get_mesh()["x"])

    def test_dtype_defaults_to_param_mytype(self, prm, monkeypatch):
        """Dtype defaults to param mytype."""
        monkeypatch.setitem(x3d.param, "mytype", np.float32)

        assert np.dtype(Xcompact3dConvention.from_parameters(prm).layout.dtype) == np.float32
        assert np.dtype(Xcompact3dConvention.from_parameters(prm, dtype=np.float64).layout.dtype) == np.float64

    @pytest.mark.parametrize("drop_coords", ["x", "y", "z"])
    def test_drop_coords_gives_a_plane_layout(self, prm, drop_coords):
        """Drop coords gives a plane layout."""
        convention = Xcompact3dConvention.from_parameters(prm, drop_coords=drop_coords)

        assert convention.layout.dims == tuple(d for d in "xyz" if d != drop_coords)

    def test_layout_carries_coordinate_labels(self, prm):
        """Layout carries coordinate labels."""
        attrs = Xcompact3dConvention.from_parameters(prm).layout.coord_attrs

        assert attrs["x"]["name"] == "Streamwise coordinate"
        assert attrs["t"]["name"] == "Time"


def _field(prm, name, **extra):
    """A zero float32 field on the mesh of ``prm`` with the extra coordinates given."""
    coords = {**extra, **prm.get_mesh()}
    shape = tuple(len(v) for v in coords.values())
    return xr.DataArray(np.zeros(shape, dtype=np.float32), coords=coords, name=name)


class TestFromParametersFiles:
    def test_pattern_follows_filename_properties_with_exact_width(self, prm):
        """Pattern follows filename properties with exact width."""
        convention = Xcompact3dConvention.from_parameters(prm)

        assert convention.pattern.template == "{name}-{step:03d}.bin"
        assert convention.pattern.exact_width is True

    def test_pattern_is_decoupled_from_the_on_demand_loader(self, prm):
        """Pattern is decoupled from the on-demand loader."""
        prm.dataset.filename_properties.set(separator="", file_extension="", number_of_digits=4)
        default = Xcompact3dConvention.from_parameters(prm)
        explicit = Xcompact3dConvention.from_parameters(
            prm, filename_properties={"separator": ".", "file_extension": ".dat", "number_of_digits": 6}
        )

        assert default.pattern.template == "{name}-{step:03d}.bin"
        assert explicit.pattern.template == "{name}.{step:06d}.dat"

    def test_reader_decodes_snapshots_with_time_from_step(self, prm, monkeypatch):
        """Reader decodes snapshots with time from step."""
        monkeypatch.setitem(x3d.param, "mytype", np.float32)
        specs = Xcompact3dConvention.from_parameters(prm).reader(Path("ux-002.bin"))

        assert specs.name == "ux"
        assert specs.dims == ("x", "y", "z", "t")
        assert specs.order == "F"
        assert specs.coords["t"].dtype == np.float32
        np.testing.assert_allclose(specs.coords["t"], [2 * prm.dt * prm.ioutput])

    def test_snapshot_step_selects_the_output_frequency(self, prm):
        """Snapshot step selects the output frequency."""
        prm.iprocessing = 50
        specs = Xcompact3dConvention.from_parameters(prm, snapshot_step="iprocessing").reader(Path("ux-001.bin"))

        np.testing.assert_allclose(specs.coords["t"], [prm.dt * 50])

    def test_reader_decodes_static_files_in_the_root(self, prm):
        """Reader decodes static files in the root."""
        specs = Xcompact3dConvention.from_parameters(prm).reader(Path("epsilon.bin"))

        assert specs.name == "epsilon"
        assert specs.dims == ("x", "y", "z")

    def test_reader_rejects_unrelated_files(self, prm):
        """Reader rejects unrelated files."""
        convention = Xcompact3dConvention.from_parameters(prm)

        for name in ("snapshots.xdmf", "notes.txt", "ux-001.bin.bak"):
            path = Path(name)
            with pytest.raises(ValueError, match="No convention accepts"):
                convention.reader(path)

    def test_no_root_static_member_without_extension_unless_named(self, prm):
        """No root static member without extension unless named."""
        bare_names = {"separator": "", "file_extension": "", "number_of_digits": 4}
        bare = Xcompact3dConvention.from_parameters(prm, filename_properties=bare_names)
        named = Xcompact3dConvention.from_parameters(prm, filename_properties=bare_names, static_names=("epsilon",))

        assert bare.reader(Path("ux0001")).name == "ux"
        readme = Path("README")
        with pytest.raises(ValueError, match="No convention accepts"):
            bare.reader(readme)
        assert named.reader(Path("epsilon")).dims == ("x", "y", "z")
        with pytest.raises(ValueError, match="No convention accepts"):
            named.reader(readme)

    def test_stacks_declare_velocity_and_scalars(self, prm):
        """Stacks declare velocity and scalars."""
        convention = Xcompact3dConvention.from_parameters(prm, filename_properties={"scalar_num_of_digits": 2})
        stacks = {s.dim: s for s in convention.stacks}

        assert stacks["i"].template == "{name}{i}"
        assert stacks["i"].values == ("x", "y", "z")
        assert stacks["n"].template == "{name}{n:02d}"
        assert stacks["n"].names == {"phi"}
        assert stacks["n"].attrs["name"] == "Scalar fraction"

    def test_writer_splits_stacked_arrays_and_steps(self, prm):
        """Writer splits stacked arrays and steps."""
        convention = Xcompact3dConvention.from_parameters(prm)
        t = np.arange(2) * prm.dt * prm.ioutput
        u = _field(prm, "u", i=["x", "y", "z"], t=t)
        phi = _field(prm, "phi", n=[1, 2], t=t)

        assert [s.filename for s in convention.writer(u)] == [f"u{i}-{k:03d}.bin" for i in "xyz" for k in range(2)]
        assert [s.filename for s in convention.writer(phi)] == [f"phi{n}-{k:03d}.bin" for n in (1, 2) for k in range(2)]
        assert all(s.order == "F" and s.sub_array.dims == ("x", "y", "z") for s in convention.writer(u))

    def test_writer_takes_the_name_from_the_file_name_attribute(self, prm):
        """Writer takes the name from the file name attribute."""
        convention = Xcompact3dConvention.from_parameters(prm)
        vort = _field(prm, "vorticity", t=[0.0]).assign_attrs(file_name="w3")
        epsi = _field(prm, "epsi").assign_attrs(file_name="geometry/epsilon")

        assert [s.filename for s in convention.writer(vort)] == ["w3-000.bin"]
        assert [s.filename for s in convention.writer(epsi)] == ["geometry/epsilon.bin"]


@pytest.fixture
def case(prm, tmp_path, monkeypatch):
    """Snapshots written by the on-demand loader, plus a static geometry file."""
    monkeypatch.setitem(x3d.param, "mytype", np.float32)
    data_path = tmp_path / "data"  # the loader already points there: <prm dir>/data
    data_path.mkdir()
    prm.dataset.set(stack_velocity=True, stack_scalar=True)
    rng = np.random.default_rng(0)
    t = np.arange(len(prm.dataset)) * prm.dt * prm.ioutput

    def field(file_name, **extra):
        """A random float32 field carrying the ``file_name`` attribute, as the loader writes it."""
        array = _field(prm, file_name, **extra, t=t).assign_attrs(file_name=file_name)
        array.values[...] = rng.random(array.shape, dtype=np.float32)
        return array

    snapshots = xr.Dataset({"u": field("u", i=["x", "y", "z"]), "phi": field("phi", n=[1, 2]), "pp": field("pp")})
    epsi = _field(prm, "epsi").assign_attrs(file_name="geometry/epsilon")
    epsi.values[...] = rng.random(epsi.shape, dtype=np.float32)
    (data_path / "geometry").mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prm.dataset.write(snapshots)
        prm.dataset.write(epsi)
    for extra in ("snapshots.xdmf", "notes.txt", "input.i3d", ".DS_Store"):
        (data_path / extra).write_text("not a field")
    return prm, data_path, snapshots, epsi


class TestOpen:
    def test_files_lists_only_the_binary_fields(self, case):
        """Files lists only the binary fields."""
        prm, root, *_ = case
        files = Xcompact3dConvention.from_parameters(prm).files(root)

        assert all(p.suffix == ".bin" and p.parent == root for p in files)
        assert len(files) == 6 * 5

    def test_open_matches_the_on_demand_loader(self, case):
        """Open matches the on-demand loader."""
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
        """Open is lazy with one file per chunk by default."""
        prm, root, *_ = case
        lazy = Xcompact3dConvention.from_parameters(prm).open(root)

        assert lazy["u"].chunks is not None
        assert lazy["u"].chunks[-1] == (1,) * 5
        spatial = Xcompact3dConvention.from_parameters(prm).open(root, chunks={"x": 3})
        assert spatial["pp"].chunks[0] == (3, 3, 3)

    def test_open_without_stacking_shows_the_files(self, case):
        """Open without stacking shows the files."""
        prm, root, *_ = case
        raw = Xcompact3dConvention.from_parameters(prm).open(root, stack=False)

        assert sorted(raw.data_vars) == ["phi1", "phi2", "pp", "ux", "uy", "uz"]

    def test_stack_rebuilds_arrays_from_any_dataset(self, case):
        """Stack rebuilds arrays from any dataset."""
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
        """Open selects variables by disk or stacked name."""
        prm, root, *_ = case
        lazy = Xcompact3dConvention.from_parameters(prm).open(root, variables=variables)

        assert sorted(lazy.data_vars) == expected

    def test_open_with_wrong_dtype_fails_early(self, case):
        """Open with wrong dtype fails early."""
        prm, root, *_ = case

        convention = Xcompact3dConvention.from_parameters(prm, dtype=np.float64)

        with pytest.raises(ValueError, match="Size mismatch"):
            convention.open(root)


class TestFolders:
    def test_folder_spec_builds_a_sub_convention_from_the_same_parameters(self, case):
        """Folder spec builds a sub convention from the same parameters."""
        prm, root, _, epsi = case
        convention = Xcompact3dConvention.from_parameters(prm, folders={"geometry": {"static": True}})

        lazy = convention.open(root)

        assert sorted(lazy.data_vars) == ["epsilon", "phi", "pp", "u"]
        np.testing.assert_array_equal(lazy["epsilon"].transpose(*epsi.dims), epsi)

    def test_folder_spec_accepts_from_parameters_overrides(self, case):
        """Folder spec accepts from parameters overrides."""
        prm, root, *_ = case
        (root / "xy_planes").mkdir()
        plane = _field(prm, "ux", t=[0.0]).isel(z=0, drop=True)
        convention = Xcompact3dConvention.from_parameters(prm, folders={"xy_planes": {"drop_coords": "z"}})
        plane.binary_engine.to_file(convention.writer, root)

        specs = convention.reader(root / "xy_planes" / "ux-000.bin")

        assert specs.dims == ("x", "y", "t")
        assert (root / "xy_planes" / "ux-000.bin").exists()

    def test_folder_accepts_a_ready_convention(self, case):
        """Folder accepts a ready convention."""
        prm, root, *_ = case
        custom = StaticFiles(Layout({"x": np.arange(3)}, dtype="<f8"), pattern="{name}.dat")
        convention = Xcompact3dConvention.from_parameters(prm, folders={"probes": custom})

        assert convention.reader(root / "probes" / "p.dat").dims == ("x",)

    def test_writer_sends_folder_prefixed_names_to_the_registered_folder(self, case):
        """Writer sends folder prefixed names to the registered folder."""
        prm, root, *_ = case
        convention = Xcompact3dConvention.from_parameters(prm, folders={"geometry": {"static": True}})
        epsi = _field(prm, "epsi").assign_attrs(file_name="geometry/epsilon")

        assert [s.filename for s in convention.writer(epsi)] == ["geometry/epsilon.bin"]


class TestParametersOpenDataset:
    def test_open_dataset_defaults_to_the_data_folder_next_to_the_parameters_file(self, case):
        """Open dataset defaults to the data folder next to the parameters file."""
        prm, root, *_ = case
        prm.dataset.set(data_path="/somewhere/else/")  # the lazy API never reads the loader's traits

        lazy = prm.open_dataset()
        prm.dataset.set(data_path=root.as_posix() + "/")
        eager = prm.dataset[:]

        assert sorted(lazy.data_vars) == ["phi", "pp", "u"]
        xr.testing.assert_allclose(lazy["pp"].transpose(*eager["pp"].dims).load(), eager["pp"])

    def test_open_dataset_forwards_convention_and_open_options(self, case):
        """Open dataset forwards convention and open options."""
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
        """Convention is exported at top level."""
        assert x3d.Xcompact3dConvention is Xcompact3dConvention


class TestWrite:
    def test_write_dataset_only_writes_variables_with_file_name(self, case):
        """Write dataset only writes variables with file name."""
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
        """Write data array with file prefix is read by the loader."""
        prm, root, snapshots, _ = case
        convention = Xcompact3dConvention.from_parameters(prm)
        vort = snapshots["u"].sel(i="y", drop=True).drop_attrs()

        convention.write(vort, root, file_prefix="w3")

        xr.testing.assert_allclose(prm.dataset["w3"], vort.transpose(*prm.dataset["w3"].dims))

    def test_write_requires_a_name(self, case):
        """Write requires a name."""
        prm, root, snapshots, _ = case
        convention = Xcompact3dConvention.from_parameters(prm)

        unnamed = snapshots["pp"].drop_attrs().rename(None)

        with pytest.raises(ValueError, match="no name"):
            convention.write(unnamed, root)

    def test_write_creates_the_directory_and_folders(self, case):
        """Write creates the directory and folders."""
        prm, root, _, epsi = case
        convention = Xcompact3dConvention.from_parameters(prm, folders={"geometry": {"static": True}})
        out = root / "fresh"

        convention.write(epsi, out)

        assert (out / "geometry" / "epsilon.bin").exists()

    def test_write_rejects_other_types(self, case):
        """Write rejects other types."""
        prm, root, *_ = case

        convention = Xcompact3dConvention.from_parameters(prm)

        with pytest.raises(TypeError, match="xarray.Dataset or xarray.DataArray"):
            convention.write([1, 2, 3], root)

    def test_write_reports_progress(self, case):
        """Write reports progress."""
        prm, root, snapshots, _ = case
        seen = []

        def progress(specs):
            """Record the filenames the writer yields."""
            for spec in specs:
                seen.append(spec.filename)
                yield spec

        Xcompact3dConvention.from_parameters(prm).write(snapshots["pp"], root / "p", progress=progress)

        assert seen == [f"pp-{k:03d}.bin" for k in range(5)]

    def test_parameters_write_dataset_mirrors_open_dataset(self, case):
        """Parameters write dataset mirrors open dataset."""
        prm, root, snapshots, _ = case
        vort = snapshots["u"].sel(i="z", drop=True).drop_attrs()

        prm.write_dataset(vort, file_prefix="w3")
        prm.write_dataset(snapshots[["phi"]], root / "copy")

        assert sorted(prm.open_dataset(variables=["w3"]).data_vars) == ["w3"]
        assert sorted(prm.open_dataset(root / "copy").data_vars) == ["phi"]


class TestStaticPlanes:
    """Sandbox inflow planes (bxx1 on (y, z), byphi1 on (n, x, z)) are written, never opened."""

    @pytest.fixture
    def sandbox(self, prm):
        """The sandbox dataset with inflow planes and two scalar fractions."""
        prm.set(nclx1=2, nclxn=2, numscalar=2)
        return x3d.init_dataset(prm)

    def test_write_accepts_planes_and_splits_scalar_fractions(self, prm, sandbox, tmp_path):
        """Write accepts planes and splits scalar fractions."""
        out = tmp_path / "sandbox"
        convention = Xcompact3dConvention.from_parameters(prm)

        convention.write(sandbox, out)

        names = sorted(p.name for p in out.glob("*.bin"))
        assert {"bxx1.bin", "bxphi11.bin", "bxphi12.bin", "byphi11.bin", "ux.bin", "phi1.bin"} <= set(names)
        assert (out / "bxx1.bin").stat().st_size == prm.ny * prm.nz * 8
        assert (out / "byphi11.bin").stat().st_size == prm.nx * prm.nz * 8

    def test_planes_are_read_back_by_the_on_demand_loader(self, prm, sandbox, tmp_path):
        """Planes are read back by the on-demand loader."""
        out = tmp_path / "sandbox"
        Xcompact3dConvention.from_parameters(prm).write(sandbox, out)
        prm._dataset.set(data_path=out.as_posix() + "/", drop_coords="x")  # noqa: SLF001

        plane = prm._dataset.load_array(str(out / "bxx1.bin"), add_time=False)  # noqa: SLF001

        np.testing.assert_array_equal(plane, sandbox["bxx1"])

    def test_open_and_files_skip_planes(self, case):
        """Open and files skip planes."""
        prm, root, *_ = case
        convention = Xcompact3dConvention.from_parameters(prm)
        plane = _field(prm, "bxx1").isel(x=0, drop=True)
        convention.write(plane, root)
        convention.write(_field(prm, "byphi1", n=[1]).isel(y=0, drop=True), root)

        assert not any(p.name.startswith("b") for p in convention.files(root))
        assert sorted(convention.open(root).data_vars) == ["phi", "pp", "u"]

    def test_truncated_snapshot_still_fails_loudly(self, case):
        """Truncated snapshot still fails loudly."""
        prm, root, *_ = case
        target = root / "pp-002.bin"
        target.write_bytes(target.read_bytes()[: prm.ny * prm.nz * 4])  # looks like a yz plane, but is a snapshot

        convention = Xcompact3dConvention.from_parameters(prm)

        with pytest.raises(ValueError, match="Size mismatch"):
            convention.open(root)

    def test_planes_with_a_time_dimension_are_rejected(self, prm, tmp_path):
        """Planes with a time dimension are rejected."""
        plane = _field(prm, "bxx1", t=[0.0]).isel(x=0, drop=True)

        convention = Xcompact3dConvention.from_parameters(prm)

        with pytest.raises(LayoutMismatchError):
            convention.write(plane, tmp_path)


class TestReviewFixes:
    def test_registered_static_plane_folders_are_listed_and_opened(self, case):
        """Registered static plane folders are listed and opened."""
        prm, root, *_ = case
        (root / "planes").mkdir()
        convention = Xcompact3dConvention.from_parameters(prm, folders={"planes": {"static": True, "drop_coords": "z"}})
        plane = _field(prm, "bxx1").isel(z=0, drop=True).assign_attrs(file_name="planes/bxx1")
        convention.write(plane, root, progress=lambda specs: specs)

        assert any(path.parent.name == "planes" for path in convention.files(root))
        assert convention.open(root, variables=["bxx1"])["bxx1"].dims == ("x", "y")

    @pytest.mark.parametrize("prefix", ["w3-mean", "ux-000.bin", "a b"])
    def test_write_explains_the_name_rule(self, case, prefix):
        """Write explains the name rule."""
        prm, root, snapshots, _ = case

        convention = Xcompact3dConvention.from_parameters(prm)

        with pytest.raises(ValueError, match="letters, digits and underscores"):
            convention.write(snapshots["pp"], root, file_prefix=prefix)

    def test_only_u_and_phi_are_stacked(self, case):
        """The stacks act on configured names only, like the old loader's is_velocity/is_scalar."""
        prm, root, snapshots, _ = case
        convention = Xcompact3dConvention.from_parameters(prm)
        for name in ("ppx", "vortx", "vorty", "vortz"):
            convention.write(snapshots["pp"] * 2, root, file_prefix=name, progress=lambda specs: specs)

        opened = convention.open(root)

        assert sorted(opened.data_vars) == ["phi", "pp", "ppx", "u", "vortx", "vorty", "vortz"]
        xr.testing.assert_allclose(opened["pp"].transpose(*snapshots["pp"].dims).load(), snapshots["pp"])
        assert sorted(convention.open(root, variables=["pp"]).data_vars) == ["pp"]
        assert sorted(convention.open(root, variables=["u"]).data_vars) == ["u"]
        assert {stack.names for stack in convention.stacks} == {frozenset({"u"}), frozenset({"phi"})}

    def test_write_splits_by_dimension_whatever_the_name(self, case):
        """Unstacking on write follows the dims i and n; pp has neither and is written as is."""
        prm, root, snapshots, _ = case
        convention = Xcompact3dConvention.from_parameters(prm)
        vort = snapshots["u"].drop_attrs().rename("vort")  # carries i
        conc = snapshots["phi"].drop_attrs().rename("conc")  # carries n
        out = root / "derived"

        for array in (vort, conc, snapshots["pp"]):
            convention.write(array, out, progress=lambda specs: specs)

        names = {path.name.split("-")[0] for path in out.glob("*.bin")}
        assert names == {"vortx", "vorty", "vortz", "conc1", "conc2", "pp"}
        # on read, only the configured names u and phi are stacked back
        assert sorted(convention.open(out).data_vars) == ["conc1", "conc2", "pp", "vortx", "vorty", "vortz"]


class TestStackNames:
    def test_default_stacks_u_and_phi_only(self, prm):
        """Default stacks u and phi only."""
        convention = Xcompact3dConvention.from_parameters(prm)

        assert {stack.dim: stack.names for stack in convention.stacks} == {"i": {"u"}, "n": {"phi"}}

    def test_extra_names_are_stacked_back(self, case):
        """Extra names are stacked back."""
        prm, root, snapshots, _ = case
        writer = Xcompact3dConvention.from_parameters(prm)
        writer.write(snapshots["u"].drop_attrs().rename("vort"), root, progress=lambda specs: specs)
        writer.write(snapshots["phi"].drop_attrs().rename("conc"), root, progress=lambda specs: specs)

        opened = prm.open_dataset(root, stack_names={"i": {"u", "vort"}, "n": {"phi", "conc"}})

        assert sorted(opened.data_vars) == ["conc", "phi", "pp", "u", "vort"]
        assert opened["vort"].dims == ("i", "x", "y", "z", "t")
        assert opened["conc"]["n"].values.tolist() == [1, 2]
        assert sorted(prm.open_dataset(root, stack_names={"i": {"vort"}}, variables=["vort"]).data_vars) == ["vort"]

    def test_a_dimension_can_be_left_unstacked(self, case):
        """A dimension can be left unstacked."""
        prm, root, *_ = case

        opened = prm.open_dataset(root, stack_names={"i": set()})

        assert sorted(opened.data_vars) == ["phi", "pp", "ux", "uy", "uz"]

    def test_unknown_dimension_is_rejected(self, prm):
        """Unknown dimension is rejected."""
        with pytest.raises(ValueError, match="stack_names.*'i' and 'n'"):
            Xcompact3dConvention.from_parameters(prm, stack_names={"j": {"u"}})

    def test_stack_names_reach_folders(self, case):
        """Stack names reach folders."""
        prm, root, snapshots, _ = case
        (root / "sub").mkdir()
        convention = Xcompact3dConvention.from_parameters(prm, stack_names={"i": {"vort"}}, folders={"sub": {}})
        vort = snapshots["u"].drop_attrs().rename("sub/vort")
        convention.write(vort, root, progress=lambda specs: specs)

        assert "vort" in convention.open(root).data_vars
