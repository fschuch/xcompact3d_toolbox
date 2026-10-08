import warnings

import numpy as np
import pytest
import xarray as xr

import xcompact3d_toolbox as x3d
from xcompact3d_toolbox.backend import Xcompact3dEntrypoint


@pytest.fixture
def case(tmp_path):
    """A case folder: input.i3d next to data/ holding snapshots written by the lazy API."""
    prm = x3d.Parameters(filename=(tmp_path / "input.i3d").as_posix(), nx=9, ny=9, nz=9, ilast=50, ioutput=25, dt=0.01)
    prm.write()
    mesh = prm.get_mesh()
    t = np.arange(3) * prm.dt * prm.ioutput
    rng = np.random.default_rng(1)
    fields = xr.Dataset({
        name: xr.DataArray(rng.random((3, 9, 9, 9)), coords={"t": t, **mesh}, attrs={"file_name": name})
        for name in ("ux", "uy", "pp")
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prm.write_dataset(fields)
    return prm, tmp_path / "data", fields


class TestEngine:
    def test_engine_is_registered(self):
        assert "xcompact3d" in xr.backends.list_engines()

    def test_open_mfdataset_with_prm_matches_open_dataset(self, case):
        prm, data, fields = case

        standard = xr.open_mfdataset(sorted(data.glob("*.bin")), engine="xcompact3d", prm=prm)
        toolbox = prm.open_dataset(stack=False)

        xr.testing.assert_allclose(standard.load(), toolbox.load())
        assert sorted(standard.data_vars) == ["pp", "ux", "uy"]

    def test_open_dataset_with_a_convention(self, case):
        prm, data, fields = case
        convention = x3d.Xcompact3dConvention.from_parameters(prm)

        single = xr.open_dataset(data / "pp-001.bin", engine="xcompact3d", convention=convention)

        xr.testing.assert_allclose(single["pp"].transpose("t", "x", "y", "z"), fields["pp"].isel(t=[1]))

    def test_convention_options_are_forwarded(self, case):
        prm, data, _ = case

        single = xr.open_dataset(data / "pp-001.bin", engine="xcompact3d", prm=prm, time_dim="time")

        assert "time" in single.dims

    def test_without_prm_the_parameters_file_next_to_the_data_folder_is_loaded(self, case):
        prm, data, fields = case

        guessed = xr.open_mfdataset(sorted(data.glob("pp-*.bin")))  # no engine, no prm

        assert guessed["pp"].dims == ("x", "y", "z", "t")
        np.testing.assert_allclose(guessed["t"], fields["t"])

    def test_without_prm_and_without_parameters_file_it_refuses(self, case):
        prm, data, _ = case
        (data.parent / "input.i3d").unlink()

        with pytest.raises(ValueError, match="prm=.*convention="):
            xr.open_dataset(data / "pp-001.bin", engine="xcompact3d")


class TestGuessCanOpen:
    entrypoint = Xcompact3dEntrypoint()

    def test_claims_bin_files_next_to_a_single_parameters_file(self, case):
        _, data, _ = case

        assert self.entrypoint.guess_can_open(data / "pp-001.bin")
        assert self.entrypoint.guess_can_open(str(data / "pp-001.bin"))

    def test_accepts_prm_files_too(self, case):
        _, data, _ = case
        (data.parent / "input.i3d").rename(data.parent / "case.prm")

        assert self.entrypoint.guess_can_open(data / "pp-001.bin")

    def test_does_not_claim_other_extensions(self, case):
        _, data, _ = case

        assert not self.entrypoint.guess_can_open(data / "snapshots.xdmf")
        assert not self.entrypoint.guess_can_open(data.parent / "input.i3d")

    def test_does_not_claim_without_or_with_several_parameters_files(self, case):
        _, data, _ = case
        (data.parent / "other.i3d").write_text("")
        assert not self.entrypoint.guess_can_open(data / "pp-001.bin")

        (data.parent / "other.i3d").unlink()
        (data.parent / "input.i3d").unlink()
        assert not self.entrypoint.guess_can_open(data / "pp-001.bin")

    def test_does_not_claim_non_paths(self):
        assert not self.entrypoint.guess_can_open(object())
