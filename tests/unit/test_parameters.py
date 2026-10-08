import os.path
import warnings

import pytest

import xcompact3d_toolbox as x3d
from xcompact3d_toolbox.gui import ParametersGui
from xcompact3d_toolbox.io import Dataset
from xcompact3d_toolbox.param import COORDS
from xcompact3d_toolbox.parameters import Parameters

PARAMETERS = (Parameters, ParametersGui)


@pytest.mark.parametrize("base_class", PARAMETERS)
class TestParameters:
    @pytest.fixture
    def parameters(self, tmp_path, base_class) -> Parameters:
        filename = (tmp_path / "test.i3d").as_posix()
        return base_class(filename=filename)

    @pytest.mark.parametrize("target_class", PARAMETERS)
    def test_io(self, parameters: Parameters, target_class: Parameters):
        prm1 = parameters

        expected_values = {k: v for k, v in prm1.trait_values().items() if prm1.trait_metadata(k, "group")}

        prm1.write()
        prm2 = target_class(filename=prm1.filename)
        prm2.load()

        actual_values = {k: v for k, v in prm2.trait_values().items() if prm2.trait_metadata(k, "group")}

        assert expected_values == actual_values

    @pytest.mark.parametrize("dimension", COORDS)
    def test_observe_resolution_and_bc(self, parameters: Parameters, dimension: str):
        prm = parameters

        # Default Values
        assert getattr(prm, f"n{dimension}") == 17
        assert getattr(prm, f"d{dimension}") == 0.0625
        assert getattr(prm, f"{dimension}l{dimension}") == 1.0
        # New nx should change just dx
        setattr(prm, f"n{dimension}", 201)
        assert getattr(prm, f"n{dimension}") == 201
        assert getattr(prm, f"d{dimension}") == 0.005
        assert getattr(prm, f"{dimension}l{dimension}") == 1.0
        # New xlx should change just dx
        setattr(prm, f"{dimension}l{dimension}", 5.0)
        assert getattr(prm, f"n{dimension}") == 201
        assert getattr(prm, f"d{dimension}") == 0.025
        assert getattr(prm, f"{dimension}l{dimension}") == 5.0
        # New dx should change just xlx
        setattr(prm, f"d{dimension}", 0.005)
        assert getattr(prm, f"n{dimension}") == 201
        assert getattr(prm, f"d{dimension}") == 0.005
        assert getattr(prm, f"{dimension}l{dimension}") == 1.0
        # One side to periodic
        setattr(prm, f"ncl{dimension}1", 0)
        assert getattr(prm, f"ncl{dimension}1") == 0
        assert getattr(prm, f"ncl{dimension}n") == 0
        assert getattr(prm.mesh, dimension).is_periodic is True
        assert getattr(prm, f"n{dimension}") == 200
        assert getattr(prm, f"d{dimension}") == 0.005
        assert getattr(prm, f"{dimension}l{dimension}") == 1.0
        # and back
        setattr(prm, f"ncl{dimension}1", 1)
        assert getattr(prm, f"ncl{dimension}1") == 1
        assert getattr(prm, f"ncl{dimension}n") == 1
        assert getattr(prm.mesh, dimension).is_periodic is False
        assert getattr(prm, f"n{dimension}") == 201
        assert getattr(prm, f"d{dimension}") == 0.005
        assert getattr(prm, f"{dimension}l{dimension}") == 1.0
        # Other side to periodic
        setattr(prm, f"ncl{dimension}n", 0)
        assert getattr(prm, f"ncl{dimension}1") == 0
        assert getattr(prm, f"ncl{dimension}n") == 0
        assert getattr(prm.mesh, dimension).is_periodic is True
        assert getattr(prm, f"n{dimension}") == 200
        assert getattr(prm, f"d{dimension}") == 0.005
        assert getattr(prm, f"{dimension}l{dimension}") == 1.0
        # and back
        setattr(prm, f"ncl{dimension}n", 2)
        assert getattr(prm, f"ncl{dimension}1") == 2
        assert getattr(prm, f"ncl{dimension}n") == 2
        assert getattr(prm.mesh, dimension).is_periodic is False
        assert getattr(prm, f"n{dimension}") == 201
        assert getattr(prm, f"d{dimension}") == 0.005
        assert getattr(prm, f"{dimension}l{dimension}") == 1.0

    @pytest.mark.parametrize(
        ("i3d_path", "data_path"),
        [
            ("./example/input.i3d", "./example/data/"),
            ("../tutorial/case/input.i3d", "../tutorial/case/data/"),
            ("input.i3d", "./data/"),
        ],
    )
    def test_initial_datapath(self, base_class, i3d_path, data_path):
        prm = base_class(filename=i3d_path)
        with pytest.warns(FutureWarning, match="deprecated"):
            loader_path = prm.dataset.data_path
        assert os.path.normpath(loader_path) == os.path.normpath(data_path)
        assert os.path.normpath(prm.default_data_path) == os.path.normpath(data_path)

    @pytest.mark.parametrize("ncores", [2, 4, 8, 16, 32, 64, 128])
    def test_observe_2decomp__ncores(self, parameters: Parameters, ncores: int):
        prm = parameters
        prm.set(ncores=ncores)
        prm.set(p_row=2, p_col=int(ncores / 2))

        assert prm.ncores == ncores
        assert prm.p_row == 2
        assert prm.p_col == int(ncores / 2)

        prm.set(ncores=1)
        assert prm.ncores == 1
        assert prm.p_row == 0
        assert prm.p_col == 0


class TestDatasetDeprecation:
    @pytest.fixture
    def prm(self, tmp_path):
        return Parameters(filename=(tmp_path / "input.i3d").as_posix(), nx=9, ny=9, nz=9)

    def test_dataset_access_warns_and_points_to_the_lazy_api(self, prm):
        with pytest.warns(FutureWarning, match=r"prm\.dataset.*deprecated.*open_dataset.*migration-guide"):
            loader = prm.dataset

        assert loader.data_path == os.path.join(os.path.dirname(prm.filename), "data")

    def test_toolbox_code_paths_do_not_trigger_the_warning(self, prm):
        prm.set(iibm=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            ds = x3d.init_dataset(prm)
            prm.write_dataset(ds)
            x3d.gene_epsi_3d(x3d.init_epsi(prm), prm)
            prm.open_dataset(stack=False)

    def test_dataset_can_still_be_assigned_with_a_warning(self, prm):
        with pytest.warns(FutureWarning, match="deprecated"):
            prm.dataset = Dataset(stack_velocity=True)
        with pytest.warns(FutureWarning, match="deprecated"):
            assert prm.dataset.stack_velocity is True

    def test_resolve_data_path_prefers_argument_then_changed_loader_path(self, prm, tmp_path):
        assert prm._resolve_data_path(tmp_path / "given") == os.fspath(tmp_path / "given")  # noqa: SLF001
        assert prm._resolve_data_path(None) == prm.default_data_path  # noqa: SLF001
        prm._dataset.set(data_path="/elsewhere/")  # noqa: SLF001
        assert prm._resolve_data_path(None) == "/elsewhere/"  # noqa: SLF001
