import hypothesis
import numpy as np
import pytest
import stl
import xarray as xr

import xcompact3d_toolbox as x3d
import xcompact3d_toolbox.sandbox


@pytest.fixture(scope="session", autouse=True)
def cube():
    """Create a cube, from:
    https://numpy-stl.readthedocs.io/en/latest/usage.html#creating-mesh-objects-from-a-list-of-vertices-and-faces
    """

    # Define the 8 vertices of the cube
    vertices = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
    )
    # Define the 12 triangles composing the cube
    faces = np.array(
        [
            [0, 3, 1],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [4, 5, 6],
            [4, 6, 7],
            [5, 1, 2],
            [5, 2, 6],
            [2, 3, 6],
            [3, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
        ]
    )

    # Create the mesh
    cube = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]

    return cube


def test_init_epsi__no_ibm():
    prm = x3d.Parameters(xlx=2.0, yly=2.0, zlz=2.0, iibm=0)
    expected_result = {}
    actual_result = x3d.init_epsi(prm)
    assert expected_result == actual_result


def test_init_epsi__ibm():
    prm = x3d.Parameters(xlx=2.0, yly=2.0, zlz=2.0, iibm=1)

    actual_result = x3d.init_epsi(prm)
    assert actual_result.keys() == {"epsi"}


@hypothesis.settings(deadline=None)
@hypothesis.given(
    x=hypothesis.strategies.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    y=hypothesis.strategies.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    z=hypothesis.strategies.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@hypothesis.example(x=1.0, y=1.0, z=1.0)  # edge case
def test_point_is_inside_geometry(cube, x, y, z):
    inside_cube = all(-1.0 <= dim <= 1.0 for dim in [x, y, z])
    assert x3d.sandbox._point_in_geometry(cube.vectors, x, y, z, 0.05) == inside_cube  # noqa: SLF001


def test_geometry_from_stl(cube):
    prm = x3d.Parameters(xlx=2.0, yly=2.0, zlz=2.0, iibm=2)
    ds_stl = x3d.init_epsi(prm)["epsi"].geo.from_stl(stl_mesh=cube, user_tol=0.05)
    ds_box = x3d.init_epsi(prm)["epsi"].geo.box(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(-1.0, 1.0))
    xr.testing.assert_equal(ds_stl, ds_box)


@pytest.mark.parametrize("var_name", ["bxx1", "bxy1", "bxz1", "noise_mod_x1"])
class TestInitDatasetInflowBoundaryCondition:
    @pytest.fixture(scope="class")
    def prm(self):
        return x3d.Parameters(nclx1=2, nclxn=2)

    @pytest.fixture(scope="class")
    def dataset(self, prm):
        return x3d.init_dataset(prm)

    def test_dataset_contains_variable(self, var_name, dataset):
        assert var_name in dataset.variables

    def test_dataset_has_correct_dimensions(self, var_name, dataset):
        assert dataset[var_name].dims == ("y", "z")

    @pytest.mark.parametrize("nclx", [0, 1])
    def test_init_dataset__no_inflow_boundary_condition(self, var_name, nclx):
        prm = x3d.Parameters(nclx1=nclx, nclxn=nclx)
        ds = x3d.init_dataset(prm)
        assert var_name not in ds.variables


class TestInitDatasetScalarBoundaryConditions:
    @pytest.fixture
    def prm(self):
        return x3d.Parameters(nclx1=2, nclxn=2, numscalar=1)

    def test_dataset_contains__bxphi1(self, prm):
        prm.nclxS1 = 2
        ds = x3d.init_dataset(prm)
        assert "bxphi1" in ds.variables
        assert ds["bxphi1"].dims == ("n", "y", "z")

    def test_dataset_contains__byphi1(self, prm):
        prm.nclyS1 = 2
        ds = x3d.init_dataset(prm)
        assert "byphi1" in ds.variables
        assert ds["byphi1"].dims == ("n", "x", "z")

    def test_dataset_contains__byphin(self, prm):
        prm.nclySn = 2
        ds = x3d.init_dataset(prm)
        assert "byphin" in ds.variables
        assert ds["byphin"].dims == ("n", "x", "z")


class TestInitDatasetInitialConditions:
    @pytest.fixture
    def prm(self):
        return x3d.Parameters(numscalar=1)

    def test_dataset_contains__ux(self, prm):
        ds = x3d.init_dataset(prm)
        assert "ux" in ds.variables
        assert ds["ux"].dims == ("x", "y", "z")

    def test_dataset_contains__uy(self, prm):
        ds = x3d.init_dataset(prm)
        assert "uy" in ds.variables
        assert ds["uy"].dims == ("x", "y", "z")

    def test_dataset_contains__uz(self, prm):
        ds = x3d.init_dataset(prm)
        assert "uz" in ds.variables
        assert ds["uz"].dims == ("x", "y", "z")

    def test_dataset_contains__phi(self, prm):
        ds = x3d.init_dataset(prm)
        assert "phi" in ds.variables
        assert ds["phi"].dims == ("n", "x", "y", "z")


class TestInitDatasetFlowrateControl:
    def test_dataset_contains__flowrate(self):
        prm = x3d.Parameters(nclx1=0, nclxn=0)
        ds = x3d.init_dataset(prm)
        assert "vol_frc" in ds.variables
        assert ds["vol_frc"].dims == ("x", "y", "z")
