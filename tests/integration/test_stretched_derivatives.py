import numpy as np
import pytest

from xcompact3d_toolbox.parameters import Parameters
from xcompact3d_toolbox.sandbox import init_dataset


@pytest.mark.parametrize("istret", [0, 1, 2, 3])
@pytest.mark.parametrize("beta", [0.75, 1.0, 4.0])
@pytest.mark.parametrize("boundary_condition", "00 11 12 21 22".split())
def test_derivative(istret, beta, boundary_condition):
    if istret == 3 and boundary_condition == "00":
        return

    prm = Parameters(
        ny=128 if boundary_condition == "00" else 129,
        yly=2.0 * np.pi,
        nclx1=1,
        nclxn=1,
        nclz1=1,
        nclzn=1,
        istret=istret,
        beta=beta,
        ncly1=int(boundary_condition[0]),
        nclyn=int(boundary_condition[1]),
    )

    tol = 1e-1

    ds = init_dataset(prm).isel(x=0, z=0)

    # Cos - Symmetric
    ds["ux"] += np.cos(ds.y)

    np.testing.assert_allclose(
        ds.ux.x3d.first_derivative("y").values,
        -np.sin(ds.y.values),
        atol=tol,
        rtol=tol,
    )

    np.testing.assert_allclose(
        ds.ux.x3d.second_derivative("y").values,
        -np.cos(ds.y.values),
        atol=tol,
        rtol=tol,
    )

    # Sin - Antisymmetric
    ds["uy"] += np.sin(ds.y)

    np.testing.assert_allclose(
        ds.uy.x3d.first_derivative("y").values,
        np.cos(ds.y.values),
        atol=tol,
        rtol=tol,
    )

    np.testing.assert_allclose(
        ds.uy.x3d.second_derivative("y").values,
        -np.sin(ds.y.values),
        atol=tol,
        rtol=tol,
    )
