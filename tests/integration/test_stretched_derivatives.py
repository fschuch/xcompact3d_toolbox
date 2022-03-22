import unittest
import numpy as np

from xcompact3d_toolbox.parameters.parameters import Parameters
from xcompact3d_toolbox.sandbox import init_dataset


class test_stretched_derive(unittest.TestCase):
    def test_derivative(self):

        prm = Parameters(ny=129, yly=2.0 * np.pi, nclx1=1, nclxn=1, nclz1=1, nclzn=1)

        for istret in [0, 1, 2, 3]:
            for beta in [0.75, 1.0, 4.0]:
                # for BC in "00 11 22".split():
                for BC in "00 11 12 21 22".split():
                    if istret == 3 and BC == "00":
                        continue

                    with self.subTest(istret=istret, beta=beta, BC=BC):

                        tol = 1e-1

                        prm.set(
                            istret = istret,
                            beta = beta,
                            ncly1 = int(BC[0]),
                            nclyn = int(BC[1])
                        )

                        ds = init_dataset(prm).isel(x=0, z=0)

                        # Cos - Symmetric
                        ds["ux"] += np.cos(ds.y)

                        self.assertTrue(
                            np.allclose(
                                ds.ux.x3d.first_derivative("y").values,
                                -np.sin(ds.y.values),
                                atol=tol,
                                rtol=tol,
                            )
                        )

                        self.assertTrue(
                            np.allclose(
                                ds.ux.x3d.second_derivative("y").values,
                                -np.cos(ds.y.values),
                                atol=tol,
                                rtol=tol,
                            )
                        )

                        # Sin - Antisymmetric
                        ds["uy"] += np.sin(ds.y)

                        self.assertTrue(
                            np.allclose(
                                ds.uy.x3d.first_derivative("y").values,
                                np.cos(ds.y.values),
                                atol=tol,
                                rtol=tol,
                            )
                        )

                        self.assertTrue(
                            np.allclose(
                                ds.uy.x3d.second_derivative("y").values,
                                -np.sin(ds.y.values),
                                atol=tol,
                                rtol=tol,
                            )
                        )


if __name__ == "__main__":
    unittest.main()
