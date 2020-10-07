import unittest
import numpy as np

from xcompact3d_toolbox.derive import SecondDerivative, FirstDerivative


class test_derivative(unittest.TestCase):
    def setUp(self):

        self.nx = 129
        self.lx = 2.0 * np.pi

        self.x = np.linspace(0.0, self.lx, num=self.nx, endpoint=True)

        self.dx = self.x[1] - self.x[0]

        self.cos = np.cos(self.x)
        self.sin = np.sin(self.x)

    def test_periodic(self):

        x = np.linspace(0.0, self.lx, num=self.nx - 1, endpoint=False)
        nx = x.size
        dx = x[1] - x[0]
        cos = np.cos(x)
        sin = np.sin(x)

        # Cos - Symmetric
        self.assertTrue(np.allclose(FirstDerivative(nx, dx, 0, 0).dot(cos), -sin))
        self.assertTrue(np.allclose(SecondDerivative(nx, dx, 0, 0).dot(cos), -cos))
        # Sin - Antisymmetric
        self.assertTrue(np.allclose(FirstDerivative(nx, dx, 0, 0).dot(sin), cos))
        self.assertTrue(np.allclose(SecondDerivative(nx, dx, 0, 0).dot(sin), -sin))

    def test_dirichlet(self):

        # Cos - Symmetric
        self.assertTrue(
            np.allclose(
                FirstDerivative(self.nx, self.dx, 2, 2).dot(self.cos),
                -self.sin,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                SecondDerivative(self.nx, self.dx, 2, 2).dot(self.cos),
                -self.cos,
                atol=1e-4,
            )
        )
        # Sin - Antisymmetric
        self.assertTrue(
            np.allclose(
                FirstDerivative(self.nx, self.dx, 2, 2).dot(self.sin),
                self.cos,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                SecondDerivative(self.nx, self.dx, 2, 2).dot(self.sin),
                -self.sin,
                atol=1e-4,
            )
        )

    def test_free_slip(self):

        # Cos - Symmetric
        self.assertTrue(
            np.allclose(
                FirstDerivative(self.nx, self.dx, 1, 1, 1).dot(self.cos), -self.sin
            )
        )
        self.assertTrue(
            np.allclose(
                SecondDerivative(self.nx, self.dx, 1, 1, 1).dot(self.cos), -self.cos
            )
        )
        # Sin - Antisymmetric
        self.assertTrue(
            np.allclose(
                FirstDerivative(self.nx, self.dx, 1, 1, 0).dot(self.sin), self.cos
            )
        )
        self.assertTrue(
            np.allclose(
                SecondDerivative(self.nx, self.dx, 1, 1, 0).dot(self.sin), -self.sin
            )
        )


if __name__ == "__main__":
    unittest.main()
