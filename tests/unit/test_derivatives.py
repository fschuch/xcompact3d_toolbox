import unittest

import numpy as np

from xcompact3d_toolbox.derive import first_derivative, second_derivative


class TestDerivative(unittest.TestCase):
    def setUp(self):
        self.nx = 129
        self.lx = 2.0 * np.pi

        self.x = np.linspace(0.0, self.lx, num=self.nx, endpoint=True)

        self.dx = self.x[1] - self.x[0]

        self.cos = np.cos(self.x)
        self.sin = np.sin(self.x)

        self.tol = 1e-4

    def test_periodic(self):
        x = np.linspace(0.0, self.lx, num=self.nx - 1, endpoint=False)
        nx = x.size
        dx = x[1] - x[0]
        cos = np.cos(x)
        sin = np.sin(x)

        # Cos - Symmetric
        np.testing.assert_allclose(first_derivative(nx, dx, 0, 0).dot(cos), -sin, atol=self.tol)
        np.testing.assert_allclose(second_derivative(nx, dx, 0, 0).dot(cos), -cos, atol=self.tol)
        # Sin - Antisymmetric
        np.testing.assert_allclose(first_derivative(nx, dx, 0, 0).dot(sin), cos, atol=self.tol)
        np.testing.assert_allclose(second_derivative(nx, dx, 0, 0).dot(sin), -sin, atol=self.tol)

    def test_dirichlet(self):
        # Cos - Symmetric
        np.testing.assert_allclose(first_derivative(self.nx, self.dx, 2, 2).dot(self.cos), -self.sin, atol=self.tol)
        np.testing.assert_allclose(second_derivative(self.nx, self.dx, 2, 2).dot(self.cos), -self.cos, atol=self.tol)

        # Sin - Antisymmetric
        np.testing.assert_allclose(first_derivative(self.nx, self.dx, 2, 2).dot(self.sin), self.cos, atol=self.tol)
        np.testing.assert_allclose(second_derivative(self.nx, self.dx, 2, 2).dot(self.sin), -self.sin, atol=self.tol)

    def test_free_slip(self):
        # Cos - Symmetric
        np.testing.assert_allclose(first_derivative(self.nx, self.dx, 1, 1, 1).dot(self.cos), -self.sin, atol=self.tol)
        np.testing.assert_allclose(second_derivative(self.nx, self.dx, 1, 1, 1).dot(self.cos), -self.cos, atol=self.tol)
        # Sin - Antisymmetric
        np.testing.assert_allclose(first_derivative(self.nx, self.dx, 1, 1, 0).dot(self.sin), self.cos, atol=self.tol)
        np.testing.assert_allclose(second_derivative(self.nx, self.dx, 1, 1, 0).dot(self.sin), -self.sin, atol=self.tol)


if __name__ == "__main__":
    unittest.main()
