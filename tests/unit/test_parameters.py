import unittest
import os.path
import pytest
import traitlets
from xcompact3d_toolbox.parameters.parameters import Parameters

# TODO - migrate do Pytest
# TODO 2 - Test ParametersGui as well


class test_parameters(unittest.TestCase):
    def test_io(self):

        prm1 = Parameters()
        prm1.write()
        prm2 = Parameters()
        prm2.load()

        for name in prm1.trait_names():
            group = prm1.trait_metadata(name, "group")
            if group is None:
                continue
            with self.subTest(name=name):
                self.assertEqual(getattr(prm1, name), getattr(prm2, name))

    def test_observe_resolution_and_BC(self):

        prm = Parameters()

        for dim in "x y z".split():
            with self.subTest(dim=dim):
                # Default Values
                self.assertEqual(getattr(prm, f"n{dim}"), 17)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.0625)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 1.0)
                # New nx should change just dx
                setattr(prm, f"n{dim}", 201)
                self.assertEqual(getattr(prm, f"n{dim}"), 201)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.005)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 1.0)
                # New xlx should change just dx
                setattr(prm, f"{dim}l{dim}", 5.0)
                self.assertEqual(getattr(prm, f"n{dim}"), 201)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.025)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 5.0)
                # New dx should change just xlx
                setattr(prm, f"d{dim}", 0.005)
                self.assertEqual(getattr(prm, f"n{dim}"), 201)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.005)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 1.0)
                # One side to periodic
                setattr(prm, f"ncl{dim}1", 0)
                self.assertEqual(getattr(prm, f"ncl{dim}1"), 0)
                self.assertEqual(getattr(prm, f"ncl{dim}n"), 0)
                self.assertEqual(getattr(prm.mesh, dim).is_periodic, True)
                self.assertEqual(getattr(prm, f"n{dim}"), 200)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.005)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 1.0)
                # and back
                setattr(prm, f"ncl{dim}1", 1)
                self.assertEqual(getattr(prm, f"ncl{dim}1"), 1)
                self.assertEqual(getattr(prm, f"ncl{dim}n"), 1)
                self.assertEqual(getattr(prm.mesh, dim).is_periodic, False)
                self.assertEqual(getattr(prm, f"n{dim}"), 201)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.005)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 1.0)
                # Other side to periodic
                setattr(prm, f"ncl{dim}n", 0)
                self.assertEqual(getattr(prm, f"ncl{dim}1"), 0)
                self.assertEqual(getattr(prm, f"ncl{dim}n"), 0)
                self.assertEqual(getattr(prm.mesh, dim).is_periodic, True)
                self.assertEqual(getattr(prm, f"n{dim}"), 200)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.005)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 1.0)
                # and back
                setattr(prm, f"ncl{dim}n", 2)
                self.assertEqual(getattr(prm, f"ncl{dim}1"), 2)
                self.assertEqual(getattr(prm, f"ncl{dim}n"), 2)
                self.assertEqual(getattr(prm.mesh, dim).is_periodic, False)
                self.assertEqual(getattr(prm, f"n{dim}"), 201)
                self.assertEqual(getattr(prm, f"d{dim}"), 0.005)
                self.assertEqual(getattr(prm, f"{dim}l{dim}"), 1.0)


@pytest.mark.parametrize(
    "i3d_path, data_path",
    [
        ("./example/input.i3d", "./example/data/"),
        ("../tutorial/case/input.i3d", "../tutorial/case/data/"),
        ("input.i3d", "./data/"),
    ],
)
def test_initial_datapath(i3d_path, data_path):
    prm = Parameters(filename=i3d_path)
    assert os.path.normpath(prm.dataset.data_path) == os.path.normpath(data_path)


if __name__ == "__main__":
    unittest.main()
