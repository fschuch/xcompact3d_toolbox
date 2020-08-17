import unittest

import traitlets
from xcompact3d_toolbox.parameters import Parameters

class test_parameters(unittest.TestCase):

    def test_io(self):

        prm1 = Parameters()
        prm1.write()
        prm2 = Parameters()
        prm2.read()

        for name in prm1.trait_names():
            if name == 'i3d':
                continue
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(prm1, name),
                    getattr(prm2, name)
                )

    def test_observe_resolution_and_BC(self):

        prm = Parameters()

        for dim in 'x y z'.split():
            with self.subTest(dim=dim):
                # Default Values
                self.assertEqual(getattr(prm, f'n{dim}'), 17)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.0625)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 1.0)
                # New nx should change just dx
                setattr(prm, f'n{dim}', 201)
                self.assertEqual(getattr(prm, f'n{dim}'), 201)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.005)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 1.0)
                # New xlx should change just dx
                setattr(prm, f'{dim}l{dim}', 5.0)
                self.assertEqual(getattr(prm, f'n{dim}'), 201)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.025)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 5.0)
                # New dx should change just xlx
                setattr(prm, f'd{dim}', 0.005)
                self.assertEqual(getattr(prm, f'n{dim}'), 201)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.005)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 1.0)
                # One side to periodic
                setattr(prm, f'ncl{dim}1', 0)
                self.assertEqual(getattr(prm, f'ncl{dim}1'), 0)
                self.assertEqual(getattr(prm, f'ncl{dim}n'), 0)
                self.assertEqual(getattr(prm, f'ncl{dim}'), True)
                self.assertEqual(getattr(prm, f'n{dim}'), 200)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.005)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 1.0)
                # End back
                setattr(prm, f'ncl{dim}1', 1)
                self.assertEqual(getattr(prm, f'ncl{dim}1'), 1)
                self.assertEqual(getattr(prm, f'ncl{dim}n'), 1)
                self.assertEqual(getattr(prm, f'ncl{dim}'), False)
                self.assertEqual(getattr(prm, f'n{dim}'), 201)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.005)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 1.0)
                # Other side to periodic
                setattr(prm, f'ncl{dim}n', 0)
                self.assertEqual(getattr(prm, f'ncl{dim}1'), 0)
                self.assertEqual(getattr(prm, f'ncl{dim}n'), 0)
                self.assertEqual(getattr(prm, f'ncl{dim}'), True)
                self.assertEqual(getattr(prm, f'n{dim}'), 200)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.005)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 1.0)
                # End back
                setattr(prm, f'ncl{dim}n', 2)
                self.assertEqual(getattr(prm, f'ncl{dim}1'), 2)
                self.assertEqual(getattr(prm, f'ncl{dim}n'), 2)
                self.assertEqual(getattr(prm, f'ncl{dim}'), False)
                self.assertEqual(getattr(prm, f'n{dim}'), 201)
                self.assertEqual(getattr(prm, f'd{dim}'), 0.005)
                self.assertEqual(getattr(prm, f'{dim}l{dim}'), 1.0)

    def test_validate_mesh(self):

        from xcompact3d_toolbox.parameters import possible_mesh, possible_mesh_p

        prm = Parameters()

        # To small grid
        for dim in 'x y z'.split():
            with self.subTest(dim=dim):
                with self.assertRaises(traitlets.TraitError):
                    setattr(prm, f'n{dim}', 8)

        # Invalid grid not periodic
        for dim in 'x y z'.split():
            for i in range(possible_mesh[-1]):
                with self.subTest(dim=dim, i=i):
                    if i not in possible_mesh:
                        with self.assertRaises(traitlets.TraitError):
                            setattr(prm, f'n{dim}', i)

        # Invalid grid Periodic
        for dim in 'x y z'.split():
            setattr(prm, f'ncl{dim}1', 0)
            for i in range(possible_mesh_p[-1]):
                with self.subTest(dim=dim, i=i):
                    if i not in possible_mesh_p:
                        with self.assertRaises(traitlets.TraitError):
                            setattr(prm, f'n{dim}', i)

if __name__ == "__main__":
    unittest.main()
