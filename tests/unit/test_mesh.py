import unittest
import numpy as np
import os.path
import glob

from xcompact3d_toolbox.parameters import Parameters


class test_mesh(unittest.TestCase):
    def test_mesh_stretching(self):
        """In order to test the stretched mesh, three values for istret:

        * ``istret = 1``;
        * ``istret = 2``;
        * ``istret = 3``;

        where combined with two values for beta:

        * ``beta = 0.75``;
        * ``beta = 1``;
        * ``beta = 4``.

        Xcompact3d wrote the coordinate y to the disc (all with ``ny=55``),
        the six files where saved as:

        * ``./test/unit/data/yp/yp_<istret>_<beta>_.dat``,

        so now we can compare and validate the python implementation.

        """

        filenames = glob.glob(os.path.join("tests", "unit", "data", "yp", "*.dat"))

        prm = Parameters()

        for file in filenames:
            with self.subTest(file=file):

                yp = np.loadtxt(file)

                prm.istret = int(file.split("_")[1])
                prm.beta = float(file.split("_")[2])
                prm.ny = yp.size

                mesh = prm.get_mesh

                self.assertTrue(np.allclose(mesh["y"], yp))


if __name__ == "__main__":
    unittest.main()
