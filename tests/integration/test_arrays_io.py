import unittest
import numpy as np
import xarray as xr

from xcompact3d_toolbox.parameters import Parameters
from xcompact3d_toolbox.array import X3dDataset, X3dDataArray


class test_arrays_io(unittest.TestCase):
    def test_io(self):
        #
        prm1 = Parameters(nx=31, ny=9, nz=19, numscalar=3, ilast=1000, ioutput=200)
        prm2 = Parameters(
            nx=9,
            ny=19,
            nz=31,
            numscalar=5,
            ilast=1000,
            ioutput=200,
            filenamedigits=1,
            ifilenameformat="(I4.4)",
        )
        #
        np.random.seed(seed=67)
        for i, (prm, fileformat) in enumerate(zip([prm1, prm2], [".bin", ""])):
            with self.subTest(parameters=i):
                #
                coords = prm.get_mesh
                coords["t"] = prm.dt * np.arange(0, prm.ilast, prm.ioutput)
                coords["n"] = range(prm.numscalar)
                # Init dataset
                ds_out = xr.Dataset(
                    dict(
                        ux=(
                            ["x", "y", "z"],
                            np.random.random((prm.nx, prm.ny, prm.nz)),
                            dict(file_name="ux"),
                        ),
                        uy=(
                            ["t", "x", "y", "z"],
                            np.random.random(
                                (coords["t"].size, prm.nx, prm.ny, prm.nz)
                            ),
                            dict(file_name="uy"),
                        ),
                        uz=(
                            ["t", "x", "y"],
                            np.random.random((coords["t"].size, prm.nx, prm.ny)),
                            dict(file_name="uz"),
                        ),
                        phi=(
                            ["n", "t", "x", "y", "z"],
                            np.random.random(
                                (
                                    prm.numscalar,
                                    coords["t"].size,
                                    prm.nx,
                                    prm.ny,
                                    prm.nz,
                                )
                            ),
                            dict(file_name="phi"),
                        ),
                    ),
                    coords=coords,
                )
                i1, i2 = prm.ifilenameformat[2:-1].split(".")
                # Writes
                ds_out.x3d.write(prm, fileformat=fileformat)
                # Reads
                ds_in = prm.read_field(f"ux{fileformat}", name="ux").to_dataset()
                ds_in["uy"] = prm.read_all_fields(f"uy-{'?'*int(i1)}{fileformat}")
                coords = prm.get_mesh
                del coords["z"]
                ds_in["uz"] = prm.read_all_fields(
                    f"uz-{'?'*int(i1)}{fileformat}", coords=coords
                )
                ds_in["phi"] = xr.concat(
                    [
                        prm.read_all_fields(f"phi{n+1}-{'?'*int(i1)}{fileformat}")
                        for n in range(prm.numscalar)
                    ],
                    "n",
                ).assign_coords(n=("n", range(prm.numscalar)))
                # Compares
                for var in ds_out.keys():
                    with self.subTest(variable=var):
                        self.assertTrue(np.allclose(ds_out[var], ds_in[var]))


if __name__ == "__main__":
    unittest.main()
