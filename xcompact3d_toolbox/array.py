from .param import param
from .derive import FirstDerivative, SecondDerivative
import xarray as xr
import numpy as np
import os.path

@xr.register_dataset_accessor("x3d")
class X3dDataset:
    def __init__(self, data_set):
        self._data_set = data_set

    def write(self, prm):
        for key, val in self._data_set.items():
            if 'file_name' in val.attrs:
                print('Writing : ' + key)
                val.x3d.write(prm)

    def cumtrapz(self, dim):
        from scipy.integrate import cumtrapz
        return xr.apply_ufunc(cumtrapz,
                              self._data_set,
                              input_core_dims=[[dim]],
                              output_core_dims=[[dim]],
                              dask="parallelized",
                              output_dtypes=[param['mytype']],
                              kwargs={
                                  'x': self._data_set[dim],
                                  'axis': -1,
                                  'initial': 0.0
                              })

    def simps(self, dim):
        from scipy.integrate import simps
        return xr.apply_ufunc(simps,
                              self._data_set,
                              input_core_dims=[[dim]],
                              dask="parallelized",
                              output_dtypes=[param['mytype']],
                              kwargs={
                                  'x': self._data_set[dim],
                                  'axis': -1
                              })

    def pencil_decomp(self, dim=None, chunks=None):

        if chunks == None and dim != None:
            chunks = {}
            for var in self._data_set.dims:
                if var in dim:
                    # no chunking along this dimension
                    chunks[var] = -1
                else:
                    # allow the chunking in this dimension to accommodate ideal chunk sizes
                    chunks[var] = 'auto'

        return self._data_set.chunk(chunks)

@xr.register_dataarray_accessor("x3d")
class X3dDataArray:
    def __init__(self, data_array):
        self._data_array = data_array

        self._Dx = {}
        self._Dxx = {}

    def write(self, prm, filename=None):
        if filename == None: # Try to get from atributes
            filename = self._data_array.attrs.get('file_name', None)
        if filename != None:
            if 'n' in self._data_array.dims:
                for n in self._data_array.n:
                    self._data_array.sel(n=n).x3d.write(prm,filename+str(n.values+1))
            elif 't' in self._data_array.dims:

                from tqdm.notebook import tqdm as tqdm

                fmt = prm.ifilenameformat
                k = 0
                for t in tqdm(self._data_array.t.values, desc = filename):
                    num = str(int(t / prm.dt)).zfill(fmt)
                    self._data_array.isel(t=k).x3d.write(prm,filename+'-'+num)
                    k += 1
            else:
                align = []
                for i in reversed(sorted(self._data_array.dims)):
                    align.append(self._data_array.get_axis_num(i))
                self._data_array.values.astype(
                    param['mytype']).transpose(align).tofile(filename + '.bin')

    def cumtrapz(self, dim):
        ds = self._data_array._to_temp_dataset().x3d.cumtrapz(dim)
        return self._data_array._from_temp_dataset(ds)

    def simps(self, dim):
        ds = self._data_array._to_temp_dataset().x3d.simps(dim)
        return self._data_array._from_temp_dataset(ds)

    def pencil_decomp(self, dim=None, chunks=None):
        ds = self._data_array._to_temp_dataset().x3d.pencil_decomp(dim, chunks)
        return self._data_array._from_temp_dataset(ds)

    def first_derivative(self, dim):

        if dim not in self._Dx:
            try:
                ncl1 = self._data_array.attrs['BC'][dim]['ncl1']
                ncln = self._data_array.attrs['BC'][dim]['ncln']
                npaire = self._data_array.attrs['BC'][dim]['npaire']
            except:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            d = (self._data_array[dim][1] - self._data_array[dim][0]).values
            self._Dx[dim] = FirstDerivative(n, d, ncl1, ncln, npaire)

        return xr.apply_ufunc(
            lambda f : self._Dx[dim].dot(f),
            self._data_array,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[param['mytype']],
        )

    def second_derivative(self, dim):

        if dim not in self._Dxx:
            try:
                ncl1 = self._data_array.attrs['BC'][dim]['ncl1']
                ncln = self._data_array.attrs['BC'][dim]['ncln']
                npaire = self._data_array.attrs['BC'][dim]['npaire']
            except:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            d = (self._data_array[dim][1] - self._data_array[dim][0]).values
            self._Dxx[dim] = SecondDerivative(n, d, ncl1, ncln, npaire)

        return xr.apply_ufunc(
            lambda f : self._Dxx[dim].dot(f),
            self._data_array,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[param['mytype']],
        )
