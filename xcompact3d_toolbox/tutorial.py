from __future__ import annotations

import warnings

import xarray as xr

from xcompact3d_toolbox.parameters import Parameters

xr.tutorial.external_urls["cylinder"] = "https://github.com/fschuch/xcompact3d_toolbox_data/raw/main/cylinder.nc"


def open_dataset(name: str, **kws) -> tuple[xr.Dataset, Parameters]:
    """
    Open a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Available datasets:
    * ``"cylinder"``: Flow around a cylinder

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'cylinder'.
    **kws : dict, optional
        Passed to :obj:`xarray.tutorial.open_dataset`

    See Also
    --------
    xarray.open_dataset
    """
    ds = xr.tutorial.open_dataset(name, **kws)
    # have a prm attribute, write it to the disc, del prm attribute

    prm = Parameters()

    if "prm" in ds.attrs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            prm.from_string(ds.attrs.get("prm"), raise_warning=True)
        del ds.attrs["prm"]

    return ds, prm
