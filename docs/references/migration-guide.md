# Migration guide: from `prm.dataset` to the lazy API

Version 1.5.0 introduces a lazy, Dask-backed way to read and write the raw binary files of XCompact3d, built on [xarray-binfile](https://docs.fschuch.com/xarray-binfile/). The on-demand loader `prm.dataset` keeps working in the 1.x series but emits a `FutureWarning`, and it will be removed in version 2.0. This page maps every call to its replacement.

## Why

- **Lazy and parallel.** `prm.open_dataset()` describes the whole simulation at once; values are read only when needed, one file per Dask task, so larger-than-memory workflows and parallel evaluation come for free.
- **One description of the files.** Naming, byte order, dtype, mesh, time and the stacked `u`/`phi` are declared once, in [`Xcompact3dConvention`](api-reference.rst), and reused for reading, writing and discovery.
- **Standard xarray.** The same convention is registered as the xarray engine `xcompact3d`, so `xarray.open_mfdataset` works out of the box.
- **Less toolbox-specific code.** The file mechanics now live in xarray-binfile, which is shared with other solvers.

## What is deprecated

Access to `prm.dataset` itself, and therefore all of its methods and settings. The two APIs are independent: the lazy API never reads `prm.dataset.filename_properties`, `data_path`, `drop_coords`, `snapshot_step`, `stack_velocity` or `stack_scalar`. Configure one or the other, never both.

## Call by call

| On-demand loader (deprecated)                                                                   | Lazy API                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prm.dataset.set(data_path="./data/")`                                                          | pass `data_path` to `prm.open_dataset(...)` / `prm.write_dataset(...)`; the default is `data/` next to the parameters file (`prm.default_data_path`)                                                                                                                          |
| `prm.dataset.filename_properties.set(separator="-", file_extension=".bin", number_of_digits=3)` | `prm.open_dataset(filename_properties={"separator": "-", "file_extension": ".bin", "number_of_digits": 3})`; the same defaults apply                                                                                                                                          |
| `prm.dataset.set(drop_coords="z")`                                                              | `prm.open_dataset(drop_coords="z")`                                                                                                                                                                                                                                           |
| `prm.dataset.set(snapshot_step="iprocessing")`                                                  | `prm.open_dataset(snapshot_step="iprocessing")`                                                                                                                                                                                                                               |
| `prm.dataset.set(stack_velocity=True, stack_scalar=True)`                                       | always stacked; `prm.open_dataset(stack=False)` to see `ux`, `uy`, `uz`, `phi1`, ... as on disk. `stack_names={"i": {"u", "vort"}}` also reassembles your own fields (`vortx`, `vorty`, `vortz` into `vort`); writing splits any array carrying `i` or `n`, whatever its name |
| `prm.dataset.set(set_of_variables={"ux", "uy"})`                                                | `prm.open_dataset(variables=["ux", "uy"])`, or `variables=["u"]` for the stacked name                                                                                                                                                                                         |
| `x3d.param["mytype"] = np.float32`                                                              | still the default dtype; override per call with `prm.open_dataset(dtype=np.float32)`                                                                                                                                                                                          |
| `prm.dataset["ux"]` (time series of one variable)                                               | `prm.open_dataset(variables=["ux"], stack=False)["ux"]`                                                                                                                                                                                                                       |
| `prm.dataset[10]` (one snapshot)                                                                | `prm.open_dataset().isel(t=10)`                                                                                                                                                                                                                                               |
| `prm.dataset[0:101:10]`, `prm.dataset[:]`                                                       | `prm.open_dataset().isel(t=slice(0, 101, 10))`, `prm.open_dataset()`                                                                                                                                                                                                          |
| `for ds in prm.dataset:` / `prm.dataset(0, 101, 5)`                                             | iterate `prm.open_dataset().t`, or better, express the computation on the whole series and let Dask split the work                                                                                                                                                            |
| `prm.dataset.load_array("ux-000.bin")`                                                          | `xr.open_dataset("data/ux-000.bin", engine="xcompact3d", prm=prm)`                                                                                                                                                                                                            |
| `prm.dataset.load_array("epsilon.bin", add_time=False)`                                         | `prm.open_dataset(variables=["epsilon"])["epsilon"]` (static files are opened with the snapshots)                                                                                                                                                                             |
| `prm.dataset.load_snapshot(10, list_of_variables=["ux"])`                                       | `prm.open_dataset(variables=["ux"]).isel(t=10)`                                                                                                                                                                                                                               |
| `prm.dataset.load_time_series("ux")`                                                            | `prm.open_dataset(variables=["ux"], stack=False)["ux"]`                                                                                                                                                                                                                       |
| `prm.dataset.write(vort, file_prefix="w3")`                                                     | `prm.write_dataset(vort, file_prefix="w3")`                                                                                                                                                                                                                                   |
| `prm.dataset.write(ds)` (variables with `file_name`)                                            | `prm.write_dataset(ds)` (same rule, same warning for the others)                                                                                                                                                                                                              |
| `prm.dataset.write_xdmf()`                                                                      | not migrated yet; keep calling `prm.dataset.write_xdmf()` (the warning is the only difference)                                                                                                                                                                                |
| `prm.dataset.load_wind_turbine_data()`                                                          | not migrated yet; keep calling it on `prm.dataset`                                                                                                                                                                                                                            |

`len(prm.dataset)`, which guessed the number of snapshots from `ilast` and `ioutput`, has no equivalent: the lazy dataset lists what is on disk, `prm.open_dataset().sizes["t"]`.

## It is a plain xarray.Dataset

`prm.open_dataset()` returns an ordinary, lazy [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html). The toolbox cannot anticipate every case, so when the defaults do not fit, open with `stack=False` or with `xr.open_mfdataset(..., engine="xcompact3d", prm=prm)` and shape the result with xarray itself: [`xr.concat`](https://docs.xarray.dev/en/stable/generated/xarray.concat.html) to stack components along a new dimension, [`xr.merge`](https://docs.xarray.dev/en/stable/generated/xarray.merge.html) and [`xr.combine_by_coords`](https://docs.xarray.dev/en/stable/generated/xarray.combine_by_coords.html) to join folders or runs, [`Dataset.stack`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.stack.html), `rename`, `assign_coords`, `chunk`, `isel`/`sel`, and so on. Everything stays lazy until you compute.

## Files in sub-folders

XCompact3d versions and user cases lay files out differently, so no folder tree is assumed. Declare the ones you have:

```python
ds = prm.open_dataset(
    folders={
        "xy_planes": {"drop_coords": "z"},
        "3d": {},
        "geometry": {"static": True},
    }
)
```

Each entry is built from the same parameters with the given overrides, or it can be a ready xarray-binfile convention. On write, an array named `geometry/epsilon` (through its `file_name` attribute) goes to that folder.

## Static planes from the sandbox

`init_dataset` produces inflow planes such as `bxx1` on `(y, z)`. `prm.write_dataset` writes them as static files, like the old writer did, but `prm.open_dataset` never lists 2D static files: it is a reader for the simulation output. Read a plane with `xr.open_dataset(path, engine="xcompact3d", prm=prm, drop_coords="x")`.

## Sandbox and genepsi

`init_epsi`, `init_dataset` and `gene_epsi_3d` now write through the lazy API and accept an explicit `data_path`. Without it they default to `prm.default_data_path`, or to a changed `prm.dataset.data_path` while that setting exists.

## Known differences

**Output names.** The old writer accepted any `file_prefix` and appended the extension when missing, so `prm.dataset.write(vort, "w3-mean")` or `"ux-000.bin"` worked. The lazy API derives the separator, step counter and extension from the filename pattern, so a name is a word made of letters, digits and underscores, optionally under sub-folders (`"geometry/epsilon"`). Other names are rejected with a message saying so: use `w3_mean` instead of `w3-mean`, and never include the counter or extension yourself.

**Assigning `prm.dataset`.** Still possible (`prm.dataset = Dataset(...)`), with the same deprecation warning as reading it.

The old writer computed the step of a snapshot as `int(t / dt)`, which truncates (`0.3 / 0.1` is `2.999...`) and could make two snapshots overwrite each other. The lazy API rounds and refuses values that are not within tolerance of an integer step.
