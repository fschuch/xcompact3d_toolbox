"""
Lazy, Dask-backed access to the raw binary files produced by XCompact3d,
built on `xarray-binfile`_.

This module describes the files of a simulation as an xarray-binfile
convention, assembled from a :obj:`xcompact3d_toolbox.parameters.Parameters`
instance. It coexists with :obj:`xcompact3d_toolbox.io.Dataset`, the
on-demand loader, and both read and write the same files.

.. _xarray-binfile: https://docs.fschuch.com/xarray-binfile/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray_binfile  # noqa: F401  (registers the .binary_engine accessors)
from xarray_binfile.conventions import (
    ConventionProtocol,
    FilenamePattern,
    FolderConventions,
    Layout,
    PatternConventions,
    StaticFiles,
    StepIndexedFiles,
    VariableStack,
)

from xcompact3d_toolbox.param import param

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

    import xarray as xr
    from xarray_binfile import ReadSpecs, WriteSpecs

    from xcompact3d_toolbox.io import FilenameProperties
    from xcompact3d_toolbox.parameters import Parameters

COORD_ATTRS: dict[str, dict[str, str]] = {
    "x": {"name": "Streamwise coordinate", "long_name": r"$x_1$"},
    "y": {"name": "Vertical coordinate", "long_name": r"$x_2$"},
    "z": {"name": "Spanwise coordinate", "long_name": r"$x_3$"},
    "t": {"name": "Time", "long_name": r"$t$"},
    "i": {"name": "Velocity component", "long_name": r"$i$"},
    "n": {"name": "Scalar fraction", "long_name": r"$\ell$"},
}
"""Attributes attached to the coordinates of the lazy dataset, matching :obj:`xcompact3d_toolbox.sandbox.init_dataset`."""

VELOCITY_COMPONENTS = ("x", "y", "z")
SCALAR_FRACTIONS = range(1, 10)


def _output_name(data_array: xr.DataArray) -> str:
    """The name an array is written under: its ``file_name`` attribute, else its name."""
    return str(data_array.attrs.get("file_name", data_array.name))


def _filename_template(filename_properties: FilenameProperties) -> str:
    fp = filename_properties
    return f"{{name}}{fp.separator}{{step:0{fp.number_of_digits}d}}{fp.file_extension}"


@dataclass(frozen=True)
class Xcompact3dConvention:
    """Describes the binary files of a simulation for `xarray-binfile`_.

    Build it with :obj:`from_parameters`. It implements the xarray-binfile
    convention protocol (:obj:`reader` and :obj:`writer`), so it can be
    handed to :obj:`xarray.open_mfdataset` and ``.binary_engine.to_file``
    directly.

    Parameters
    ----------
    convention : :obj:`xarray_binfile.conventions.ConventionProtocol`
        The composed xarray-binfile conventions doing the work.
    layout : :obj:`xarray_binfile.conventions.Layout`
        Dimension order, coordinates, dtype and memory order of one file on disk.
    pattern : :obj:`xarray_binfile.conventions.FilenamePattern`
        Filename pattern of the snapshots.
    stacks : tuple of :obj:`xarray_binfile.conventions.VariableStack`
        Dimensions encoded in variable names (velocity components on ``i``,
        scalar fractions on ``n``).

    .. _xarray-binfile: https://docs.fschuch.com/xarray-binfile/
    """

    convention: ConventionProtocol
    layout: Layout
    pattern: FilenamePattern
    stacks: tuple[VariableStack, ...]
    _time_dim: str = "t"

    @classmethod
    def from_parameters(
        cls,
        prm: Parameters,
        *,
        dtype: Any = None,
        drop_coords: str = "",
        filename_properties: FilenameProperties | Mapping[str, Any] | None = None,
        snapshot_step: str = "ioutput",
        time_dim: str = "t",
        static_names: Sequence[str] | None = None,
    ) -> Xcompact3dConvention:
        """Build the convention from a :obj:`xcompact3d_toolbox.parameters.Parameters` instance.

        The dataset root holds the snapshots (``ux-000.bin``) next to static
        fields (``epsilon.bin``). Velocity components are stacked on ``i``
        (``u`` from ``ux``, ``uy``, ``uz``) and scalar fractions on ``n``
        (``phi`` from ``phi1``, ``phi2``, ...). The name an array is written
        under is its ``file_name`` attribute, falling back to its name, like
        :obj:`xcompact3d_toolbox.io.Dataset.write`.

        Parameters
        ----------
        prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
            The simulation parameters; mesh, time step and output frequency are read from it.
        dtype : dtype_like, optional
            On-disk data type of every file. Defaults to ``xcompact3d_toolbox.param["mytype"]``
            at call time.
        drop_coords : str, optional
            Coordinate to drop when the files hold 2D planes: ``"x"``, ``"y"`` or ``"z"``
            (default is ``""``, for 3D fields).
        filename_properties : :obj:`xcompact3d_toolbox.io.FilenameProperties` or dict, optional
            Naming of the files. Defaults to ``prm.dataset.filename_properties``; a dict
            of its keyword arguments is accepted too.
        snapshot_step : str, optional
            The parameter giving the number of time steps between snapshots, ``"ioutput"``
            (default) or ``"iprocessing"``; with ``prm.dt`` it sets the ``t`` coordinate.
        time_dim : str, optional
            Name of the time dimension (default is ``"t"``).
        static_names : sequence of str, optional
            Names of the static fields stored in the root. By default any name is accepted
            when ``file_extension`` is set; with an empty extension, a bare name would also
            match unrelated files (``README``), so static files are only recognised when
            listed here.

        Returns
        -------
        :obj:`Xcompact3dConvention`
            The convention.
        """
        from xcompact3d_toolbox.io import FilenameProperties  # noqa: PLC0415  (import cycle)

        if filename_properties is None:
            fp = prm.dataset.filename_properties
        elif isinstance(filename_properties, FilenameProperties):
            fp = filename_properties
        else:
            fp = FilenameProperties(**filename_properties)

        layout = Layout(
            prm.mesh.drop(*drop_coords),
            dtype=np.dtype(param["mytype"] if dtype is None else dtype),
            order="F",
            coord_attrs=COORD_ATTRS,
        )
        pattern = FilenamePattern(_filename_template(fp), exact_width=True)
        stacks = (
            VariableStack("i", "{name}{i}", values=VELOCITY_COMPONENTS, attrs=COORD_ATTRS["i"]),
            VariableStack(
                "n",
                f"{{name}}{{n:0{fp.scalar_num_of_digits}d}}",
                values=SCALAR_FRACTIONS,
                names=("phi",),
                attrs=COORD_ATTRS["n"],
            ),
        )
        snapshots = StepIndexedFiles(
            layout,
            pattern=pattern,
            time_dim=time_dim,
            time_step=prm.dt * getattr(prm, snapshot_step),
            time_dtype=layout.dtype,
            stacks=stacks,
            name_of=_output_name,
        )
        members: list[ConventionProtocol] = [snapshots]
        if fp.file_extension or static_names is not None:
            members.append(
                StaticFiles(
                    layout,
                    pattern=FilenamePattern(f"{{name}}{fp.file_extension}"),
                    names=static_names,
                    name_of=_output_name,
                )
            )
        convention = FolderConventions({".": PatternConventions(members)})
        return cls(convention=convention, layout=layout, pattern=pattern, stacks=stacks, _time_dim=time_dim)

    def reader(self, path: Path) -> ReadSpecs:
        """Read specs getter: decode one file (see the xarray-binfile read protocol)."""
        return self.convention.reader(path)

    def writer(self, data_array: xr.DataArray) -> Iterator[WriteSpecs]:
        """Write specs getter: split one array into files (see the xarray-binfile write protocol)."""
        return self.convention.writer(data_array)

    def files(self, directory: str | os.PathLike[str]) -> list[Path]:
        """List the binary fields in ``directory`` that follow the convention.

        Unrelated files (``snapshots.xdmf``, notes, hidden files) are never listed.

        Parameters
        ----------
        directory : str or path-like
            The data folder.

        Returns
        -------
        list of :obj:`pathlib.Path`
            The matching files, sorted.
        """
        return self.convention.files(directory)  # type: ignore[attr-defined]

    def stack(self, dataset: xr.Dataset) -> xr.Dataset:
        """Rebuild ``u`` from ``ux``, ``uy``, ``uz`` and ``phi`` from ``phi1``, ``phi2``, ... (lazily).

        Parameters
        ----------
        dataset : :obj:`xarray.Dataset`
            A dataset as decoded by the ``binfile`` engine.

        Returns
        -------
        :obj:`xarray.Dataset`
            The dataset with the stacked arrays.
        """
        return self.convention.stack(dataset)  # type: ignore[attr-defined]

    def _disk_names(self, variables: Iterable[str]) -> set[str]:
        """Expand stacked names (``u``, ``phi``) into the names found on disk."""
        names: set[str] = set()
        for variable in variables:
            expanded = [
                stack.template.format(name=variable, **{stack.dim: value})
                for stack in self.stacks
                if stack.names is None or variable in stack.names
                for value in stack.values
            ]
            names.update(expanded or [variable])
            names.add(variable)
        return names

    def open(
        self,
        directory: str | os.PathLike[str],
        *,
        variables: Iterable[str] | None = None,
        stack: bool = True,
        chunks: Any = None,
        parallel: bool = True,
        **open_mfdataset_kwargs: Any,
    ) -> xr.Dataset:
        """Open every field in ``directory`` as one lazy, Dask-backed :obj:`xarray.Dataset`.

        Nothing is read until values are needed; see `xarray's guide on Dask`_.

        Parameters
        ----------
        directory : str or path-like
            The data folder.
        variables : iterable of str, optional
            Names to load, either as found on disk (``"ux"``) or stacked (``"u"``, ``"phi"``).
            By default every field is loaded.
        stack : bool, optional
            Rebuild ``u`` and ``phi`` from their components (default is :obj:`True`).
        chunks : dict or str, optional
            Dask chunking, forwarded to :obj:`xarray.open_mfdataset` and applied to each
            file before they are combined, so a ``t`` chunk is at most one file. The
            default, ``{"t": 1}``, is one task per file. Choose chunks for your workload:
            spatial chunks for large planes, or ``.chunk({"t": 10})`` on the result for
            time statistics over many snapshots.
        parallel : bool, optional
            Open files in parallel with Dask (default is :obj:`True`).
        **open_mfdataset_kwargs
            Other options for :obj:`xarray.open_mfdataset`.

        Returns
        -------
        :obj:`xarray.Dataset`
            The lazy dataset, with dims ``(x, y, z, t)`` plus ``i`` and ``n`` when stacked.

        Raises
        ------
        FileNotFoundError
            If no field is found.
        ValueError
            If a file size does not match the layout, for instance a wrong ``dtype``.

        .. _`xarray's guide on Dask`: https://docs.xarray.dev/en/stable/user-guide/dask.html
        """
        if chunks is None:
            chunks = {self._time_dim: 1}
        if variables is not None:
            variables = self._disk_names(variables)
        return self.convention.open(  # type: ignore[attr-defined]
            directory,
            variables=variables,
            stack=stack,
            chunks=chunks,
            parallel=parallel,
            **open_mfdataset_kwargs,
        )
