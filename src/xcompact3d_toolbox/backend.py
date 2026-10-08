"""
An `xarray backend`_ for the raw binary files of XCompact3d, so they open with the
standard :obj:`xarray.open_dataset` and :obj:`xarray.open_mfdataset`::

    xr.open_mfdataset("case/data/*.bin", engine="xcompact3d", prm=prm)

It is a thin layer over the ``binfile`` engine from `xarray-binfile`_ that builds
the :obj:`xcompact3d_toolbox.binfile.Xcompact3dConvention` for you. When no
engine is given, xarray asks every backend whether it can open the file; this one
claims a ``.bin`` file only when exactly one parameters file (``*.i3d`` or
``*.prm``) sits next to its data folder, and then loads it::

    xr.open_mfdataset("case/data/*.bin")  # finds case/input.i3d

A backend decodes one file at a time, so velocity components and scalar
fractions come back as ``ux``, ``uy``, ``uz`` and ``phi1``, ``phi2``, ... Use
:obj:`xcompact3d_toolbox.parameters.Parameters.open_dataset` to get them
stacked into ``u`` and ``phi``, or call ``convention.stack(ds)`` afterwards.

.. _`xarray backend`: https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html
.. _xarray-binfile: https://docs.fschuch.com/xarray-binfile/
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xarray_binfile import RawBinaryEntrypoint

from xcompact3d_toolbox.binfile import Xcompact3dConvention
from xcompact3d_toolbox.parameters import Parameters

if TYPE_CHECKING:
    from collections.abc import Iterable

    import xarray as xr

PARAMETERS_SUFFIXES = (".i3d", ".prm")


def find_parameters_file(path: str | os.PathLike[str]) -> Path | None:
    """Find the single parameters file next to the data folder of a binary field.

    For ``case/data/ux-000.bin`` it looks for ``case/*.i3d`` and ``case/*.prm``.

    Parameters
    ----------
    path : str or path-like
        A binary field.

    Returns
    -------
    :obj:`pathlib.Path` or None
        The parameters file, or :obj:`None` when there is none or more than one.
    """
    case = Path(os.path.abspath(path)).parent.parent
    if not case.is_dir():
        return None
    candidates = [p for p in case.iterdir() if p.is_file() and p.suffix in PARAMETERS_SUFFIXES]
    return candidates[0] if len(candidates) == 1 else None


@lru_cache(maxsize=64)
def load_parameters(parameters_file: str, mtime_ns: int) -> Parameters:  # noqa: ARG001  (mtime keys the cache)
    """Load a parameters file, reusing the result while the file is unchanged.

    :obj:`xarray.open_mfdataset` opens every file through the backend, so without this
    cache a run with thousands of snapshots would parse the same ``.i3d`` thousands of
    times. The modification time is part of the key, so an edited file is reloaded.

    Parameters
    ----------
    parameters_file : str
        The ``.i3d`` or ``.prm`` file.
    mtime_ns : int
        Its modification time, ``Path(parameters_file).stat().st_mtime_ns``.

    Returns
    -------
    :obj:`xcompact3d_toolbox.parameters.Parameters`
        The loaded parameters (shared between calls; treat it as read-only).
    """
    return Parameters(loadfile=parameters_file)


def parameters_for(path: str | os.PathLike[Any]) -> Parameters:
    """The parameters of the case a binary field belongs to.

    Parameters
    ----------
    path : str or path-like
        A binary field.

    Returns
    -------
    :obj:`xcompact3d_toolbox.parameters.Parameters`
        The parameters loaded from the single ``*.i3d`` or ``*.prm`` next to the data folder.

    Raises
    ------
    ValueError
        If there is no such file, or more than one.
    """
    parameters_file = find_parameters_file(path)
    if parameters_file is None:
        msg = (
            f"Cannot open {path}: pass prm= (a Parameters instance) or convention= "
            "(an Xcompact3dConvention), or keep exactly one *.i3d or *.prm file next to the data folder."
        )
        raise ValueError(msg)
    return load_parameters(parameters_file.as_posix(), parameters_file.stat().st_mtime_ns)


class Xcompact3dEntrypoint(RawBinaryEntrypoint):
    """Backend entry point registered as ``engine="xcompact3d"``.

    See :obj:`open_dataset` for the keywords it accepts on top of the usual ones.

    .. versionadded:: 1.5.0
    """

    description = "Read the raw binary files of XCompact3d, with the mesh and time from its parameters file."
    url = "https://docs.fschuch.com/xcompact3d_toolbox/references/api-reference.html#module-xcompact3d_toolbox.backend"

    open_dataset_parameters = (
        "filename_or_obj",
        "drop_variables",
        "prm",
        "convention",
        *sorted(Xcompact3dConvention.FROM_PARAMETERS_KEYS),
    )

    def open_dataset(  # type: ignore[override]
        self,
        filename_or_obj: str | os.PathLike[Any],
        *,
        drop_variables: str | Iterable[str] | None = None,
        prm: Parameters | None = None,
        convention: Xcompact3dConvention | None = None,
        **convention_kwargs: Any,
    ) -> xr.Dataset:
        """Open one binary field as a lazy :obj:`xarray.Dataset`.

        Parameters
        ----------
        filename_or_obj : str or path-like
            The binary field.
        drop_variables : str or iterable of str, optional
            Variables to leave out.
        prm : :obj:`xcompact3d_toolbox.parameters.Parameters`, optional
            The simulation parameters. Loaded from the case folder when omitted.
        convention : :obj:`xcompact3d_toolbox.binfile.Xcompact3dConvention`, optional
            A ready convention, taking precedence over ``prm``.
        **convention_kwargs
            Options for :obj:`Xcompact3dConvention.from_parameters`.

        Returns
        -------
        :obj:`xarray.Dataset`
            The decoded field.

        Raises
        ------
        ValueError
            If neither ``prm`` nor ``convention`` is given and no single parameters file
            is found next to the data folder.
        """
        if convention is None:
            if prm is None:
                prm = parameters_for(filename_or_obj)
            convention = Xcompact3dConvention.from_parameters(prm, **convention_kwargs)
        return super().open_dataset(filename_or_obj, read_specs_getter=convention.reader, drop_variables=drop_variables)

    @staticmethod
    def guess_can_open(filename_or_obj: Any) -> bool:  # type: ignore[override]
        """Claim ``.bin`` files that have exactly one parameters file next to their data folder."""
        try:
            path = Path(filename_or_obj)
        except TypeError:
            return False
        return path.suffix == ".bin" and find_parameters_file(path) is not None
