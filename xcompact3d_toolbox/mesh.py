"""Objects to handle the coordinates and coordinate system.
Note they are an attribute at :obj:`xcompact3d_toolbox.parameters.ParametersExtras`,
so they work together with all the other parameters. They are presented here for reference.
"""

from __future__ import annotations

from enum import IntEnum
from functools import partial
from math import isclose

import numpy as np
import traitlets
from deprecated.sphinx import versionadded

from xcompact3d_toolbox.param import param


@versionadded(version="1.2.0")
class Istret(IntEnum):
    """Mesh refinement type."""

    NO_REFINEMENT = 0
    CENTER_REFINEMENT = 1
    BOTH_SIDES_REFINEMENT = 2
    BOTTOM_REFINEMENT = 3


class Coordinate(traitlets.HasTraits):
    """A coordinate.

    Thanks to traitlets_, the attributes can be type checked, validated and also trigger
    "on change" callbacks. It means that:

    - :obj:`grid_size` is validated to just accept the values expected by XCompact3d
      (see :obj:`xcompact3d_toolbox.mesh.Coordinate.possible_grid_size`);
    - :obj:`delta` is updated after any change on :obj:`grid_size` or :obj:`length`;
    - :obj:`length` is updated after any change on :obj:`delta` (:obj:`grid_size` remains constant);
    - :obj:`grid_size` is reduced automatically by 1 when :obj:`is_periodic` changes to :obj:`True`
      and it is added by 1 when :obj:`is_periodic` changes back to :obj:`False`
      (see :obj:`xcompact3d_toolbox.mesh.Coordinate.possible_grid_size`);

    All these functionalities aim to make a user-friendly interface, where the consistency
    between different coordinate parameters is ensured even when they change at runtime.

    .. _traitlets: https://traitlets.readthedocs.io/en/stable/index.html

    Parameters
    ----------
    length : float
        Length of the coordinate (default is 1.0).
    grid_size : int
        Number of mesh points (default is 17).
    delta : float
        Mesh resolution (default is 0.0625).
    is_periodic : bool
        Specifies if the boundary condition is periodic (True) or not (False) (default is False).

    Notes
    -----
        There is no need to specify both :obj:`length` and :obj:`delta`, because they are
        a function of each other, the missing value is automatically computed from the other.

    Returns
    -------
    :obj:`xcompact3d_toolbox.mesh.Coordinate`
        Coordinate
    """

    length = traitlets.Float(default_value=1.0, min=0.0, max=1e10)
    grid_size = traitlets.Int(default_value=17)
    delta = traitlets.Float(default_value=0.0625, min=0.0)
    is_periodic = traitlets.Bool(default_value=False)

    _sub_grid_size = traitlets.Int(default_value=16)
    _possible_grid_size = traitlets.List(trait=traitlets.Int())

    def __init__(self, **kwargs):
        """Initializes the Coordinate class.

        Parameters
        ----------
        **kwargs
            Keyword arguments for attributes, like :obj:`grid_size`, :obj:`length` and so on.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import Coordinate
        >>> coord = Coordinate(length=1.0, grid_size=9, is_periodic=False)
        """

        self._possible_grid_size = _possible_size_not_periodic
        self.set(**kwargs)

    def __array__(self) -> np.ndarray:
        """This method makes the coordinate automatically work as a numpy
        like array in any function from numpy.

        Returns
        -------
        :obj:`numpy.ndarray`
            A numpy array.

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import Coordinate
        >>> import numpy
        >>> coord = Coordinate(length=1.0, grid_size=9)
        >>> numpy.sin(coord)
        array([0.        , 0.12467473, 0.24740396, 0.36627253, 0.47942554,
               0.58509727, 0.68163876, 0.7675435 , 0.84147098])
        >>> numpy.cos(coord)
        array([1.        , 0.99219767, 0.96891242, 0.93050762, 0.87758256,
                0.81096312, 0.73168887, 0.64099686, 0.54030231])
        """
        return np.linspace(
            start=0.0,
            stop=self.length,
            num=self.grid_size,
            endpoint=not self.is_periodic,
            dtype=param["mytype"],
        )

    def __len__(self):
        """Make the coordinate work with the Python function :obj:`len`.

        Returns
        -------
        int
            Coordinate size (:obj:`grid_size`)

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import Coordinate
        >>> coord = Coordinate(grid_size=9)
        >>> len(coord)
        9
        """
        return self.grid_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(length = {self.length}, "
            f"grid_size = {self.grid_size}, is_periodic = {self.is_periodic})"
        )

    def set(self, **kwargs) -> None:
        """Set a new value for any parameter after the initialization.

        Parameters
        ----------
        **kwargs
            Keyword arguments for attributes, like :obj:`grid_size`, :obj:`length` and so on.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import Coordinate
        >>> coord = Coordinate()
        >>> coord.set(length=1.0, grid_size=9, is_periodic=False)
        """
        if "is_periodic" in kwargs:
            self.is_periodic = kwargs.get("is_periodic")
            del kwargs["is_periodic"]
        for key, arg in kwargs.items():
            if key not in self.trait_names():
                msg = f"{key} is not a valid parameter"
                raise KeyError(msg)
            setattr(self, key, arg)

    @traitlets.validate("grid_size")
    def _validate_grid_size(self, proposal):
        if not _validate_grid_size(proposal.get("value"), self.is_periodic):
            msg = f'{proposal.get("value")} is an invalid value for grid size'
            raise traitlets.TraitError(msg)
        return proposal.get("value")

    @traitlets.observe("is_periodic")
    def _observe_is_periodic(self, change):
        if change.get("new"):
            new_grid = self.grid_size - 1
            self._possible_grid_size = _possible_size_periodic
            self.grid_size = new_grid
        else:
            new_grid = self.grid_size + 1
            self._possible_grid_size = _possible_size_not_periodic
            self.grid_size = new_grid

    @traitlets.observe("_sub_grid_size")
    def _observe_sub_grid_size(self, change):
        new_delta = self.length / change.get("new")
        if new_delta != self.delta:
            self.delta = new_delta

    @traitlets.observe("grid_size")
    def _observe_grid_size(self, change):
        new_sgs = change.get("new") if self.is_periodic else change.get("new") - 1
        if new_sgs != self._sub_grid_size:
            self._sub_grid_size = new_sgs

    @traitlets.observe("length")
    def _observe_length(self, change):
        new_delta = change.get("new") / self._sub_grid_size
        if new_delta != self.delta:
            self.delta = new_delta

    @traitlets.observe("delta")
    def _observe_delta(self, change):
        new_length = change.get("new") * self._sub_grid_size
        if new_length != self.length:
            self.length = new_length

    @property
    def vector(self) -> np.ndarray:
        """Construct a vector with :obj:`numpy.linspace` and return it.

        Returns
        -------
        :obj:`numpy.ndarray`
            Numpy array
        """
        return self.__array__()

    @property
    def size(self) -> int:
        """An alias for :obj:`grid_size`.

        Returns
        -------
        int
            Grid size
        """
        return self.grid_size

    @property
    def possible_grid_size(self) -> list:
        """Possible values for grid size.

        Due to restrictions at the FFT library, they must be equal to:

        .. math::
            n = 2^{1+a} \\times 3^b \\times 5^c,

        if the coordinate is periodic, and:

        .. math::
            n = 2^{1+a} \\times 3^b \\times 5^c + 1,

        otherwise, where :math:`a`, :math:`b` and :math:`c` are non negative integers.

        Additionally, the derivative's stencil imposes that :math:`n \\ge 8` if periodic
        and :math:`n \\ge 9` otherwise.

        Returns
        -------
        list
            Possible values for grid size

        Notes
        -----
        There is no upper limit, as long as the restrictions are satisfied.

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import Coordinate
        >>> coordinate(is_periodic=True).possible_grid_size
        [8, 10, 12, 16, 18, 20, 24, 30, 32, 36, 40, 48, 50, 54, 60, 64, 72, 80,
         90, 96, 100, 108, 120, 128, 144, 150, 160, 162, 180, 192, 200, 216, 240,
         250, 256, 270, 288, 300, 320, 324, 360, 384, 400, 432, 450, 480, 486,
         500, 512, 540, 576, 600, 640, 648, 720, 750, 768, 800, 810, 864, 900,
         960, 972, 1000, 1024, 1080, 1152, 1200, 1250, 1280, 1296, 1350, 1440,
         1458, 1500, 1536, 1600, 1620, 1728, 1800, 1920, 1944, 2000, 2048, 2160,
         2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000, 3072,
         3200, 3240, 3456, 3600, 3750, 3840, 3888, 4000, 4050, 4096, 4320, 4374,
         4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400, 5760, 5832, 6000, 6144,
         6250, 6400, 6480, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
         8192, 8640, 8748, 9000]
        >>> coordinate(is_periodic=False).possible_grid_size
        [9, 11, 13, 17, 19, 21, 25, 31, 33, 37, 41, 49, 51, 55, 61, 65, 73, 81,
         91, 97, 101, 109, 121, 129, 145, 151, 161, 163, 181, 193, 201, 217, 241,
         251, 257, 271, 289, 301, 321, 325, 361, 385, 401, 433, 451, 481, 487,
         501, 513, 541, 577, 601, 641, 649, 721, 751, 769, 801, 811, 865, 901,
         961, 973, 1001, 1025, 1081, 1153, 1201, 1251, 1281, 1297, 1351, 1441,
         1459, 1501, 1537, 1601, 1621, 1729, 1801, 1921, 1945, 2001, 2049, 2161,
         2251, 2305, 2401, 2431, 2501, 2561, 2593, 2701, 2881, 2917, 3001, 3073,
         3201, 3241, 3457, 3601, 3751, 3841, 3889, 4001, 4051, 4097, 4321, 4375,
         4501, 4609, 4801, 4861, 5001, 5121, 5185, 5401, 5761, 5833, 6001, 6145,
         6251, 6401, 6481, 6751, 6913, 7201, 7291, 7501, 7681, 7777, 8001, 8101,
         8193, 8641, 8749, 9001]
        """
        return self._possible_grid_size


class StretchedCoordinate(Coordinate):
    """Another coordinate, as a subclass of :obj:`Coordinate`.
    It includes parameters and methods to handle stretched coordinates,
    which is employed by XCompact3d at the vertical dimension ``y``.

    Parameters
    ----------
    length : float
        Length of the coordinate (default is 1.0).
    grid_size : int
        Number of mesh points (default is 17).
    delta : float
        Mesh resolution (default is 0.0625).
    is_periodic : bool
        Specifies if the boundary condition is periodic (True) or not (False) (default is False).
    istret : int
        Type of mesh refinement:

            * 0 - No refinement (default);
            * 1 - Refinement at the center;
            * 2 - Both sides;
            * 3 - Just near the bottom.
    beta : float
        Refinement parameter.

    Notes
    -----
        There is no need to specify both :obj:`length` and :obj:`delta`, because they are
        a function of each other, the missing value is automatically computed from the other.

    Returns
    -------
    :obj:`xcompact3d_toolbox.mesh.StretchedCoordinate`
        Stretched coordinate
    """

    istret = traitlets.UseEnum(Istret, default_value=0)
    beta = traitlets.Float(default_value=1.0, min=0.0)

    def __repr__(self):
        if self.istret == 0:
            return (
                f"{self.__class__.__name__}(length = {self.length}, grid_size = {self.grid_size}, "
                f"is_periodic = {self.is_periodic})"
            )
        return (
            f"{self.__class__.__name__}(length = {self.length}, grid_size = {self.grid_size}, "
            f"is_periodic = {self.is_periodic}, istret = {self.istret}, beta = {self.beta})"
        )

    def __array__(self):
        """This method makes the coordinate automatically work as a numpy
        like array in any function from numpy.

        Returns
        -------
        :obj:`numpy.ndarray`
            A numpy array.

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import StretchedCoordinate
        >>> import numpy
        >>> coord = StretchedCoordinate(length=1.0, grid_size=9)
        >>> numpy.sin(coord)
        array([0.        , 0.12467473, 0.24740396, 0.36627253, 0.47942554,
               0.58509727, 0.68163876, 0.7675435 , 0.84147098])
        >>> numpy.cos(coord)
        array([1.        , 0.99219767, 0.96891242, 0.93050762, 0.87758256,
                0.81096312, 0.73168887, 0.64099686, 0.54030231])
        """
        if self.istret == Istret.NO_REFINEMENT:
            return super().__array__()
        return _stretching(
            istret=self.istret,
            beta=self.beta,
            yly=self.length,
            my=self._sub_grid_size,
            ny=self.grid_size,
            return_auxiliary_variables=False,
        )

    @traitlets.validate("istret")
    def _validate_istret(self, proposal):
        if proposal.get("value") == Istret.BOTTOM_REFINEMENT and self.is_periodic:
            msg = (
                f"mesh refinement at the bottom (istret={Istret.BOTTOM_REFINEMENT.value}) is not possible when periodic"
            )
            raise traitlets.TraitError(msg)
        return proposal.get("value")

    @traitlets.validate("is_periodic")
    def _validate_is_periodic(self, proposal):
        if proposal.get("value") and self.istret == Istret.BOTTOM_REFINEMENT:
            msg = (
                f"mesh refinement at the bottom (istret={Istret.BOTTOM_REFINEMENT.value}) is not possible when periodic"
            )
            raise traitlets.TraitError(msg)
        return proposal.get("value")


class Mesh3D(traitlets.HasTraits):
    """A three-dimensional coordinate system

    Parameters
    ----------
    x : :obj:`xcompact3d_toolbox.mesh.Coordinate`
        Streamwise coordinate
    y : :obj:`xcompact3d_toolbox.mesh.StretchedCoordinate`
        Vertical coordinate
    z : :obj:`xcompact3d_toolbox.mesh.Coordinate`
        Spanwise coordinate

    Notes
    -----
        :obj:`mesh` is in fact an attribute of :obj:`xcompact3d_toolbox.parameters.ParametersExtras`,
        so there is no need to initialize it manually for most of the common use cases.
        The features of each coordinate are copled by a two-way link with their corresponding
        values at the Parameters class. For instance, the length of each of them is copled to
        :obj:`xlx`, :obj:`yly` and :obj:`zlz`, grid size to :obj:`nx`, :obj:`ny` and :obj:`nz`
        and so on.

    Returns
    -------
    :obj:`xcompact3d_toolbox.mesh.Mesh3D`
        Coordinate system
    """

    x = traitlets.Instance(klass=Coordinate)
    y = traitlets.Instance(klass=StretchedCoordinate)
    z = traitlets.Instance(klass=Coordinate)

    def __init__(self, **kwargs):
        """Initializes the 3DMesh class.

        Parameters
        ----------
        **kwargs
            Keyword arguments for each coordinate (x, y and z), containing a :obj:`dict`
            with the parameters for them, like :obj:`grid_size`, :obj:`length` and so on.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid coordinate.

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import Mesh3D
        >>> mesh = Mesh3D(
        ...     x=dict(length=4.0, grid_size=65, is_periodic=False),
        ...     y=dict(length=1.0, grid_size=17, is_periodic=False, istret=0),
        ...     z=dict(length=1.0, grid_size=16, is_periodic=True),
        ... )
        """

        self.x = Coordinate()
        self.y = StretchedCoordinate()
        self.z = Coordinate()

        self.set(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(x = {self.x}, y = {self.y}, z = {self.z})"

    def __len__(self):
        """Make the coordinate work with the Python function :obj:`len`.

        Returns
        -------
        int
            Mesh size is calculated by multiplying the size of the three coordinates
        """
        return self.size

    def set(self, **kwargs) -> None:
        """Set new values for any of the coordinates after the initialization.

        Parameters
        ----------
        **kwargs
            Keyword arguments for each coordinate (x, y and z), containing a :obj:`dict`
            with the parameters for them, like :obj:`grid_size`, :obj:`length` and so on.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Examples
        --------

        >>> from xcompact3d_toolbox.mesh import Mesh3D
        >>> mesh = Mesh3D()
        >>> mesh.set(
        ...     x=dict(length=4.0, grid_size=65, is_periodic=False),
        ...     y=dict(length=1.0, grid_size=17, is_periodic=False, istret=0),
        ...     z=dict(length=1.0, grid_size=16, is_periodic=True),
        ... )
        """
        for key in kwargs:
            if key in self.trait_names():
                getattr(self, key).set(**kwargs.get(key))
            else:
                msg = f"{key} is not a valid coordinate for Mesh3D"
                raise KeyError(msg)

    def get(self) -> dict:
        """Get the three coordinates in a dictionary, where the keys are their names (x, y and z)
        and the values are their vectors.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Returns
        -------
        :obj:`dict` of :obj:`numpy.ndarray`
            A dict containing the coordinates

        Notes
        -----

        It is an alias for ``Mesh3D.drop(None)``.
        """
        return self.drop(None)

    def drop(self, *args) -> dict:
        """Get the coordinates in a dictionary, where the keys are their names and the values
        are their vectors. It is possible to drop any of the coordinates in case they are
        needed to process planes. For instance:

        * Drop ``x`` if working with ``yz`` planes;
        * Drop ``y`` if working with ``xz`` planes;
        * Drop ``z`` if working with ``xy`` planes.

        Parameters
        ----------
        *args : str or list of str
            Name of the coordinate(s) to be dropped

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Returns
        -------
        :obj:`dict` of :obj:`numpy.ndarray`
            A dict containing the desired coordinates
        """
        for arg in args:
            if not arg:
                continue
            if arg not in self.trait_names():
                msg = f"{arg} is not a valid coordinate for Mesh3D"
                raise KeyError(msg)
        return {d: getattr(self, d).vector for d in self.trait_names() if d not in args}

    def copy(self):
        """Return a copy of the Mesh3D object."""
        return Mesh3D(**{dim: getattr(self, dim).trait_values() for dim in self.trait_names()})

    @property
    def size(self):
        """Mesh size

        Returns
        -------
        int
            Mesh size is calculated by multiplying the size of the three coordinates
        """
        return self.x.size * self.y.size * self.z.size


def _validate_grid_size(grid_size, is_periodic):
    size = grid_size if is_periodic else grid_size - 1

    if size < 8:  # noqa: PLR2004
        return False

    if size % 2 == 0:
        size //= 2

        for num in [2, 3, 5]:
            while True:
                if size % num == 0:
                    size //= num
                else:
                    break

    if size != 1:
        return False
    return True


def _get_possible_grid_values(start: int = 0, end: int = 9002, *, is_periodic: bool) -> list:
    return list(
        filter(
            lambda num: _validate_grid_size(num, is_periodic),
            range(start, end),
        )
    )


_possible_size_periodic = _get_possible_grid_values(is_periodic=True)

_possible_size_not_periodic = _get_possible_grid_values(is_periodic=False)


def _stretching(istret, beta, yly, my, ny, *, return_auxiliary_variables=True):
    close_enough = partial(isclose, abs_tol=1e-8, rel_tol=1e-8)

    yp = np.zeros(ny, dtype=param["mytype"])
    yeta = np.zeros_like(yp)
    ypi = np.zeros_like(yp)
    yetai = np.zeros_like(yp)
    ppy = np.zeros_like(yp)
    pp2y = np.zeros_like(yp)
    pp4y = np.zeros_like(yp)
    ppyi = np.zeros_like(yp)
    pp2yi = np.zeros_like(yp)
    pp4yi = np.zeros_like(yp)
    #
    yinf = -0.5 * yly
    den = 2.0 * beta * yinf
    xnum = -yinf - np.sqrt(np.pi * np.pi * beta * beta + yinf * yinf)
    alpha = np.abs(xnum / den)
    if not close_enough(alpha, 0.0):
        if istret == Istret.CENTER_REFINEMENT:
            yp[0] = 0.0
        if istret == Istret.BOTH_SIDES_REFINEMENT:
            yp[0] = 0.0
        if istret == Istret.CENTER_REFINEMENT:
            yeta[0] = 0.0
        if istret == Istret.BOTH_SIDES_REFINEMENT:
            yeta[0] = -0.5
        if istret == Istret.BOTTOM_REFINEMENT:
            yp[0] = 0.0
        if istret == Istret.BOTTOM_REFINEMENT:
            yeta[0] = -0.5
        for j in range(1, ny):
            if istret == Istret.CENTER_REFINEMENT:
                yeta[j] = j / my
            if istret == Istret.BOTH_SIDES_REFINEMENT:
                yeta[j] = j / my - 0.5
            if istret == Istret.BOTTOM_REFINEMENT:
                yeta[j] = 0.5 * j / my - 0.5
            den1 = np.sqrt(alpha * beta + 1.0)
            xnum = den1 / np.sqrt(alpha / np.pi) / np.sqrt(beta) / np.sqrt(np.pi)
            den = 2.0 * np.sqrt(alpha / np.pi) * np.sqrt(beta) * np.pi * np.sqrt(np.pi)
            den3 = ((np.sin(np.pi * yeta[j])) * (np.sin(np.pi * yeta[j])) / beta / np.pi) + alpha / np.pi
            den4 = 2.0 * alpha * beta - np.cos(2.0 * np.pi * yeta[j]) + 1.0
            xnum1 = (np.arctan(xnum * np.tan(np.pi * yeta[j]))) * den4 / den1 / den3 / den
            cst = np.sqrt(beta) * np.pi / (2.0 * np.sqrt(alpha) * np.sqrt(alpha * beta + 1.0))
            if istret == Istret.CENTER_REFINEMENT:
                if yeta[j] < 0.5:  # noqa: PLR2004
                    yp[j] = xnum1 - cst - yinf
                if close_enough(yeta[j], 0.5):
                    yp[j] = -yinf
                if yeta[j] > 0.5:  # noqa: PLR2004
                    yp[j] = xnum1 + cst - yinf
            elif istret == Istret.BOTH_SIDES_REFINEMENT:
                if yeta[j] < 0.5:  # noqa: PLR2004
                    yp[j] = xnum1 - cst + yly
                if close_enough(yeta[j], 0.5):
                    yp[j] = yly
                if yeta[j] > 0.5:  # noqa: PLR2004
                    yp[j] = xnum1 + cst + yly
            elif istret == Istret.BOTTOM_REFINEMENT:
                if yeta[j] < 0.5:  # noqa: PLR2004
                    yp[j] = (xnum1 - cst + yly) * 2.0
                if close_enough(yeta[j], 0.5):
                    yp[j] = yly * 2.0
                if yeta[j] > 0.5:  # noqa: PLR2004
                    yp[j] = (xnum1 + cst + yly) * 2.0
            else:
                msg = "Unsupported: invalid value for istret"
                raise NotImplementedError(msg)
    if close_enough(alpha, 0.0):
        yp[0] = -1.0e10
        for j in range(1, ny):
            yeta[j] = j / ny
            yp[j] = -beta * np.cos(np.pi * yeta[j]) / np.sin(yeta[j] * np.pi)

    if not close_enough(alpha, 0.0):
        for j in range(ny):
            if istret == Istret.CENTER_REFINEMENT:
                yetai[j] = (j + 0.5) * (1.0 / my)
            if istret == Istret.BOTH_SIDES_REFINEMENT:
                yetai[j] = (j + 0.5) * (1.0 / my) - 0.5
            if istret == Istret.BOTTOM_REFINEMENT:
                yetai[j] = (j + 0.5) * (0.5 / my) - 0.5
            den1 = np.sqrt(alpha * beta + 1.0)
            xnum = den1 / np.sqrt(alpha / np.pi) / np.sqrt(beta) / np.sqrt(np.pi)
            den = 2.0 * np.sqrt(alpha / np.pi) * np.sqrt(beta) * np.pi * np.sqrt(np.pi)
            den3 = ((np.sin(np.pi * yetai[j])) * (np.sin(np.pi * yetai[j])) / beta / np.pi) + alpha / np.pi
            den4 = 2.0 * alpha * beta - np.cos(2.0 * np.pi * yetai[j]) + 1.0
            xnum1 = (np.arctan(xnum * np.tan(np.pi * yetai[j]))) * den4 / den1 / den3 / den
            cst = np.sqrt(beta) * np.pi / (2.0 * np.sqrt(alpha) * np.sqrt(alpha * beta + 1.0))
            if istret == Istret.CENTER_REFINEMENT:
                if yetai[j] < 0.5:  # noqa: PLR2004
                    ypi[j] = xnum1 - cst - yinf
                elif close_enough(yetai[j], 0.5):
                    ypi[j] = 0.0 - yinf
                elif yetai[j] > 0.5:  # noqa: PLR2004
                    ypi[j] = xnum1 + cst - yinf
            elif istret == Istret.BOTH_SIDES_REFINEMENT:
                if yetai[j] < 0.5:  # noqa: PLR2004
                    ypi[j] = xnum1 - cst + yly
                elif close_enough(yetai[j], 0.5):
                    ypi[j] = 0.0 + yly
                elif yetai[j] > 0.5:  # noqa: PLR2004
                    ypi[j] = xnum1 + cst + yly
            elif istret == Istret.BOTTOM_REFINEMENT:
                if yetai[j] < 0.5:  # noqa: PLR2004
                    ypi[j] = (xnum1 - cst + yly) * 2.0
                elif close_enough(yetai[j], 0.5):
                    ypi[j] = (0.0 + yly) * 2.0
                elif yetai[j] > 0.5:  # noqa: PLR2004
                    ypi[j] = (xnum1 + cst + yly) * 2.0

    if close_enough(alpha, 0.0):
        ypi[0] = -1e10
        for j in range(1, ny):
            yetai[j] = j * (1.0 / ny)
            ypi[j] = -beta * np.cos(np.pi * yetai[j]) / np.sin(yetai[j] * np.pi)

    # Mapping!!, metric terms
    if istret != Istret.BOTTOM_REFINEMENT:
        for j in range(ny):
            ppy[j] = yly * (alpha / np.pi + (1.0 / np.pi / beta) * np.sin(np.pi * yeta[j]) * np.sin(np.pi * yeta[j]))
            pp2y[j] = ppy[j] * ppy[j]
            pp4y[j] = -2.0 / beta * np.cos(np.pi * yeta[j]) * np.sin(np.pi * yeta[j])
        for j in range(ny):
            ppyi[j] = yly * (alpha / np.pi + (1.0 / np.pi / beta) * np.sin(np.pi * yetai[j]) * np.sin(np.pi * yetai[j]))
            pp2yi[j] = ppyi[j] * ppyi[j]
            pp4yi[j] = -2.0 / beta * np.cos(np.pi * yetai[j]) * np.sin(np.pi * yetai[j])

    if istret == Istret.BOTTOM_REFINEMENT:
        for j in range(ny):
            ppy[j] = yly * (alpha / np.pi + (1.0 / np.pi / beta) * np.sin(np.pi * yeta[j]) * np.sin(np.pi * yeta[j]))
            pp2y[j] = ppy[j] * ppy[j]
            pp4y[j] = (-2.0 / beta * np.cos(np.pi * yeta[j]) * np.sin(np.pi * yeta[j])) / 2.0
        for j in range(ny):
            ppyi[j] = yly * (alpha / np.pi + (1.0 / np.pi / beta) * np.sin(np.pi * yetai[j]) * np.sin(np.pi * yetai[j]))
            pp2yi[j] = ppyi[j] * ppyi[j]
            pp4yi[j] = (-2.0 / beta * np.cos(np.pi * yetai[j]) * np.sin(np.pi * yetai[j])) / 2.0
    if return_auxiliary_variables:
        return yp, ppy, pp2y, pp4y
    return yp
