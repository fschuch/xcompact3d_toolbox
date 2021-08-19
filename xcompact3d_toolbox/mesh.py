import warnings
from collections import OrderedDict

import numpy as np
import traitlets

from .param import param


class Coordinate(traitlets.HasTraits):
    length = traitlets.Float(default_value=1.0, min=0.0, max=1e10)
    grid_size = traitlets.Int(default_value=17)
    delta = traitlets.Float(default_value=0.0625, min=0.0)
    is_periodic = traitlets.Bool(default_value=False)
    _sub_grid_size = traitlets.Int(default_value=16)

    def __init__(self, **kwargs):

        self.set(**kwargs)

    def __array__(self):
        return np.linspace(
            start=0.0,
            stop=self.length,
            num=self.grid_size,
            endpoint=not self.is_periodic,
            dtype=param["mytype"],
        )

    def __len__(self):
        return self.grid_size

    def __repr__(self):
        return f"{self.__class__.__name__}(length = {self.length}, grid_size = {self.grid_size}, is_periodic = {self.is_periodic})"

    def get_possible_grid_size_values(self, start: int = 0, stop: int = 9002) -> list:
        return list(
            filter(
                lambda num: _validate_grid_size(num, self.is_periodic),
                range(start, stop),
            )
        )

    def set(self, **kwargs) -> None:
        """[summary]
        """
        if "is_periodic" in kwargs:
            self.is_periodic = kwargs.get("is_periodic")
            del kwargs["is_periodic"]
        for key, arg in kwargs.items():
            if key not in self.trait_names():
                warnings.warn(f"{key} is not a valid parameter and was not loaded")
            setattr(self, key, arg)

    @traitlets.validate("grid_size")
    def _validate_grid_size(self, proposal):
        if not _validate_grid_size(proposal.get("value"), self.is_periodic):
            raise traitlets.TraitError(
                f'{proposal.get("value")} is an invalid value for grid size'
            )
        return proposal.get("value")

    @traitlets.observe("is_periodic")
    def _observe_is_periodic(self, change):
        if change.get("new"):
            self.grid_size -= 1
        else:
            self.grid_size += 1

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
    def vector(self):
        return self.__array__()

    @property
    def size(self):
        return self.grid_size


class StretchedCoordinate(Coordinate):
    istret = traitlets.Int(
        default_value=0,
        min=0,
        max=3,
        help="type of mesh refinement (0:no, 1:center, 2:both sides, 3:bottom)",
    )
    beta = traitlets.Float(default_value=1.0, min=0.0, help="Refinement parameter")

    def __repr__(self):
        if self.istret == 0:
            return f"{self.__class__.__name__}(length = {self.length}, grid_size = {self.grid_size}, is_periodic = {self.is_periodic})"
        return f"{self.__class__.__name__}(length = {self.length}, grid_size = {self.grid_size}, is_periodic = {self.is_periodic}, istret = {self.istret}, beta = {self.beta})"

    def __array__(self):
        if self.istret == 0:
            return super().__array__()
        return stretching(
            istret=self.istret,
            beta=self.beta,
            yly=self.length,
            my=self._sub_grid_size,
            ny=self.grid_size,
            return_auxiliar_variables=False,
        )

    @traitlets.validate("istret")
    def _validate_istret(self, proposal):
        if proposal.get("value") == 3 and self.is_periodic:
            raise traitlets.TraitError(
                f"mesh refinement at the bottom (istret=3) is not possible when periodic"
            )
        return proposal.get("value")

    @traitlets.validate("is_periodic")
    def _validate_is_periodic(self, proposal):
        if proposal.get("value") and self.istret == 3:
            raise traitlets.TraitError(
                f"mesh refinement at the bottom (istret=3) is not possible when periodic"
            )
        return proposal.get("value")


class Mesh3D(traitlets.HasTraits):
    x = traitlets.Instance(klass=Coordinate)
    y = traitlets.Instance(klass=StretchedCoordinate)
    z = traitlets.Instance(klass=Coordinate)

    def __init__(self, **kwargs):

        self.x = Coordinate(**kwargs.get("x", {}))
        self.y = StretchedCoordinate(**kwargs.get("y", {}))
        self.z = Coordinate(**kwargs.get("z", {}))

        for key in kwargs.keys():
            if key not in self.trait_names():
                warnings.warn(f"{key} is not a valid key parameter for Mesh3D")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    x = {self.x},\n"
            f"    y = {self.y},\n"
            f"    z = {self.z},\n"
            ")"
        )

    def __len__(self):
        return self.size

    def get(self) -> dict:
        return self.drop(None)

    def drop(self, *args) -> dict:
        for arg in args:
            if not arg:
                continue
            if arg not in self.trait_names():
                warnings.warn(f"{arg} is not a valid key parameter for Mesh3D")
        return OrderedDict(
            {
                dir: getattr(self, dir).vector
                for dir in self.trait_names()
                if dir not in args
            }
        )

    def copy(self):
        return Mesh3D(
            **{dim: getattr(self, dim).trait_values() for dim in self.trait_names()}
        )

    @property
    def size(self):
        return self.x.size * self.y.size * self.z.size


def _validate_grid_size(grid_size, is_periodic):

    size = grid_size if is_periodic else grid_size - 1

    if size < 8:
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


def stretching(istret, beta, yly, my, ny, return_auxiliar_variables=True):

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
    xcx = 1.0 / beta / alpha
    if alpha != 0.0:
        if istret == 1:
            yp[0] = 0.0
        if istret == 2:
            yp[0] = 0.0
        if istret == 1:
            yeta[0] = 0.0
        if istret == 2:
            yeta[0] = -0.5
        if istret == 3:
            yp[0] = 0.0
        if istret == 3:
            yeta[0] = -0.5
        for j in range(1, ny):
            if istret == 1:
                yeta[j] = j / my
            if istret == 2:
                yeta[j] = j / my - 0.5
            if istret == 3:
                yeta[j] = 0.5 * j / my - 0.5
            den1 = np.sqrt(alpha * beta + 1.0)
            xnum = den1 / np.sqrt(alpha / np.pi) / np.sqrt(beta) / np.sqrt(np.pi)
            den = 2.0 * np.sqrt(alpha / np.pi) * np.sqrt(beta) * np.pi * np.sqrt(np.pi)
            den3 = (
                (np.sin(np.pi * yeta[j])) * (np.sin(np.pi * yeta[j])) / beta / np.pi
            ) + alpha / np.pi
            den4 = 2.0 * alpha * beta - np.cos(2.0 * np.pi * yeta[j]) + 1.0
            xnum1 = (
                (np.arctan(xnum * np.tan(np.pi * yeta[j]))) * den4 / den1 / den3 / den
            )
            cst = (
                np.sqrt(beta)
                * np.pi
                / (2.0 * np.sqrt(alpha) * np.sqrt(alpha * beta + 1.0))
            )
            if istret == 1:
                if yeta[j] < 0.5:
                    yp[j] = xnum1 - cst - yinf
                if yeta[j] == 0.5:
                    yp[j] = -yinf
                if yeta[j] > 0.5:
                    yp[j] = xnum1 + cst - yinf
            elif istret == 2:
                if yeta[j] < 0.5:
                    yp[j] = xnum1 - cst + yly
                if yeta[j] == 0.5:
                    yp[j] = yly
                if yeta[j] > 0.5:
                    yp[j] = xnum1 + cst + yly
            elif istret == 3:
                if yeta[j] < 0.5:
                    yp[j] = (xnum1 - cst + yly) * 2.0
                if yeta[j] == 0.5:
                    yp[j] = yly * 2.0
                if yeta[j] > 0.5:
                    yp[j] = (xnum1 + cst + yly) * 2.0
            else:
                raise NotImplementedError("Unsupported: invalid value for istret")
    if alpha == 0.0:
        yp[0] = -1.0e10
        for j in range(1, ny):
            yeta[j] = j / ny
            yp[j] = -beta * np.cos(np.pi * yeta[j]) / np.sin(yeta[j] * np.pi)

    if alpha != 0.0:
        for j in range(ny):
            if istret == 1:
                yetai[j] = (j + 0.5) * (1.0 / my)
            if istret == 2:
                yetai[j] = (j + 0.5) * (1.0 / my) - 0.5
            if istret == 3:
                yetai[j] = (j + 0.5) * (0.5 / my) - 0.5
            den1 = np.sqrt(alpha * beta + 1.0)
            xnum = den1 / np.sqrt(alpha / np.pi) / np.sqrt(beta) / np.sqrt(np.pi)
            den = 2.0 * np.sqrt(alpha / np.pi) * np.sqrt(beta) * np.pi * np.sqrt(np.pi)
            den3 = (
                (np.sin(np.pi * yetai[j])) * (np.sin(np.pi * yetai[j])) / beta / np.pi
            ) + alpha / np.pi
            den4 = 2.0 * alpha * beta - np.cos(2.0 * np.pi * yetai[j]) + 1.0
            xnum1 = (
                (np.arctan(xnum * np.tan(np.pi * yetai[j]))) * den4 / den1 / den3 / den
            )
            cst = (
                np.sqrt(beta)
                * np.pi
                / (2.0 * np.sqrt(alpha) * np.sqrt(alpha * beta + 1.0))
            )
            if istret == 1:
                if yetai[j] < 0.5:
                    ypi[j] = xnum1 - cst - yinf
                elif yetai[j] == 0.5:
                    ypi[j] = 0.0 - yinf
                elif yetai[j] > 0.5:
                    ypi[j] = xnum1 + cst - yinf
            elif istret == 2:
                if yetai[j] < 0.5:
                    ypi[j] = xnum1 - cst + yly
                elif yetai[j] == 0.5:
                    ypi[j] = 0.0 + yly
                elif yetai[j] > 0.5:
                    ypi[j] = xnum1 + cst + yly
            elif istret == 3:
                if yetai[j] < 0.5:
                    ypi[j] = (xnum1 - cst + yly) * 2.0
                elif yetai[j] == 0.5:
                    ypi[j] = (0.0 + yly) * 2.0
                elif yetai[j] > 0.5:
                    ypi[j] = (xnum1 + cst + yly) * 2.0

    if alpha == 0.0:
        ypi[0] = -1e10
        for j in range(1, ny):
            yetai[j] = j * (1.0 / ny)
            ypi[j] = -beta * np.cos(np.pi * yetai[j]) / np.sin(yetai[j] * np.pi)

    # Mapping!!, metric terms
    if istret != 3:
        for j in range(ny):
            ppy[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yeta[j])
                * np.sin(np.pi * yeta[j])
            )
            pp2y[j] = ppy[j] * ppy[j]
            pp4y[j] = -2.0 / beta * np.cos(np.pi * yeta[j]) * np.sin(np.pi * yeta[j])
        for j in range(ny):
            ppyi[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yetai[j])
                * np.sin(np.pi * yetai[j])
            )
            pp2yi[j] = ppyi[j] * ppyi[j]
            pp4yi[j] = -2.0 / beta * np.cos(np.pi * yetai[j]) * np.sin(np.pi * yetai[j])

    if istret == 3:
        for j in range(ny):
            ppy[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yeta[j])
                * np.sin(np.pi * yeta[j])
            )
            pp2y[j] = ppy[j] * ppy[j]
            pp4y[j] = (
                -2.0 / beta * np.cos(np.pi * yeta[j]) * np.sin(np.pi * yeta[j])
            ) / 2.0
        for j in range(ny):
            ppyi[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yetai[j])
                * np.sin(np.pi * yetai[j])
            )
            pp2yi[j] = ppyi[j] * ppyi[j]
            pp4yi[j] = (
                -2.0 / beta * np.cos(np.pi * yetai[j]) * np.sin(np.pi * yetai[j])
            ) / 2.0
    if return_auxiliar_variables:
        return yp, ppy, pp2y, pp4y
    return yp
