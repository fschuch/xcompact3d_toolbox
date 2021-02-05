from .array import X3dDataset, X3dDataArray
from .genepsi import gene_epsi_3D
from .gui import ParametersGui
from .io import readfield, read_all, write_xdmf
from .param import param
from .parameters import Parameters

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
