from .array import X3dDataset, X3dDataArray
from .param import mytype
from .parameters import Parameters
from .genepsi import gene_epsi_3D
from .io import readfield, read_all, write_xdmf

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
