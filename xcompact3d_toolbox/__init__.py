from . import tutorial
from ._version import get_versions
from .array import X3dDataArray, X3dDataset
from .genepsi import gene_epsi_3D
from .gui import ParametersGui
from .param import param
from .parameters import Parameters
from .sandbox import init_dataset, init_epsi

__version__ = get_versions()["version"]
del get_versions
