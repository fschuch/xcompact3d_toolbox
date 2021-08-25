from .array import X3dDataset, X3dDataArray
from .param import param
from .parameters import Parameters

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
