"""
pyrtools: Python R tools—wrappers for rpy2 functions and R models.

Provides two main classes:

- RFunctionWrapper: wrap any R function as a Python callable.
- RModelWrapper: wrap a fitted R model object and apply R methods.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # When running in editable/dev mode before installation
    __version__ = "0.0.0+dev"

from .r_env import lazy_import_r_env

from .function_wrapper import RFunctionWrapper
from .model_wrapper import RModelWrapper
from .r_types import RMatrix
from .r_types import RSparseMatrix
from .r_types import RList
from .r_types import RDataFrame
from .r_types import RInteger
from .r_types import RNumeric
from .r_types import RCharacter
from .r_types import RLogical
from .r_types import RFactor


__all__ = [
    "__version__", 
    "lazy_import_r_env", 
    "RFunctionWrapper", 
    "RModelWrapper", 
    "RMatrix", 
    "RSparseMatrix", 
    "RList",
    "RDataFrame",
    "RInteger",
    "RNumeric",
    "RCharacter",
    "RLogical",
    "RFactor"
]

