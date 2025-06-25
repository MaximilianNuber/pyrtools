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

from .function_wrapper import RFunctionWrapper
from .model_wrapper import RModelWrapper

__all__ = ["__version__", "RFunctionWrapper", "RModelWrapper"]
