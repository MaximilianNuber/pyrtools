from types import SimpleNamespace
from functools import lru_cache

@lru_cache(maxsize=1)
def lazy_import_r_env() -> SimpleNamespace:
    """Lazy import rpy2 and core R packages.
    
    Lazily import rpy2 entrypoints, core R packages, and custom converters into a
    single namespace for convenient use throughout the package.

    This function delays all rpy2 and R package loading until first use, minimizing
    overhead at import time.

    Returns:
        SimpleNamespace: A namespace with the following attributes:
            ro (module): The main rpy2.robjects module.
            importr (callable): rpy2.robjects.packages.importr function.
            STAP (callable): rpy2.robjects.packages.STAP function.
            utils (RPackage): The imported R 'utils' package.
            methods_pkg (RPackage): The imported R 'methods' package.
            localconverter (callable): Context manager for custom converters.
            default_converter (Converter): Base converter for rpy2.
            pandas2ri (Converter): Converter for pandas objects.
            numpy2ri (Converter): Converter for numpy objects.
            lazy_import_r_packages (callable): Helper to lazily load R packages.
            py2r (callable): Python->R conversion dispatcher.
            r2py (callable): R->Python conversion dispatcher.
            RRuntimeError (Exception): The rpy2.rinterface_lib.embedded.RRuntimeError class.
    """
    # Core rpy2 entrypoints
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr, STAP
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects import conversion
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import get_conversion
    from rpy2.robjects.vectors import ListVector, IntVector, FloatVector, StrVector, BoolVector

    # Your own converters & lazy‐loader
    from .rpy_conversions import _py_to_r, _r_to_py, lazy_import_r_packages

    # Load the R "utils" and "methods" packages for method-discovery
    utils, methods_pkg = lazy_import_r_packages(["utils", "methods"])

    return SimpleNamespace(
        ro=ro,
        importr=importr,
        STAP=STAP,
        utils=utils,
        methods_pkg=methods_pkg,
        localconverter=localconverter,
        default_converter=default_converter,
        pandas2ri=pandas2ri,
        numpy2ri=numpy2ri,
        get_conversion=get_conversion,
        ListVector=ListVector,
        IntVector=IntVector,
        FloatVector=FloatVector,
        StrVector=StrVector,
        BoolVector=BoolVector,
        lazy_import_r_packages=lazy_import_r_packages,
        py2r=_py_to_r,
        r2py=_r_to_py,
        RRuntimeError=RRuntimeError,
    )

def clear_lazy_import_r_env_cache():
    """Clea the cached R namespace (for debugging or forced refresh)."""

    lazy_import_r_env.cache_clear()