import functools
import re
from types import MappingProxyType
import inspect
# from rpy2.rinterface_lib.embedded import RRuntimeError

from .r_env import lazy_import_r_env
from .exceptions import RFunctionNotFoundError, RPackageNotLoadedError

def _sanitize_arg_name(name: str) -> str:
    """Convert an R argument name into a valid Python identifier.

    Replaces any character that is not alphanumeric or underscore with
    an underscore, and prefixes the result with an underscore if it
    begins with a digit.

    Args:
        name: The original R argument name.

    Returns:
        A valid Python identifier derived from `name`.
    """
    py = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if re.match(r"^[0-9]", py):
        py = f"_{py}"
    return py

class RFunctionWrapper:
    """Wrap an R function to produce a Python-callable with automatic conversion.

    Sanitizes R argument names for Python, applies Python→R conversion
    on inputs and R→Python conversion on outputs, and allows optional
    control over output conversion. Uses a shared, lazily initialized
    rpy2 environment.

    Attributes:
        _shared_env: Class-level cache for the rpy2 environment namespace.
    """
    # Shared environment across all instances
    _shared_env = None

    @classmethod
    def _get_env(cls):
        """Get or initialize the shared rpy2 environment namespace.

        Returns:
            A SimpleNamespace containing rpy2 entry points, converters,
            and helper functions.
        """
        if cls._shared_env is None:
            cls._shared_env = lazy_import_r_env()
        return cls._shared_env

    def __init__(
        self,
        r_function,
        package: str = None
    ):
        """Initialize the wrapper around an R function.

        Args:
            r_function: Either the name of an R function (as a string)
                or an rpy2 Function object.
            package: Optional name of the R package to load if
                `r_function` is given as a string.

        Raises:
            RPackageNotLoadedError: If `package` is provided but cannot be loaded.
            RFunctionNotFoundError: If the named R function cannot be found.
        """
        # Use shared or lazy init of R environment
        self._env = self._get_env()
        ro = self._env.ro
        self._py2r = self._env.py2r
        self._r2py = self._env.r2py

        # Default: convert R output back to Python
        self._convert_output = True

        # Resolve string name → rpy2 Function
        if isinstance(r_function, str):
            if package:
                try:
                    self._env.lazy_import_r_packages(package)
                except Exception as e:
                    raise RPackageNotLoadedError(
                        f"Could not load R package '{package}': {e}"
                    )
            try:
                self._rfn = ro.r[r_function]
            except Exception as e:
                raise RFunctionNotFoundError(
                    f"R function '{r_function}' not found: {e}"
                )
            self._name = r_function
        else:
            self._rfn = r_function
            self._name = getattr(r_function, 'name', '<unnamed>')

        # Inspect R formals
        formals = ro.r['formals'](self._rfn)
        r_names = list(formals.names) if hasattr(formals, 'names') else []

        # Detect and strip ellipsis
        self._has_ellipsis = '...' in r_names
        if self._has_ellipsis:
            r_names.remove('...')

        # Map Python-safe names to original R names
        py_map = {}
        for rname in r_names:
            py = _sanitize_arg_name(rname)
            orig_py = py
            i = 1
            while py in py_map:
                py = f"{orig_py}_{i}"
                i += 1
            py_map[py] = rname

        self._py_to_rname = MappingProxyType(py_map)
        self._r_to_pyname = {v: k for k, v in py_map.items()}
        self._py_args = list(py_map.keys())

        # Default converters for each R argument
        self._converters = {rname: self._py2r for rname in r_names}

    def set_arg_converter(self, arg_name: str, converter: callable):
        """Override the converter used for a specific argument.

        Args:
            arg_name: The Python-safe or original R name of the argument.
            converter: A function that takes a Python object and returns
                an rpy2.robjects R object.

        Raises:
            ValueError: If `arg_name` is not among the recognized arguments.
        """
        # Normalize to R name
        # if arg_name in self._py_to_rname:
        #     rname = self._py_to_rname[arg_name]
        # else:
        #     rname = arg_name
        # if rname not in self._converters:
        #     raise ValueError(
        #         f"Unknown argument '{arg_name}'. Available: {list(self._converters)}"
        #     )
        # self._converters[rname] = converter
        """Override the converter for either a formal or, if `...` is present, any keyword."""
        # 1) Map a sanitized-Python name → the real R name if it was a formal
        if arg_name in self._py_to_rname:
            rname = self._py_to_rname[arg_name]
        else:
            # if it's not one of the true formals, only allow it if we saw "..."
            if not self._has_ellipsis:
                raise ValueError(
                    f"Cannot set converter for '{arg_name}': not a formal and no `...`"
                )
            rname = arg_name

        # 2) Register (or overwrite) the converter
        self._converters[rname] = converter

    def set_convert_output(self, convert: bool):
        """Enable or disable Python conversion of the R function’s output.

        Args:
            convert: If True, outputs will be converted to Python types.
        """
        self._convert_output = bool(convert)

    def get_python_function(
        self,
        convert_output: bool = None
    ) -> callable:
        """Generate a Python function wrapping the R function.

        The returned function will have a sanitized signature matching
        the R function’s formal arguments and will automatically
        convert inputs and, by default, outputs.

        Args:
            convert_output: If provided, overrides the instance default
                conversion behavior.

        Returns:
            A Python callable that wraps the R function.
        """
        # Decide final conversion behavior
        use_convert = self._convert_output if convert_output is None else bool(convert_output)

        # Build Python signature from sanitized names
        params = [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect._empty
            )
            for name in self._py_args
        ]
        if self._has_ellipsis:
            params.append(
                inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)
            )
        sig = inspect.Signature(params)

        @functools.wraps(self._rfn)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            r_kwargs = {}

            # Convert fixed arguments
            for py_name, r_name in self._py_to_rname.items():
                if py_name in bound.arguments:
                    val = bound.arguments[py_name]
                    if isinstance(val, self._env.ro.RObject):
                        # Already an R object, no conversion needed
                        r_kwargs[r_name] = val
                    else:
                        r_kwargs[r_name] = self._converters[r_name](val)

            # Handle extras via ellipsis
            if self._has_ellipsis:
                extra = bound.arguments.get('kwargs', {})
                for k, v in extra.items():
                    if isinstance(v, self._env.ro.RObject):
                        r_kwargs[k] = v
                    else:
                        conv = self._converters.get(k, self._py2r)
                        r_kwargs[k] = conv(v)

            # Call R
            try:
                res = self._rfn(**r_kwargs)
            except self._env.RRuntimeError as e:
                raise RuntimeError(f"Error calling R '{self._name}': {e}")

            # Return either raw R object or converted Python object
            return self._r2py(res) if use_convert else res

        # Attach signature and doc
        wrapper.__signature__ = sig
        wrapper.__doc__ = (
            f"Wrapper for R '{self._name}'. Args: {', '.join(self._py_args)}"
            + (", plus ..." if self._has_ellipsis else "")
        )
        return wrapper