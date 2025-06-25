import functools
import inspect
from typing import TYPE_CHECKING, Any, Dict, Tuple, Optional, Union
from types import SimpleNamespace

# Lazy importer and exceptions
from .r_env import lazy_import_r_env
# from .exceptions import RFunctionNotFoundError, RPackageNotLoadedError

if TYPE_CHECKING:
    import rpy2.robjects as ro
    import pandas as pd
    import numpy as np


class RModelWrapper:
    """Wrap an R model object (S3 list-like or S4) for Python access.

    Provides attribute access to R “slots” and a generic `apply` method
    to call any R function (generics or otherwise) with the model
    as the first argument.
    """

    def __init__(
        self,
        r_model_object: Any
    ):
        """Initialize the wrapper and detect slots and methods.

        Args:
            r_model_object: An rpy2 R object representing a fitted model.
                Must be either an S3 list-like (`ListVector`) or an S4 (`RS4`).

        Raises:
            TypeError: If `r_model_object` is not a supported R type.
        """
        # Lazily load the shared R environment and helpers
        self._r: SimpleNamespace = lazy_import_r_env()
        ro = self._r.ro

        # Aliases for R object types
        ListVector = ro.vectors.ListVector
        RS4        = ro.methods.RS4

        # Verify we have an R list (S3) or S4
        if not isinstance(r_model_object, (ListVector, RS4)):
            expected = (
                "rpy2_vectors.ListVector or rpy2_methods.RS4"
                if TYPE_CHECKING
                else "an R S3 list-like or S4 object"
            )
            raise TypeError(f"r_model_object must be {expected}")

        self._r_model = r_model_object
        self.r_slot_names: Tuple[str, ...] = ()
        self._populate_slots()

        # Holds the R generic functions we found
        self._r_methods: Dict[str, 'ro.Function'] = {}
        self._populate_r_methods()

        # Dynamically bind each detected method to this instance
        for name in list(self._r_methods):
            bound = functools.partial(self.call_r_method, name)
            setattr(self, name, bound)

    def _populate_slots(self):
        """Detect and store S3 names or S4 slot names."""
        ro = self._r.ro
        ListVector = self._r.ro.vectors.ListVector
        RS4        = self._r.ro.methods.RS4

        if isinstance(self._r_model, ListVector):
            self.r_slot_names = tuple(self._r_model.names)
        elif isinstance(self._r_model, RS4):
            self.r_slot_names = tuple(self._r_model.slotnames())

    # def get_r_slot(self, slot_name: str, convert: bool = True) -> Any:
    #     """Access a named slot by name, optionally converting to Python."""
    #     if slot_name not in self.r_slot_names:
    #         raise AttributeError(f"Slot '{slot_name}' not found on R object.")

    #     # Choose S3 or S4 accessor
    #     if hasattr(self._r_model, 'rx2'):
    #         r_val = self._r_model.rx2(slot_name)
    #     else:
    #         r_val = self._r_model.slots[slot_name]

    #     if convert:
    #         with self._r.localconverter(self._r.default_converter):
    #             return self._r.ro.conversion.rpy2py(r_val)
    #     return r_val
    def get_r_slot(self, slot_name: str, convert: bool = True) -> Any:
        """Retrieve a named slot from the R model.

        Args:
            slot_name: Name of the slot to retrieve.
            convert: If True, convert the R object to Python using `r2py`.

        Returns:
            The slot value, either as a raw R object (if `convert=False`)
            or converted to a Python type.

        Raises:
            AttributeError: If the slot name is not present on the model.
        """
        if slot_name not in self.r_slot_names:
            raise AttributeError(f"Slot '{slot_name}' not found on R object.")

        # pick S3 or S4
        if hasattr(self._r_model, 'rx2'):
            r_val = self._r_model.rx2(slot_name)
        else:
            r_val = self._r_model.slots[slot_name]

        if convert:
            return self._r.r2py(r_val)
        return r_val

    def __getattr__(self, name: str) -> Any:
        """Expose slots and detected methods as attributes."""
        # Expose slots as attributes
        if name in self.r_slot_names:
            return self.get_r_slot(name)
        # Expose detected R methods
        if name in self._r_methods:
            return functools.partial(self.call_r_method, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        """Return a brief representation including the R class(es)."""
        cls = getattr(self._r_model, 'rclass', None)
        classes = ','.join(cls) if cls is not None else 'Unknown'
        return f"<RModelWrapper wrapping R classes: [{classes}]>"

    def _r_get_s3_method(self, generic: str, class_name: str) -> Optional['ro.Function']:
        """Try to fetch an S3 method for this class.

        Args:
            generic: The generic function name (e.g. "summary").
            class_name: The R class name (e.g. "lm").

        Returns:
            The rpy2 Function if found, or None otherwise.
        """
        try:
            method = self._r.utils.getS3method(
                generic,
                class_name,
                optional=self._r.ro.BoolVector([True])
            )
            return None if method is self._r.ro.NULL else method
        except self._r.RRuntimeError:
            return None

    def _r_get_s4_method(self, generic: str, class_name: str) -> Optional['ro.Function']:
        """Try to fetch an S4 method for this class.

        Args:
            generic: The generic function name.
            class_name: The R class name.

        Returns:
            The rpy2 Function if found, or None otherwise.
        """
        try:
            sig = self._r.ro.StrVector([class_name])
            method = self._r.methods_pkg.getMethod(
                generic,
                sig,
                optional=self._r.ro.BoolVector([True])
            )
            return None if method is self._r.ro.NULL else method
        except self._r.RRuntimeError:
            return None

    def _populate_r_methods(self):
        """Detect a core set of R generics implemented for this model."""
        generics = [
            'print', 'summary', 'predict', 'plot', 'coef',
            'residuals', 'fitted', 'vcov', 'model.frame',
            'model.matrix', 'AIC', 'BIC'
        ]
        classes = getattr(self._r_model, 'rclass', ())
        primary = classes[0] if classes else None

        for generic in generics:
            fn = None
            if primary:
                if isinstance(self._r_model, self._r.ro.vectors.ListVector):
                    fn = self._r_get_s3_method(generic, primary)
                else:
                    fn = self._r_get_s4_method(generic, primary)
            if fn:
                self._r_methods[generic] = fn

    def call_r_method(self, generic: str, *args: Any, **kwargs: Any) -> Any:
        """Call a detected R method on this model.

        Args:
            generic: Name of the generic to call.
            *args: Positional arguments (after the model object).
            **kwargs: Keyword arguments for the R function.

        Returns:
            The result converted to Python via `r2py`.

        Raises:
            AttributeError: If the generic was not detected.
            RuntimeError: If the R function call fails.
        """
        if generic not in self._r_methods:
            raise AttributeError(
                f"R generic '{generic}' not available; found: {list(self._r_methods)}"
            )
        fn = self._r_methods[generic]

        # Prepare arguments
        r_args = [self._r_model]
        r_kwargs = {}
        with self._r.localconverter(self._r.default_converter):
            for a in args:
                r_args.append(self._r.ro.conversion.py2rpy(a))
            for k, v in kwargs.items():
                r_kwargs[k] = self._r.ro.conversion.py2rpy(v)

        # Call and convert result
        try:
            res = fn(*r_args, **r_kwargs)
        except self._r.RRuntimeError as e:
            raise RuntimeError(f"Error in R method '{generic}': {e}")
        return self._r.ro.conversion.rpy2py(res)
    
    def apply(
        self,
        r_function: Union[str, "ro.Function"],
        package: str = None,
        convert_output: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Apply an R function to this model, using the model as first argument.

        This bypasses auto-detected generics and lets you call any R function
        (generic or not) by name or by rpy2 Function object. The R object is 
        inserted as the first argument to the function, allowing it to
        operate on the model directly and making it particularly useful for 
        generic methods in R which dispatch on the first argument to the generic function/method.

        Args:
            r_function: Name of the R function or an rpy2 Function object.
            package: If `r_function` is a string in an R package that needs
                loading, specify its package name here.
            convert_output: If True, convert the R result to Python via `r2py`.
            *args: Additional positional arguments after the model object.
            **kwargs: Additional keyword arguments for the R function.

        Returns:
            The result of the R call, converted to Python if requested.
        """
        # 1) build a one‐off wrapper around that function
        from .function_wrapper import RFunctionWrapper

        fn_wrap = RFunctionWrapper(r_function, package=package)
        fn = fn_wrap.get_python_function(convert_output=convert_output)

        # 2) NOTE: for positional dispatch we pass model as first arg
        return fn(self._r_model, *args, **kwargs)