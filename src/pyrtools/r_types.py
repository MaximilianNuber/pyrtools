# src/pyrtools/matrix_wrapper.py

import warnings
from typing import Any, Optional, Sequence, Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from .r_env import lazy_import_r_env

# from .r_env import lazy_import_r_env


class RObjectWrapper(ABC):
    """
    Abstract base for any class that wraps an rpy2 R object.

    Subclasses must implement the `r_object` property to expose
    the raw `rpy2.robjects` instance.
    """

    @property
    @abstractmethod
    def r_object(self) -> Any:
        """
        The underlying raw rpy2 object (e.g. IntVector, Matrix, ListVector, etc.).

        Returns:
            Any: an rpy2.robjects.* instance
        """
        ...


# from .base import RObjectWrapper
# from .r_env import lazy_import_r_env


class RInteger(RObjectWrapper):
    """
    A thin wrapper around an R integer vector (IntVector).

    Provides methods for constructing from NumPy arrays or Python lists,
    with optional overflow warnings for 64-bit integers.
    """

    def __init__(self, rvec: Any):
        """
        Initialize and validate an R integer vector.

        Args:
            rvec: A raw rpy2 IntVector instance.

        Raises:
            TypeError: if `rvec` is not an R integer vector.
        """
        renv = lazy_import_r_env()
        cls_vec = renv.ro.baseenv["class"](rvec)
        cls_py = renv.r2py(cls_vec)
        # normalize to a sequence of names
        if isinstance(cls_py, str):
            cls_list = [cls_py]
        else:
            cls_list = list(cls_py)
        if "integer" not in cls_list:
            raise TypeError(f"Expected R integer vector, got R classes {cls_list!r}")
        self._rvec = rvec

    @property
    def r_object(self) -> Any:
        """
        The underlying raw rpy2 IntVector.

        Returns:
            Any: the rpy2.robjects.vectors.IntVector instance
        """
        return self._rvec

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "RInteger":
        """
        Construct an R integer vector from a NumPy array.

        Args:
            arr: A NumPy array of integer dtype.

        Returns:
            RInteger: wrapper around the corresponding R IntVector.

        Raises:
            TypeError: if `arr` is not of integer dtype.
        Warns:
            UserWarning: if `arr.dtype` is int64, since R integer may overflow.
        """
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"Array dtype must be integer, got {arr.dtype}")

        if arr.dtype == np.int64:
            warnings.warn(
                "NumPy int64 may overflow when converted to R integer (32-bit).",
                UserWarning,
            )

        flat = arr.ravel().tolist()
        renv = lazy_import_r_env()
        IntVector = renv.IntVector
        rvec = IntVector(flat)
        return cls(rvec)

    @classmethod
    def from_list(cls, xs: Sequence[int]) -> "RInteger":
        """
        Construct an R integer vector from a Python sequence of ints.

        Args:
            xs: Sequence of Python ints.

        Returns:
            RInteger: wrapper around the corresponding R IntVector.

        Raises:
            TypeError: if any element of `xs` is not an int.
        """
        for i, x in enumerate(xs):
            if not isinstance(x, (int, np.integer)):
                raise TypeError(f"Element at index {i} is not an integer: {x!r}")
        renv = lazy_import_r_env()
        IntVector = renv.IntVector
        rvec = IntVector(list(xs))
        return cls(rvec)

    def to_list(self) -> list[int]:
        """
        Convert this R integer vector to a Python list of ints.

        Returns:
            list[int]: the integer elements.
        """
        return list(self._rvec)

    '''
    def to_numpy(self) -> np.ndarray:
        """
        Convert this R integer vector to a NumPy array of dtype int32.

        Returns:
            np.ndarray: the integer elements as a 1-D array.
        """
        renv = lazy_import_r_env()
        from rpy2.robjects import conversion
        from rpy2.robjects.conversion import localconverter, default_converter

        with localconverter(default_converter + renv.numpy2ri.converter):
            arr = conversion.rpy2py(self._rvec)
        return np.asarray(arr, dtype=np.int32)
    '''

    def to_numpy(self) -> np.ndarray:
        """Convert this R integer vector to a NumPy array."""
        return np.asarray(self._rvec)


class RNumeric(RObjectWrapper):
    """
    Thin wrapper around an R numeric (double) vector.
    """

    def __init__(self, rvec: Any):
        renv = lazy_import_r_env()
        cls_vec = renv.ro.baseenv["class"](rvec)
        cls_py = renv.r2py(cls_vec)
        cls_list = [cls_py] if isinstance(cls_py, str) else list(cls_py)
        if "numeric" not in cls_list and "double" not in cls_list:
            raise TypeError(f"Expected R numeric vector, got R classes {cls_list!r}")
        self._rvec = rvec

    @property
    def r_object(self) -> Any:
        """The underlying rpy2 FloatVector."""
        return self._rvec

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "RNumeric":
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError(f"Array dtype must be float, got {arr.dtype}")
        flat = arr.ravel().tolist()
        renv = lazy_import_r_env()
        FloatVector = renv.ro.vectors.FloatVector
        return cls(FloatVector(flat))

    @classmethod
    def from_list(cls, xs: Sequence[Union[float, int]]) -> "RNumeric":
        for i, x in enumerate(xs):
            if not isinstance(x, (float, int, np.floating, np.integer)):
                raise TypeError(f"Element at index {i} is not numeric: {x!r}")
        renv = lazy_import_r_env()
        FloatVector = renv.ro.vectors.FloatVector
        return cls(FloatVector(list(xs)))

    def to_numpy(self) -> np.ndarray:
        """Convert to a NumPy array of floats."""
        return np.asarray(self._rvec, dtype=np.float64)

    def to_list(self) -> list[float]:
        """Convert to a Python list of floats."""
        return list(self._rvec)
    

class RCharacter(RObjectWrapper):
    """
    Thin wrapper around an R character (string) vector.
    """

    def __init__(self, rvec: Any):
        renv = lazy_import_r_env()
        cls_vec = renv.ro.baseenv["class"](rvec)
        cls_py = renv.r2py(cls_vec)
        cls_list = [cls_py] if isinstance(cls_py, str) else list(cls_py)
        if "character" not in cls_list:
            raise TypeError(f"Expected R character vector, got R classes {cls_list!r}")
        self._rvec = rvec

    @property
    def r_object(self) -> Any:
        """The underlying rpy2 StrVector."""
        return self._rvec

    @classmethod
    def from_list(cls, xs: Sequence[str]) -> "RCharacter":
        for i, x in enumerate(xs):
            if not isinstance(x, str):
                raise TypeError(f"Element at index {i} is not a string: {x!r}")
        renv = lazy_import_r_env()
        StrVector = renv.ro.vectors.StrVector
        return cls(StrVector(list(xs)))

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "RCharacter":
        if arr.dtype.kind not in ("U", "S", "O"):
            raise TypeError(f"Array dtype must be string-like, got {arr.dtype}")
        flat = arr.ravel().tolist()
        renv = lazy_import_r_env()
        StrVector = renv.ro.vectors.StrVector
        return cls(StrVector(flat))

    def to_list(self) -> list[str]:
        """Convert to a Python list of strings."""
        return list(self._rvec)

    def to_numpy(self) -> np.ndarray:
        """Convert to a NumPy array of strings."""
        return np.asarray(self._rvec, dtype=str)
    

class RLogical(RObjectWrapper):
    """
    Thin wrapper around an R logical (boolean) vector.
    """

    def __init__(self, rvec: Any):
        renv = lazy_import_r_env()
        cls_vec = renv.ro.baseenv["class"](rvec)
        cls_py = renv.r2py(cls_vec)
        cls_list = [cls_py] if isinstance(cls_py, str) else list(cls_py)
        if "logical" not in cls_list:
            raise TypeError(f"Expected R logical vector, got R classes {cls_list!r}")
        self._rvec = rvec

    @property
    def r_object(self) -> Any:
        """The underlying rpy2 BoolVector."""
        return self._rvec

    @classmethod
    def from_list(cls, xs: Sequence[bool]) -> "RLogical":
        for i, x in enumerate(xs):
            if not isinstance(x, (bool, np.bool_)):
                raise TypeError(f"Element at index {i} is not a bool: {x!r}")
        renv = lazy_import_r_env()
        BoolVector = renv.ro.vectors.BoolVector
        return cls(BoolVector(list(xs)))

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "RLogical":
        if arr.dtype.kind != "b":
            raise TypeError(f"Array dtype must be boolean, got {arr.dtype}")
        flat = arr.ravel().tolist()
        renv = lazy_import_r_env()
        BoolVector = renv.ro.vectors.BoolVector
        return cls(BoolVector(flat))

    def to_list(self) -> list[bool]:
        """Convert to a Python list of bools."""
        return [bool(x) for x in self._rvec]

    def to_numpy(self) -> np.ndarray:
        """Convert to a NumPy boolean array."""
        return np.asarray(self._rvec, dtype=bool)
    

class RFactor(RObjectWrapper):
    """
    Thin wrapper around an R factor (categorical) vector.
    """

    def __init__(self, rvec: Any):
        """
        Args:
            rvec: A raw rpy2 StrVector with class 'factor'.

        Raises:
            TypeError: if `rvec` is not an R factor.
        """
        renv = lazy_import_r_env()
        cls_py = renv.r2py(renv.ro.baseenv["class"](rvec))
        # Normalize into a list of class names
        cls_list = [cls_py] if isinstance(cls_py, str) else list(cls_py)
        if "factor" not in cls_list:
            raise TypeError(f"Expected R factor, got R classes {cls_list!r}")
        self._rvec = rvec

    @property
    def r_object(self) -> Any:
        """The underlying rpy2 StrVector (factor)."""
        return self._rvec

    @classmethod
    def from_list(
        cls,
        xs: Sequence[str],
        levels: Optional[Sequence[str]] = None,
        ordered: bool = False
    ) -> "RFactor":
        """
        Build an R factor from a sequence of strings.

        Args:
            xs: sequence of categorical values (must be among `levels`).
            levels: optional full list of allowed levels. If None,
                    inferred from sorted(unique(xs)).

        Raises:
            ValueError: if any x ∉ levels.
        """
        xs = list(xs)
        if levels is None:
            levels = sorted(set(xs))
        else:
            levels = list(levels)

        for x in xs:
            if x not in levels:
                raise ValueError(f"Value {x!r} not in levels {levels}")

        renv = lazy_import_r_env()
        # call base::factor(..., levels=..., ordered=...)
        factor_fn = renv.importr("base").factor
        levels_r = renv.ro.vectors.StrVector(levels)
        ordered_r = renv.ro.vectors.BoolVector([ordered])
        rfac = factor_fn(xs, levels=levels_r, ordered=ordered_r)
        return cls(rfac)

    @classmethod
    def from_pandas(
        cls,
        cat: pd.Categorical
    ) -> "RFactor":
        """
        Build an R factor from a pandas.Categorical.

        Args:
            cat: a pandas.Categorical object.

        Raises:
            TypeError: if `cat` is not a Categorical.
        """
        if not isinstance(cat, pd.Categorical):
            raise TypeError(f"Expected pandas.Categorical, got {type(cat)}")

        # Extract values and categories
        xs = cat.astype(str).tolist()
        levels = cat.categories.tolist()
        # renv = lazy_import_r_env()
        # return cls.from_list(xs, levels=levels)
        # preserve ordered flag
        return cls.from_list(xs, levels=levels, ordered=cat.ordered)

    def to_pandas(self) -> pd.Categorical:
        """
        Convert this R factor to a pandas.Categorical,
        preserving categories and orderedness.
        """
        import pandas as pd
        renv = lazy_import_r_env()
        # 1) coerce to character on the R side, then convert
        as_char = renv.ro.baseenv["as.character"](self._rvec)
        vals = renv.r2py(as_char)
        # 2) the levels attribute
        levels = renv.r2py(self._rvec.slots["levels"])
        # 3) detect orderedness via base::is.ordered()
        is_ord = renv.ro.baseenv["is.ordered"](self._rvec)
        ordered = bool(renv.r2py(is_ord))
        return pd.Categorical(vals, categories=levels, ordered=ordered)
    

class RDataFrame(RObjectWrapper):
    """
    Thin wrapper around an R data.frame.
    """

    def __init__(self, rdf: Any):
        renv = lazy_import_r_env()
        cls_vec = renv.ro.baseenv["class"](rdf)
        cls_py  = renv.r2py(cls_vec)
        cls_list = [cls_py] if isinstance(cls_py, str) else list(cls_py)
        if "data.frame" not in cls_list:
            raise TypeError(f"Expected R data.frame, got R classes {cls_list!r}")
        self._rdf = rdf

    @property
    def r_object(self) -> Any:
        return self._rdf

    @classmethod
    def from_pandas(cls, df: "pd.DataFrame") -> "RDataFrame":
        """
        Build an R data.frame from a pandas DataFrame, preserving index & columns.
        """
        renv = lazy_import_r_env()
        # use pandas2ri conversion
        with renv.localconverter(renv.default_converter + renv.pandas2ri.converter):
            rdf = renv.get_conversion().py2rpy(df)
        return cls(rdf)

    # def to_pandas(self) -> "pd.DataFrame":
    #     """
    #     Convert back to pandas.DataFrame (preserving dtypes & indices).
    #     """
    #     import pandas as pd
    #     renv = lazy_import_r_env()
    #     with renv.localconverter(renv.default_converter + renv.pandas2ri.converter):
    #         return renv.get_conversion().rpy2py(self._rdf)
    def to_pandas(self) -> "pd.DataFrame":
        """Convert this R data.frame back into pandas, up‐casting ints to int64."""
        renv = lazy_import_r_env()
        with renv.localconverter(renv.default_converter + renv.pandas2ri.converter):
            df2: pd.DataFrame = renv.get_conversion().rpy2py(self._rdf)
        # pandas2ri gives us int32 for R integer columns; cast these up to int64
        for col, dtype in df2.dtypes.items():
            if dtype == "int32":
                df2[col] = df2[col].astype("int64")
        return df2




# class RMatrix:
#     """
#     A thin wrapper around an R matrix object.
#     You can construct one from any NumPy array, pandas DataFrame,
#     or — if you install the extra — an AnnData.
#     """

#     def __init__(self, r_matrix: Any):
#         """
#         Parameters
#         ----------
#         r_matrix
#             An R matrix-like object (e.g. a base matrix/array or a
#             dgCMatrix S4 object).  We will verify at runtime that
#             its R class contains "matrix" (case-insensitive).
#         """
#         # grab our lazy R environment
#         renv = lazy_import_r_env()

#         # ask R what class(es) this object has
#         cls_vec = renv.ro.baseenv["class"](r_matrix)
#         cls_list = list(renv.ro.conversion.rpy2py(cls_vec))

#         # require at least one of the classes to mention "matrix"
#         if not any("matrix" in str(c).lower() for c in cls_list):
#             raise TypeError(
#                 f"Expected an R matrix-like object (class includes 'matrix'), "
#                 f"but got R classes: {cls_list}"
#             )

#         # store the validated R object
#         self._rmat = r_matrix

class RMatrix(RObjectWrapper):
    """
    A thin wrapper around an R matrix object.
    You can construct one from any NumPy array, pandas DataFrame,
    or — if you install the extra — an AnnData.
    """

    def __init__(self, r_matrix: Any):
        """
        Parameters
        ----------
        r_matrix
            An R matrix-like object (e.g. a base matrix/array or a
            dgCMatrix S4 object).  We will verify at runtime that
            its R class contains "matrix" (case-insensitive).
        """
        # grab our lazy R environment
        renv = lazy_import_r_env()

        # ask R what class(es) this object has
        cls_vec = renv.ro.baseenv["class"](r_matrix)
        # cls_list = list(renv.ro.conversion.rpy2py(cls_vec))
        cls_py  = renv.r2py(cls_vec)
        cls_list = [cls_py] if isinstance(cls_py, str) else list(cls_py)

        # require at least one of the classes to mention "matrix"
        if not any("matrix" in str(c).lower() for c in cls_list):
            raise TypeError(
                f"Expected an R matrix-like object (class includes 'matrix'), "
                f"but got R classes: {cls_list}"
            )

        # store the validated R object
        self._rmat = r_matrix

    def __repr__(self) -> str:
        # try to pull rclass off the underlying R object directly
        try:
            rclasses = tuple(self._r_model.rclass)
        except Exception:
            rclasses = ()
        name = type(self).__name__
        if rclasses:
            cls_str = ','.join(rclasses)
        else:
            cls_str = 'Unknown'
        return f"<{name} wrapping R classes: [{cls_str}]>"

    @property
    def r(self) -> Any:
        """Raw rpy2 object (ro.Matrix or an S4 Matrix instance)."""
        return self._rmat
    
    @property
    def r_object(self) -> Any:
        """
        The underlying raw rpy2 object (e.g. IntVector, Matrix, ListVector, etc.).

        Returns:
            Any: an rpy2.robjects.* instance
        """
        return self._rmat

    def to_numpy(self) -> np.ndarray:
        """Convert back into a NumPy array."""
        renv = lazy_import_r_env()
        with renv.localconverter(
            renv.default_converter + renv.numpy2ri.converter
        ):
            return renv.ro.conversion.rpy2py(self._rmat)

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        rownames: Optional[Sequence[str]] = None,
        colnames: Optional[Sequence[str]] = None
    ) -> "RMatrix":
        """
        Build an R matrix from a numpy array, with optional dimnames.
        """
        renv = lazy_import_r_env()
        # convert the array
        with renv.localconverter(renv.default_converter + renv.numpy2ri.converter):
            rmat = renv.py2r(arr)

        # attach dimnames if provided
        if rownames is not None or colnames is not None:
            rn = renv.ro.StrVector(list(rownames)) if rownames is not None else renv.ro.NULL
            cn = renv.ro.StrVector(list(colnames)) if colnames is not None else renv.ro.NULL
            dn = renv.ro.baseenv["list"](rn, cn)
            rmat = renv.ro.baseenv["dimnames<-"](rmat, dn)

        return cls(rmat)

    # ... (rest of your methods unchanged) ...

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame") -> "RMatrix":
        """
        Build an R matrix from a pandas DataFrame, preserving index/columns.
        """
        
        renv = lazy_import_r_env()
        colnames = np.asarray(df.columns)
        rownames = np.asarray(df.index)
        mat = df.values
        with renv.localconverter(
            renv.default_converter + renv.numpy2ri.converter
        ):
            cv = renv.ro.conversion.get_conversion()
            rmat = cv.py2rpy(mat)

        assign_dimnames = renv.ro.baseenv["dimnames<-"]
        colnames = renv.py2r(colnames)
        rownames = renv.py2r(rownames)
        rmat = assign_dimnames(rmat, renv.ro.r.list(rownames, colnames))
        return cls(rmat)

    def to_dataframe(self) -> pd.DataFrame:
        renv = lazy_import_r_env()
        rmat = self.r
        colnames = renv.ro.baseenv["colnames"](rmat)
        colnames = renv.r2py(colnames)
        rownames = renv.ro.baseenv["rownames"](rmat)
        rownames = renv.r2py(rownames)

        mat = self.to_numpy()
        
        return pd.DataFrame(
            data = mat,
            index = rownames,
            columns = colnames
        )

    @classmethod
    def from_anndata(
        cls,
        adata: "anndata.AnnData",
        layer: Optional[str] = None
    ) -> "RMatrix":
        """
        Build an R matrix from an AnnData object’s `.X` or a named layer.
        
        This method requires the `anndata` and `anndata2ri` packages;
        if they’re not installed you’ll get an ImportError.
        """
        try:
            import anndata  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AnnData support requires `pip install pyrtools[anndata]`"
            ) from e

        # select the array
        mat = adata.X if layer is None else adata.layers[layer]

        if not isinstance(mat, (np.ndarray,)):
            # if it’s sparse you could add anndata2ri import here…
            raise ValueError("AnnData matrix must be a numpy.ndarray")

        # convert to R matrix, preserving row/col names
        rown = np.asarray(adata.obs_names)
        coln = np.asarray(adata.var_names)
        return cls.from_numpy(mat, rownames=rown, colnames=coln)


    def to_anndata(
        self,
        obs_names: Optional[Sequence[str]] = None,
        var_names: Optional[Sequence[str]] = None
    ) -> "anndata.AnnData":
        """Convert this R matrix into an AnnData, setting obs_names/var_names.

        Args:
            obs_names: If provided, a list of strings to use as `.obs_names`.
                Must match the number of rows.  If None, uses the R rownames.
            var_names: If provided, a list of strings to use as `.var_names`.
                Must match the number of columns.  If None, uses the R colnames.

        Returns:
            AnnData with `.X` set to the matrix values, and index names set.

        Raises:
            ImportError: if `anndata` isn’t installed.
            ValueError: if the provided names don’t match the matrix dimensions.
        """
        try:
            import anndata  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AnnData support requires `pip install pyrtools[anndata]`"
            ) from e

        # 1) pull out values and dimnames
        arr = self.to_numpy()
        df  = self.to_dataframe()
        n_obs, n_var = arr.shape

        # 2) determine obs_names / var_names
        default_obs = list(df.index)
        default_var = list(df.columns)

        if obs_names is None:
            obs_idx = default_obs
        else:
            if len(obs_names) != n_obs:
                raise ValueError(
                    f"obs_names length {len(obs_names)} != number of rows {n_obs}"
                )
            obs_idx = list(obs_names)

        if var_names is None:
            var_idx = default_var
        else:
            if len(var_names) != n_var:
                raise ValueError(
                    f"var_names length {len(var_names)} != number of cols {n_var}"
                )
            var_idx = list(var_names)

        # 3) build minimal obs/var DataFrames
        import pandas as pd
        obs = pd.DataFrame(index=obs_idx)
        var = pd.DataFrame(index=var_idx)

        # 4) assemble AnnData
        return anndata.AnnData(X=arr, obs=obs, var=var)


# src/pyrtools/sparse_wrapper.py

import numpy as np
from typing import Any, Optional, Sequence, Union
from types import SimpleNamespace

# from .r_env import lazy_import_r_env


# class RSparseMatrix:
class RSparseMatrix(RObjectWrapper):
    """
    A thin wrapper around an R sparse‐matrix (dgCMatrix or dgRMatrix).
    Can round‐trip to/from SciPy CSR/CSC, NumPy dense, pandas DataFrame,
    or AnnData.
    """

    def __init__(self, r_s4: Any):
        """
        Parameters
        ----------
        r_s4
            An R S4 sparse‐matrix object (class 'dgCMatrix' or 'dgRMatrix').
        Raises
        ------
        TypeError
            If the provided object is not a dgCMatrix or dgRMatrix.
        """
        renv = lazy_import_r_env()
        # get R class vector
        cls_vec  = renv.ro.baseenv["class"](r_s4)
        # cls_list = list(renv.ro.conversion.rpy2py(cls_vec))
        cls_py  = renv.r2py(cls_vec)
        cls_list = [cls_py] if isinstance(cls_py, str) else list(cls_py)

        # must contain one of the sparse‐matrix classes
        allowed = {"dgCMatrix", "dgRMatrix"}
        if not any(c in allowed for c in cls_list):
            raise TypeError(
                f"Expected an R dgCMatrix/dgRMatrix, got classes: {cls_list}"
            )

        self._rmat = r_s4

    def __repr__(self) -> str:
        renv = lazy_import_r_env()
        try:
            cls_vec  = renv.ro.baseenv["class"](self._rmat)
            cls_list = list(renv.ro.conversion.rpy2py(cls_vec))
            cls_str  = ",".join(cls_list)
        except Exception:
            cls_str = "Unknown"
        return f"<RSparseMatrix wrapping R classes: [{cls_str}]>"

    @property
    def r_object(self) -> Any:
        """
        The underlying raw rpy2 S4 sparse‐matrix object (dgCMatrix or dgRMatrix).

        Returns:
            Any: an rpy2.robjects.methods.RS4 instance.
        """
        return self._rmat
    
    @property
    def r(self) -> Any:
        """Raw rpy2 S4 object (dgCMatrix/dgRMatrix)."""
        return self._rmat

    def to_scipy(self) -> Union["csr_matrix", "csc_matrix"]:
        """
        Convert this R sparse‐matrix into a SciPy CSR or CSC matrix.
        """
        # delayed imports
        from scipy.sparse import csr_matrix, csc_matrix
        from anndata2ri    import scipy2ri

        renv = lazy_import_r_env()
        with renv.localconverter(
            renv.default_converter + scipy2ri.converter
        ):
            return renv.get_conversion().rpy2py(self._rmat)

    def to_numpy(self) -> np.ndarray:
        """
        Convert to a dense NumPy array (via .to_scipy()).
        """
        return self.to_scipy().toarray()

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Convert to a pandas DataFrame (dense), preserving dimnames.
        """
        import pandas as pd
        arr = self.to_numpy()

        renv = lazy_import_r_env()
        # fetch dimnames from R
        rn = renv.ro.baseenv["rownames"](self._rmat)
        cn = renv.ro.baseenv["colnames"](self._rmat)
        rownames = renv.r2py(rn)
        colnames = renv.r2py(cn)

        return pd.DataFrame(arr, index=rownames, columns=colnames)

    @classmethod
    def from_scipy(
        cls,
        mat: Union["csr_matrix", "csc_matrix"],
        rownames: Optional[Sequence[str]] = None,
        colnames: Optional[Sequence[str]] = None
    ) -> "RSparseMatrix":
        """
        Build an R dgCMatrix/dgRMatrix from a SciPy sparse matrix.

        Parameters
        ----------
        mat
            A SciPy CSR or CSC sparse matrix.
        rownames
            Optional list of row names (length = mat.shape[0]).
        colnames
            Optional list of column names (length = mat.shape[1]).

        Returns
        -------
        RSparseMatrix
            Wrapper around the newly created R S4 sparse‐matrix.
        """
        # from anndata2ri import scipy2ri
        renv = lazy_import_r_env()

        # convert the SciPy matrix into R
        # with renv.localconverter(
        #     renv.default_converter + scipy2ri.converter
        # ):
            # rmat = renv.ro.conversion.py2rpy(mat)
        rmat = renv.py2r(mat)

        # attach dimnames if requested
        if rownames is not None or colnames is not None:
            rn = renv.ro.StrVector(list(rownames)) if rownames is not None else renv.ro.NULL
            cn = renv.ro.StrVector(list(colnames)) if colnames is not None else renv.ro.NULL
            dn = renv.ro.baseenv["list"](rn, cn)
            rmat = renv.ro.baseenv["dimnames<-"](rmat, dn)

        return cls(rmat)

    @classmethod
    def from_anndata(
        cls,
        adata: "anndata.AnnData",
        layer: Optional[str] = None
    ) -> "RSparseMatrix":
        """
        Build an R sparse‐matrix from an AnnData’s `.X` (if sparse)
        or a named layer.

        Requires `anndata` and `anndata2ri` to be installed.

        Parameters
        ----------
        adata
            AnnData object whose .X (or .layers[layer]) is a sparse matrix.
        layer
            Name of the layer to use instead of .X.

        Returns
        -------
        RSparseMatrix
        """
        try:
            import anndata  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "AnnData support requires `pip install pyrtools[anndata]`"
            ) from e

        mat = adata.X if layer is None else adata.layers[layer]
        from scipy.sparse import csr_matrix, csc_matrix
        if not isinstance(mat, (csr_matrix, csc_matrix)):
            raise ValueError("AnnData layer must be CSR/CSC sparse matrix")

        rown = adata.obs_names.tolist()
        coln = adata.var_names.tolist()
        return cls.from_scipy(mat, rownames=rown, colnames=coln)



from typing import Any, Mapping, Sequence, Union
from types import SimpleNamespace

# from .r_env import lazy_import_r_env


# class RList:
class RList(RObjectWrapper):
    """
    Wraps an R list (rpy2 ListVector), with helpers to convert
    to/from Python lists or dicts.
    """

    def __init__(self, rlist_obj: Any):
        """
        Parameters
        ----------
        rlist_obj
            An rpy2.robjects.vectors.ListVector instance (an R list).
        
        Raises
        ------
        TypeError
            If `rlist_obj` is not a ListVector.
        """
        renv = lazy_import_r_env()
        ListVector = renv.ro.vectors.ListVector

        if not isinstance(rlist_obj, ListVector):
            raise TypeError(
                f"Expected an R ListVector, got {type(rlist_obj)}"
            )
        self._rlist = rlist_obj

    def __repr__(self) -> str:
        renv = lazy_import_r_env()
        # show names if present
        try:
            names = list(self._rlist.names) or []
        except Exception:
            names = []
        return f"<RList names={names}>"

    @property
    def r_object(self) -> Any:
        """
        The underlying raw rpy2 ListVector.

        Returns:
            Any: an rpy2.robjects.vectors.ListVector instance.
        """
        return self._rlist
    
    @property
    def r(self) -> Any:
        """Access the raw rpy2 ListVector."""
        return self._rlist

    @classmethod
    def from_list(cls, seq: Sequence[Any]) -> "RList":
        """
        Convert a Python sequence into an R list.

        Elements will be converted via your `_py_to_r` registry.

        Parameters
        ----------
        seq
            Any Python sequence (list, tuple, etc.).

        Returns
        -------
        RList
        """
        renv = lazy_import_r_env()
        r_obj = renv.py2r(seq)
        return cls(r_obj)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RList":
        """
        Convert a Python dict into an R named list.

        Keys become names(), values converted via `_py_to_r`.

        Parameters
        ----------
        d
            A dict of {name: value}.

        Returns
        -------
        RList
        """
        renv = lazy_import_r_env()
        r_obj = renv.py2r(d)
        return cls(r_obj)

    def to_list(self) -> list:
        """
        Convert this R list back to a Python list.

        If the R list is named, the names are discarded.
        """
        renv = lazy_import_r_env()
        return renv.r2py(self._rlist)

    def to_dict(self) -> dict:
        """
        Convert this R list back to a Python dict.

        Requires that the R list is named (i.e. names() is non-null).
        """
        renv = lazy_import_r_env()
        names = renv.r2py(renv.ro.baseenv["names"](self._rlist))
        if names is None:
            raise TypeError("For conversion to dictionary, the R List must contain names")
        py = renv.r2py(self._rlist)
        if not isinstance(py, dict):
            raise ValueError("Underlying R list has no names, cannot to_dict()")
        return py