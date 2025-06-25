# src/pyrtools/matrix_wrapper.py

import numpy as np
from typing import Any, Optional, Sequence
from types import SimpleNamespace

# from .r_env import lazy_import_r_env


class RMatrix:
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
        cls_list = list(renv.ro.conversion.rpy2py(cls_vec))

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


class RSparseMatrix:
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
        cls_list = list(renv.ro.conversion.rpy2py(cls_vec))

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
            return renv.ro.conversion.rpy2py(self._rmat)

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


class RList:
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