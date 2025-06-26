# pyrtools

A lightweight toolbox for seamless interop between Python and R via **rpy2**.  
It provides:

1. **Basic conversions** (Python ↔ R) bundled in a **lazy** namespace.  
2. **`RFunctionWrapper`** for interactive analysis: wrap any R function as a Python-callable.  
3. **`RModelWrapper`** for advanced workflows: wrap S3/S4 model objects, expose slots, call methods.  
4. **Type-focused wrappers** (`RMatrix`, `RSparseMatrix`, `RList`): round-trip data based on target R type, not just Python type.

---

## Installation

For now, please install this package from GitHub:

```bash
pip install git+https://github.com/MaximilianNuber/pyrtools.git
```

The future installation from pip:

```bash
pip install pyrtools
```

> **Note:** **R** and **rpy2** are _optional_ dependencies.  
> You must install them yourself (e.g. `conda install -c conda-forge r-base rpy2`) before using any of the R-calling features.

If you need AnnData support:

```bash
pip install pyrtools[anndata]
```

---

## 1. Basic Conversions & Lazy Namespace

All core R entry-points and your `_py_to_r` / `_r_to_py` converters live in one lazy namespace:

```python
from pyrtools import lazy_import_r_env
env = lazy_import_r_env()
print(env)
```

Examples:

```python
import numpy as np
import pandas as pd
from pyrtools import lazy_import_r_env

env = lazy_import_r_env()
ro, py2r, r2py = env.ro, env.py2r, env.r2py

# Python list → R vector
r_vec = py2r([1,2,3])
assert r2py(r_vec) == [1,2,3]

# NumPy → R matrix
arr = np.arange(6).reshape(2,3)
r_mat = py2r(arr)
np.testing.assert_array_equal(r2py(r_mat), arr)

# pandas DataFrame → R data.frame
df = pd.DataFrame({"x":[1,2],"y":[3,4]})
r_df = py2r(df)
# because R automatically uses integer strings for rownames, pandas uses range(...):
df.index = ["0", "1"] 
pd.testing.assert_frame_equal(r2py(r_df), df)
```

---

## 2. Interactive Analysis with `RFunctionWrapper`

Wrap any R function as a Python-callable:

- sanitized argument names  
- automatic Python→R and R→Python conversion  
- `convert_output=False` to keep raw R objects  

### 2.1 Simple `lm()` example

```python
from pyrtools import RFunctionWrapper
import seaborn as sns

lm_py = RFunctionWrapper("lm", package="stats").get_python_function()
df    = sns.load_dataset("penguins").dropna(subset=["body_mass_g"])
fit   = lm_py("flipper_length_mm ~ body_mass_g", df)
print(fit.keys())
```

To keep the raw R model object:

```python
lm_raw = RFunctionWrapper("lm").get_python_function(convert_output=False)
r_model = lm_raw("Sepal.Length ~ Sepal.Width", df)
```

### 2.2 edgeR on the `airway` dataset

```python
from pyrtools import lazy_import_r_env, RFunctionWrapper
import numpy as np

renv = lazy_import_r_env()
ro   = renv.ro

# load airway data
ro.r('library(airway); data(airway)')
counts_r = ro.r('assay(airway)')
obs_r    = ro.r('as.data.frame(colData(airway))')
var_r    = ro.r('as.data.frame(rowData(airway))')

counts = renv.r2py(counts_r)
obs    = renv.r2py(obs_r)

def to_formula(f):
    return renv.ro.r["as.formula"](f)

dge_list     = RFunctionWrapper("DGEList", package="edgeR").get_python_function(convert_output=False)
model_matrix = RFunctionWrapper("model.matrix", package="stats").get_python_function(convert_output=False)
filter_expr  = RFunctionWrapper("filterByExpr", package="edgeR").get_python_function(convert_output=False)
glm_ql_fit   = RFunctionWrapper("glmQLFit", package="edgeR").get_python_function(convert_output=False)
glm_ql_test  = RFunctionWrapper("glmQLFTest", package="edgeR").get_python_function(convert_output=False)

r_design = model_matrix(to_formula("~ dex"), data=obs)
dge      = dge_list(counts, samples=obs)
mask     = filter_expr(dge, design=r_design)
idx_all  = renv.py2r(np.ones(counts.shape[1], dtype=bool))
dge2     = dge.rx(mask, idx_all)
fit1     = glm_ql_fit(dge2, design=r_design)
fit2     = glm_ql_test(fit1, coef=2)
top      = RFunctionWrapper("topTags").get_python_function()(fit2)
print(top["table"].head())
```

---

## 3. Advanced Workflows with `RModelWrapper`

Wrap a fitted R model to:

- inspect R Model slot names via `.r_slot_names()`
- expose R Model slots as Python attributes  
- call any R method via `.apply()` 

```python
from pyrtools import RModelWrapper

mod = RModelWrapper(r_model)
print(mod.coefficients)
summary = mod.apply("summary")
```

### 3.1 Subclassing for edgeR’s DGEList pipeline

```python
import numpy as np
from pyrtools import RFunctionWrapper, RModelWrapper, lazy_import_r_env

class DGEList(RModelWrapper):
    def __init__(self, r_model_object):
        super().__init__(r_model_object)
        renv = lazy_import_r_env()
        rcls = renv.r2py(renv.ro.baseenv["class"](self._r_model))
        if isinstance(rcls, list):
            ok = "DGEList" in rcls
        else:
            ok = (rcls == "DGEList")
        if not ok:
            raise TypeError(f"Expected a DGEList, got {rcls!r}")

    def subset(self, rows=None, cols=None):
        renv = lazy_import_r_env()
        dge   = self._r_model
        nrows = renv.ro.baseenv["nrow"](dge)
        ncols = renv.ro.baseenv["ncol"](dge)
        rows = rows if rows is not None else [True]*nrows
        cols = cols if cols is not None else [True]*ncols
        sub_r = renv.ro.baseenv["["](dge, renv.py2r(rows), renv.py2r(cols))
        return DGEList(sub_r)

    @property
    def shape(self):
        renv = lazy_import_r_env()
        return tuple(renv.ro.baseenv["dim"](self._r_model))
```

```python
from pyrtools import RFunctionWrapper
from diffexptools.dgelist_wrapper import DGEList

def make_dge_list(counts, **kwargs):
    raw = RFunctionWrapper("DGEList", package="edgeR")\
              .get_python_function(convert_output=False)
    return DGEList(raw(counts, **kwargs))

def filter_by_expr(dge, design):
    fn = RFunctionWrapper("filterByExpr", package="edgeR")\
             .get_python_function(convert_output=False)
    return DGEList(fn(dge._r_model, design=design))

def glmqlfit(dge, design):
    fn = RFunctionWrapper("glmQLFit", package="edgeR")\
             .get_python_function(convert_output=False)
    return DGEList(fn(dge._r_model, design=design))

def glmqlftest(fit, **kwargs):
    fn = RFunctionWrapper("glmQLFTest", package="edgeR")\
             .get_python_function(convert_output=False)
    return DGEList(fn(fit._r_model, **kwargs))
```

```python
# continuing the airway example
dge    = make_dge_list(counts, samples=obs)
mask   = filter_by_expr(dge, design=r_design)
dge2   = dge.subset(rows=mask)
fit    = glmqlfit(dge2, design=r_design)
result = glmqlftest(fit, coef=2)
print(result.r_slot_names)
print(result.table.head())
```

---

## 4. Type-Focused Wrappers: `RMatrix`, `RSparseMatrix`, `RList`

UPDATE: Beyond these three R type classes I have added:`RInteger`, `RNumeric`, `RCharacter`, `RBool`, `RDataFrame` and `RFactor`.
Please consider how these can replace the default _py_to_r and _r_to_py conversion functions, which exist in `lazy_import_r_env` as `py2r` and `r2py` and enable not only a mapping from one python type to one R-type, but enable many-to-many mappings for different input and output types.

All R-Object classes, `RModelWrapper` now, too, inherit from an AbstractBaseClass enforcing the `.r_object` method to expose the actual sexp-wrapper or rpy2 class.
Subsequently, after each conversion we can extract `x.r_object` and use it as input to RFunctionWrapper.

Also, each of these classes exposes explicitely named (class-) methods to convert from and to a Python type.

All R-Vector wrappers can convert to and from Python numpy.ndarray, and python list, with `.from_numpy(arr)` or `.from_list(l)`.

RFactor can convert from list, where a Sequence for levels is set in addition to the python list, or from pandas.Categorical.

RDataFrame is run through:
```python
df = ...
with localconverter(default_converter + pandas2ri.converter):
    cv = get_conversion()
    cv.py2rpy(df)
```

Examples for these and better docstrings will follow soon.

### RMatrix

```python
from pyrtools import RMatrix
import numpy as np

arr  = np.random.rand(10,5)
rmat = RMatrix.from_numpy(arr,
                         rownames=[f"r{i}" for i in range(10)],
                         colnames=[f"c{j}" for j in range(5)])
back = rmat.to_numpy()
df2  = rmat.to_dataframe()
```

### RSparseMatrix

```python
from scipy.sparse import csr_matrix
from pyrtools import RSparseMatrix

sp  = csr_matrix([[0,1,0],[2,0,3]])
rsp = RSparseMatrix.from_scipy(sp,
                               rownames=["a","b"],
                               colnames=["x","y","z"])
csr2  = rsp.to_scipy()
dense = rsp.to_numpy()
```

### RList

```python
from pyrtools import RList

rl = RList.from_list([1, "foo", True])
assert rl.to_list() == [1, "foo", True]

rd = RList.from_dict({"x":10, "y":[1,2,3]})
assert rd.to_dict() == {"x":10, "y":[1,2,3]}
```

---

## Contributing & License

- Core converters, wrappers, and lazy-import live in **pyrtools**.  
- PRs welcome!  
- **MIT License** — see [LICENSE](LICENSE).



```python

```
