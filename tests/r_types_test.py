import pytest
import numpy as np
import pandas as pd

from pyrtools.r_env import lazy_import_r_env
from pyrtools.r_types import (
    RInteger,
    RLogical,
    RNumeric,
    RCharacter,
    RList,
    RFactor,
    RDataFrame,
    RMatrix
    # RExpression,
)
# from pyrtools.matrix_wrapper import RMatrix


@pytest.fixture(scope="module")
def renv():
    """Shared lazy R environment for all tests."""
    return lazy_import_r_env()


# ─── Scalars and Simple Vectors ───────────────────────────────────────────────

def test_rinteger_from_list_and_back(renv):
    xs = [1, 2, 3, -5, 42]
    rint = RInteger.from_list(xs)
    assert hasattr(rint, "r_object")
    arr = rint.to_numpy()
    assert isinstance(arr, np.ndarray) and arr.dtype == np.int32
    assert arr.tolist() == xs
    # and back to Python list
    assert rint.to_list() == xs

def test_rlogical_from_list_and_back(renv):
    vals = [True, False, True, False]
    rlog = RLogical.from_list(vals)
    lst = rlog.to_list()
    assert lst == vals
    arr = rlog.to_numpy()
    assert isinstance(arr, np.ndarray) and arr.dtype == bool
    assert arr.tolist() == vals

def test_rnumeric_from_numpy_and_list(renv):
    arr = np.array([0.1, 3.14, -2.5], dtype=float)
    rnum1 = RNumeric.from_numpy(arr)
    back = rnum1.to_numpy()
    np.testing.assert_allclose(back, arr)
    rnum2 = RNumeric.from_list([0.5, -1.25, 2.0])
    assert isinstance(rnum2.to_list(), list)
    assert all(isinstance(x, float) for x in rnum2.to_list())

def test_rcharacter_roundtrip(renv):
    xs = ["apple", "banana", "🍒"]
    rchr = RCharacter.from_list(xs)
    assert rchr.to_list() == xs


# ─── R list wrapper ─────────────────────────────────────────────────────────

def test_rlist_basic(renv):
    # mixed list: int, float, str, logical
    native = [1, 2.5, "foo", True]
    rl = RList.from_list(native)
    # raw r object
    robj = rl.r_object
    # convert back
    back = rl.to_list()
    assert back == native


# ─── RFactor ────────────────────────────────────────────────────────────────

def test_rfactor_unordered_and_ordered(renv):
    values = ["low", "medium", "high", "low"]
    # unordered
    rf1 = RFactor.from_list(values)
    cat1 = rf1.to_pandas()
    assert isinstance(cat1, pd.Categorical)
    assert not cat1.ordered
    assert list(cat1) == values

    # ordered with explicit levels
    levels = ["low", "medium", "high"]
    rf2 = RFactor.from_list(values, levels=levels, ordered=True)
    cat2 = rf2.to_pandas()
    assert cat2.ordered
    assert list(cat2.categories) == levels

    # round‐trip from pandas
    pcat = pd.Categorical(values, categories=levels, ordered=True)
    rf3 = RFactor.from_pandas(pcat)
    cat3 = rf3.to_pandas()
    assert cat3.ordered
    assert list(cat3) == values


# ─── RDataFrame ────────────────────────────────────────────────────────────

def test_rdataframe_roundtrip(renv):
    df = pd.DataFrame({
        "A": [1,2,3],
        "B": ["x","y","z"],
        "C": [True, False, True]
    }, index=["r1","r2","r3"])

    rdf = RDataFrame.from_pandas(df)
    df2 = rdf.to_pandas()
    pd.testing.assert_frame_equal(df2, df)


# ─── RMatrix ────────────────────────────────────────────────────────────────

def test_rmatrix_numpy_roundtrip(renv):
    arr = np.arange(12).reshape(3,4)
    rm = RMatrix.from_numpy(arr, rownames=["r1","r2","r3"], colnames=["c1","c2","c3","c4"])
    back = rm.to_numpy()
    np.testing.assert_array_equal(back, arr)
    repr_str = repr(rm)
    assert "RMatrix" in repr_str

def test_rmatrix_dataframe_roundtrip(renv):
    df = pd.DataFrame(np.random.randn(2,3), index=["i","ii"], columns=["X","Y","Z"])
    rm2 = RMatrix.from_dataframe(df)
    df2 = rm2.to_dataframe()
    pd.testing.assert_frame_equal(df2, df)


# ─── RExpression ────────────────────────────────────────────────────────────

# def test_rexpression_call_and_eval(renv):
#     # build R call: sum(1:5)
#     ro = renv.ro
#     call = ro.baseenv["call"]("sum", ro.baseenv[":" ](1,5))
#     expr = RExpression(call)
#     # repr shows the R code
#     s = repr(expr)
#     assert "sum(1:5)" in s
#     # eval → IntVector([15])
#     out = expr.eval()
#     py = renv.r2py(out)
#     assert py == 15


# ─── raw RObjectWrapper enforcement ────────────────────────────────────────

def test_robjectwrapper_abstract():
    from pyrtools.r_types import RObjectWrapper
    # you should not be able to instantiate the ABC
    with pytest.raises(TypeError):
        RObjectWrapper()  # abstract, or missing r_object property

