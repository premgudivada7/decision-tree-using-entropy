"""The tree must run on any (tabular, classification) dataset."""
import datetime as dt
import warnings

import numpy as np
import pytest

from gaintree import GainTreeClassifier

pd = pytest.importorskip("pandas")


def fit_predict(X, y, **kw):
    t = GainTreeClassifier(**kw).fit(X, y)
    pred = t.predict(X)
    assert len(pred) == len(y)
    p = t.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1)
    t.to_text(); t.to_dot()
    return t


# ---------- missing values ------------------------------------------------- #
def test_nan_in_numeric_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(int)
    X[rng.random(X.shape) < 0.2] = np.nan
    t = fit_predict(X, y)
    assert t.score(X, y) > 0.8
    assert t.predict([[np.nan, np.nan, np.nan]]).shape == (1,)


def test_none_and_pd_na_everywhere():
    df = pd.DataFrame({
        "age": pd.array([22, None, 35, 58, None, 41, 19, 63], dtype="Int64"),
        "city": ["Paris", None, "Lyon", "Paris", "Nice", None, "Lyon", "Nice"],
        "vip": [True, False, None, True, False, False, True, None],
    })
    y = ["a", "b", "a", "b", "b", "a", "a", "b"]
    t = fit_predict(df, y)
    assert t.categorical_mask_.tolist() == [False, True, True] or \
        t.categorical_mask_.tolist() == [False, True, False]


def test_all_missing_column_is_ignored():
    X = np.column_stack([np.full(50, np.nan), np.arange(50)])
    y = (np.arange(50) > 25).astype(int)
    t = fit_predict(X, y)
    assert t.feature_importances_[0] == 0


def test_missing_labels_give_clear_error():
    with pytest.raises(ValueError, match="missing labels"):
        GainTreeClassifier().fit([[1], [2], [3]], [0, None, 1])
    with pytest.raises(ValueError, match="missing labels"):
        GainTreeClassifier().fit([[1], [2], [3]], [0.0, np.nan, 1.0])


# ---------- types ---------------------------------------------------------- #
def test_mixed_types_in_one_column_become_categorical():
    X = [[1, "a"], ["x", 2.5], [3, "a"], [None, "b"]]
    t = fit_predict(X, [0, 1, 0, 1])
    assert t.categorical_mask_.tolist() == [True, True]


def test_numeric_strings_are_numbers():
    X = [["1.5"], ["2.0"], ["10"], [" 7 "]]
    t = fit_predict(X, ["lo", "lo", "hi", "hi"])
    assert not t.categorical_mask_[0]


def test_dates_datetimes_timedeltas():
    df = pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=40, freq="D", tz="Europe/Paris"),
        "dur": pd.to_timedelta(np.arange(40), unit="h"),
        "d": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(40)],
    })
    y = (np.arange(40) > 20).astype(int)
    t = fit_predict(df, y)
    assert t.score(df, y) == 1.0
    X = np.array(["2024-01-%02d" % (i + 1) for i in range(20)], dtype="datetime64[D]").reshape(-1, 1)
    fit_predict(X, (np.arange(20) > 9).astype(int))


def test_bool_and_category_dtypes():
    df = pd.DataFrame({"b": [True, False] * 10,
                       "c": pd.Categorical(["x", "y", "z", "x"] * 5)})
    fit_predict(df, df["b"].to_numpy())


def test_labels_of_any_type():
    X = np.arange(12).reshape(-1, 1)
    for y in ([True, False] * 6, ["cat", "dog", "fish"] * 4, list(range(12)),
              np.array([1.5, 2.5] * 6), [1, "1", 2.0, None or "x"] * 3):
        fit_predict(X, y)


def test_infinite_values():
    X = np.array([[-np.inf], [-5.0], [0.0], [3.0], [np.inf], [np.inf]])
    y = [0, 0, 0, 1, 1, 1]
    t = fit_predict(X, y)
    assert t.score(X, y) == 1.0


def test_huge_and_tiny_numbers():
    X = np.array([[1e308], [-1e308], [1e-308], [5e-324], [0.0], [1.0]])
    fit_predict(X, [1, 0, 0, 1, 0, 1])


# ---------- shapes / degenerate cases ------------------------------------- #
def test_single_class():
    t = fit_predict(np.random.rand(10, 3), ["only"] * 10)
    assert t.n_leaves() == 1 and set(t.predict(np.random.rand(3, 3))) == {"only"}


def test_single_row_and_single_column():
    fit_predict([[1.0]], [3])
    fit_predict(np.array([3, 1, 2]), [0, 1, 0])       # 1-D X = one feature


def test_zero_features():
    t = fit_predict(np.empty((5, 0)), [0, 1, 1, 1, 0])
    assert t.predict(np.empty((2, 0))).tolist() == [1, 1]


def test_constant_features_and_duplicate_rows_with_conflicting_labels():
    X = np.ones((20, 4))
    y = [0, 1] * 10
    t = fit_predict(X, y)
    assert t.n_leaves() == 1


def test_many_classes_and_high_cardinality_category():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"id": [f"user_{i}" for i in range(500)],
                      "x": rng.normal(size=500)})
    y = rng.integers(0, 200, size=500)
    fit_predict(X, y)
    fit_predict(X, y, min_samples_leaf=5)


def test_very_deep_tree_no_recursion_error():
    n = 4000
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.arange(n) % 2               # alternating labels -> many splits
    t = fit_predict(X, y)
    assert t.score(X, y) == 1.0
    t.prune(X, y)


def test_unseen_categories_and_garbage_at_predict_time():
    X = [["red", 1.0], ["blue", 2.0], ["red", 3.0], ["green", 4.0]]
    t = GainTreeClassifier().fit(X, [0, 1, 0, 1])
    out = t.predict([["purple", "not a number"], [None, None], ["red", np.inf]])
    assert len(out) == 3


def test_dataframe_columns_reordered_at_predict_time():
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10)[::-1] * 0})
    t = GainTreeClassifier().fit(df, (df.a > 4).astype(int))
    assert t.score(df[["b", "a"]], (df.a > 4).astype(int)) == 1.0


def test_list_of_rows_sparse_and_series_inputs():
    sp = pytest.importorskip("scipy.sparse")
    X = sp.random(50, 5, density=0.3, format="csr", random_state=0)
    fit_predict(X, (X.toarray()[:, 0] > 0).astype(int))
    fit_predict(pd.Series([10, 20, 30, 40]), ["a", "a", "b", "b"])


def test_continuous_target_warns_but_runs():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_predict(np.random.rand(100, 2), np.random.rand(100))
    assert any("continuous" in str(x.message) for x in w)


def test_empty_predict():
    t = GainTreeClassifier().fit([[0.5], [7.0]], ["x", "y"])
    assert t.predict(np.empty((0, 1))).shape == (0,)


def test_big_tree_plot_is_truncated():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = rng.integers(0, 2, 2000)
    t = GainTreeClassifier().fit(X, y)
    with pytest.warns(UserWarning, match="too big"):
        t.draw()
    assert "… " in t.to_text(max_depth=2)


def test_id_like_text_column_is_ignored():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"name": [f"n{i}" for i in range(200)], "x": rng.normal(size=200)})
    y = (df.x > 0).astype(int)
    with pytest.warns(UserWarning, match="id-like"):
        t = GainTreeClassifier().fit(df, y)
    assert t.ignored_features_ == ["name"] and t.feature_names_[t.trunk_.split_on] == "x"
    t2 = GainTreeClassifier(skip_id_columns=False).fit(df, y)
    assert t2.feature_names_[t2.trunk_.split_on] == "name"


def test_dates_as_text_become_numeric():
    X = [["2024-01-%02d" % (i + 1), "2024-01-01 10:%02d:00" % i] for i in range(20)]
    t = GainTreeClassifier().fit(X, (np.arange(20) > 9).astype(int))
    assert not t.categorical_mask_.any() and t.n_leaves() == 2
    t2 = GainTreeClassifier().fit([["1-2"], ["abc"]], [0, 1])   # not a date
    assert t2.categorical_mask_[0]
