"""Turn "any" tabular input into clean per-column arrays.

Accepted inputs: NumPy arrays (any dtype), lists of rows, pandas / polars
DataFrames, scipy sparse matrices, a single 1-D column.

Handled for you:
    * missing values: None, NaN, pd.NA, NaT        -> NaN (numeric) / "<missing>" (categorical)
    * dates / datetimes / timedeltas                 -> numbers (seconds)
    * booleans                                       -> 0 / 1
    * numbers stored as strings ("3.5")              -> floats
    * ISO dates stored as strings ("2024-05-01")     -> numbers (seconds)
    * pandas nullable / category / string dtypes
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Tuple

import numpy as np

MISSING = "<missing>"


# --------------------------------------------------------------------------- #
# scalar helpers
# --------------------------------------------------------------------------- #
def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    if isinstance(v, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(v))
    try:  # pd.NA, pd.NaT and friends
        return bool(v != v)
    except (TypeError, ValueError):
        return False


def to_float(v) -> float:
    """Convert one value to float or raise ValueError."""
    if is_missing(v):
        return np.nan
    if isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        try:
            return float(v)
        except ValueError:
            pass
        if v[:1].isdigit() and ("-" in v or "/" in v or ":" in v):   # ISO date / time text
            try:
                return np.datetime64(v.replace("/", "-"), "ns").astype(np.int64) / 1e9
            except ValueError:
                pass
        raise ValueError(f"not numeric: {v!r}")
    if isinstance(v, np.datetime64):
        return v.astype("datetime64[ns]").astype(np.int64) / 1e9
    if isinstance(v, np.timedelta64):
        return v.astype("timedelta64[ns]").astype(np.int64) / 1e9
    if isinstance(v, _dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=_dt.timezone.utc)
        return v.timestamp()
    if isinstance(v, _dt.date):
        return float(v.toordinal() * 86400)
    if isinstance(v, _dt.timedelta):
        return v.total_seconds()
    raise ValueError(f"not numeric: {v!r}")


def to_key(v) -> str:
    return MISSING if is_missing(v) else str(v)


# --------------------------------------------------------------------------- #
# column helpers
# --------------------------------------------------------------------------- #
def column_to_float(col: np.ndarray, strict: bool) -> Optional[np.ndarray]:
    """Float version of a column.

    strict=True  -> return None if any non-missing value is not numeric
    strict=False -> non-numeric values become NaN (used at predict time)
    """
    if col.dtype.kind in "biuf":
        return col.astype(float)
    if col.dtype.kind == "M":
        out = col.astype("datetime64[ns]").astype(np.int64) / 1e9
        out[np.isnat(col)] = np.nan
        return out
    if col.dtype.kind == "m":
        out = col.astype("timedelta64[ns]").astype(np.int64) / 1e9
        out[np.isnat(col)] = np.nan
        return out
    try:  # fast path: numbers, None, numeric strings
        return np.array(col, dtype=float)
    except (TypeError, ValueError):
        pass
    out = np.empty(len(col), dtype=float)
    for i, v in enumerate(col):
        try:
            out[i] = to_float(v)
        except (ValueError, TypeError, OverflowError):
            if strict:
                return None
            out[i] = np.nan
    return out


def column_to_keys(col: np.ndarray) -> np.ndarray:
    return np.array([to_key(v) for v in col], dtype=str)


def _pandas_columns(df) -> Tuple[List[np.ndarray], List[str]]:
    import pandas as pd

    cols = []
    for name in df.columns:
        s = df[name]
        if isinstance(s, pd.DataFrame):          # duplicated column names
            s = s.iloc[:, 0]
        if pd.api.types.is_datetime64_any_dtype(s):
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert(None)
            cols.append(s.to_numpy(dtype="datetime64[ns]"))
        elif pd.api.types.is_timedelta64_dtype(s):
            cols.append(s.dt.total_seconds().to_numpy(dtype=float))
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            cols.append(s.astype("float64").to_numpy())
        else:
            obj = s.astype(object)
            cols.append(obj.where(s.notna(), None).to_numpy())
    return cols, [str(c) for c in df.columns]


def split_columns(X) -> Tuple[List[np.ndarray], Optional[List[str]], int]:
    """Return (list of 1-D column arrays, column names or None, n_rows)."""
    mod = type(X).__module__
    if mod.startswith("polars"):
        X = X.to_pandas()
        mod = "pandas"
    if mod.startswith("pandas"):
        if X.ndim == 1:                           # Series -> one column
            X = X.to_frame()
        cols, names = _pandas_columns(X)
        return cols, names, len(X)
    if hasattr(X, "toarray") and hasattr(X, "nnz"):   # scipy sparse
        X = X.toarray()
    if not isinstance(X, np.ndarray):
        try:
            X = np.asarray(X)
        except ValueError:                        # ragged / mixed rows
            X = np.asarray(X, dtype=object)
        if X.dtype.kind in "OUS" and X.ndim == 2:
            X = np.asarray(X, dtype=object)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return [X[:, j] for j in range(X.shape[1])], None, X.shape[0]
