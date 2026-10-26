"""GainTreeClassifier — a classification tree grown by maximising information gain.

How a node is split
    * number column -> two branches, "low" (x <= cut) and "high" (x > cut);
                       every midpoint between neighbouring distinct values is a candidate cut
    * text column   -> one branch per category (ID3 style)

Size controls: max_depth, min_samples_split, min_samples_leaf, min_gain,
max_leaves (best-first growth) and reduced-error pruning (`prune`).

Built for messy data: missing values, text, booleans, dates, inf, constant
columns, a single class, very deep trees and large tables.
"""

from __future__ import annotations

import heapq
import warnings
from dataclasses import dataclass, field
from itertools import count
from typing import Any, List, Optional, Sequence

import numpy as np

from .columns import column_to_float, column_to_keys, is_missing, split_columns
from .info import entropy_of_counts

_EPS = 1e-12


@dataclass(eq=False)
class TreeNode:
    """A node. `split_on is None` means it is terminal (a leaf)."""

    class_counts: np.ndarray            # how many training rows of each class reached it
    level: int                          # 0 for the trunk
    impurity: float                     # Shannon entropy of class_counts, in bits
    split_on: Optional[int] = None      # column index used for the split
    cut: Optional[float] = None         # number split: x <= cut -> low, else high
    nan_goes_low: bool = True           # number split: branch taken by missing values
    low: Optional["TreeNode"] = None
    high: Optional["TreeNode"] = None
    branches: dict = field(default_factory=dict)   # text split: category -> TreeNode
    info_gain: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.split_on is None

    @property
    def size(self) -> int:
        return int(self.class_counts.sum())

    @property
    def majority(self) -> int:
        """Index of the most frequent class (the vote of an impure leaf)."""
        return int(np.argmax(self.class_counts))

    def collapse(self) -> None:
        self.split_on = self.cut = self.low = self.high = None
        self.branches = {}
        self.info_gain = 0.0

    def edges(self):
        if self.cut is not None:
            yield f"<= {self.cut:.4g}", self.low
            yield f"> {self.cut:.4g}", self.high
        else:
            for value, child in self.branches.items():
                yield f"= {value}", child


class GainTreeClassifier:
    """Classification tree whose splits maximise information gain.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree (root = depth 0). None = grow until pure.
    min_samples_split : int
        Minimum samples a node needs before we try to split it.
    min_samples_leaf : int
        Minimum samples every child of a split must keep.
    min_gain : float
        A split is only accepted if its information gain is strictly above this.
    max_leaves : int or None
        If set, the tree is grown best-first (largest gain first) and stops
        once it has this many leaves.
    categorical : "auto", "none" or list of int / str
        Which columns get multiway categorical splits.
        "auto" = every column that is not numeric (e.g. text).
        "none" = treat everything as numeric (label-encoded data).
    skip_id_columns : bool
        Skip categorical columns where almost every value is unique (names,
        ids, emails...). Information gain always favours them (each row gets
        its own pure leaf) but they never generalise.

    Missing values
    --------------
    Numerical: the split is learned on the known values (gain scaled by the
    share of known values, as in C4.5); missing values follow the larger child.
    Categorical: "missing" is treated as its own category.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 0.0,
        max_leaves: Optional[int] = None,
        categorical: Any = "auto",
        skip_id_columns: bool = True,
    ):
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if max_leaves is not None and max_leaves < 1:
            raise ValueError("max_leaves must be >= 1 or None")
        if min_gain < 0:
            raise ValueError("min_gain must be >= 0")
        for name, value in [("max_depth", max_depth), ("min_samples_split", min_samples_split),
                            ("min_samples_leaf", min_samples_leaf), ("min_gain", min_gain),
                            ("max_leaves", max_leaves), ("categorical", categorical),
                            ("skip_id_columns", skip_id_columns)]:
            setattr(self, name, value)

    # ------------------------------------------------------------------ #
    # sklearn-like helpers
    # ------------------------------------------------------------------ #
    def get_params(self, deep: bool = True) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain": self.min_gain,
            "max_leaves": self.max_leaves,
            "categorical": self.categorical,
            "skip_id_columns": self.skip_id_columns,
        }

    def set_params(self, **params) -> "GainTreeClassifier":
        for k, v in params.items():
            if k not in self.get_params():
                raise ValueError(f"Unknown parameter {k!r}")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"GainTreeClassifier({args})"

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def _columns(self, X, fitting: bool) -> List[np.ndarray]:
        cols, names, n = split_columns(X)
        if fitting:
            return cols, names, n
        if (names is not None and self._names_from_frame
                and names != self.feature_names_
                and set(names) >= set(self.feature_names_)):
            lookup = {nm: c for nm, c in zip(names, cols)}   # reorder by name
            cols = [lookup[nm] for nm in self.feature_names_]
        if len(cols) != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {len(cols)}")
        return cols, n

    def _resolve_categorical(self, cols) -> np.ndarray:
        spec = self.categorical
        p = len(cols)
        mask = np.zeros(p, dtype=bool)
        if isinstance(spec, str) and spec == "auto":
            for j, c in enumerate(cols):
                mask[j] = column_to_float(c, strict=True) is None
        elif spec is None or (isinstance(spec, str) and spec == "none"):
            pass
        else:
            for s in ([spec] if isinstance(spec, (str, int)) else spec):
                if isinstance(s, str):
                    if s not in self.feature_names_:
                        raise ValueError(f"Unknown feature name {s!r} in categorical")
                    mask[self.feature_names_.index(s)] = True
                else:
                    mask[int(s)] = True
        for j in np.where(~mask)[0]:
            if column_to_float(cols[j], strict=True) is None:
                raise ValueError(
                    f"Feature {self.feature_names_[j]!r} is not numeric; "
                    "declare it in categorical (or use 'auto')."
                )
        return mask

    def _encode(self, cols, strict: bool):
        """numeric columns -> float arrays, categorical columns -> str arrays."""
        out = []
        for j, c in enumerate(cols):
            if self.categorical_mask_[j]:
                out.append(column_to_keys(c))
            else:
                out.append(column_to_float(c, strict=False))
        return out

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y, feature_names: Optional[Sequence[str]] = None):
        cols, names, n = self._columns(X, fitting=True)
        y = np.asarray(y, dtype=object if np.asarray(y).dtype.kind == "O" else None)
        if y.ndim == 2 and 1 in y.shape:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be 1-D (one label per row)")
        if len(y) != n:
            raise ValueError(f"X has {n} rows but y has {len(y)}")
        if n == 0:
            raise ValueError("Cannot fit on an empty dataset")
        missing_y = np.array([is_missing(v) for v in y]) if y.dtype.kind in "Of" \
            else np.zeros(n, bool)
        if missing_y.any():
            raise ValueError(f"y contains {int(missing_y.sum())} missing labels; "
                             "drop those rows before fitting.")
        if y.dtype.kind == "O":                    # mixed label types -> compare as text
            if len({type(v) for v in y}) > 1:
                y = np.array([str(v) for v in y])
        if y.dtype.kind == "f" and not np.all(np.mod(y, 1) == 0) \
                and len(np.unique(y)) > max(20, 0.5 * n):
            warnings.warn("y looks continuous: every distinct value becomes a class. "
                          "This is a classifier, not a regressor.", stacklevel=2)

        self.n_features_in_ = len(cols)
        self._names_from_frame = names is not None and feature_names is None
        if feature_names is not None:
            if len(feature_names) != len(cols):
                raise ValueError("feature_names has the wrong length")
            self.feature_names_ = [str(f) for f in feature_names]
        else:
            self.feature_names_ = names or [f"X[{j}]" for j in range(len(cols))]
        self.categorical_mask_ = self._resolve_categorical(cols)
        labels, codes = np.unique(y, return_inverse=True)
        self.classes_, self.n_classes_ = labels, labels.size

        self._cols = self._encode(cols, strict=True)
        self._y = codes.astype(np.int64)

        self.ignored_features_ = []
        if self.skip_id_columns:
            for j in np.where(self.categorical_mask_)[0]:
                n_unique = len(np.unique(self._cols[j]))
                if n_unique > max(20, 0.5 * n):
                    self.ignored_features_.append(self.feature_names_[j])
            if self.ignored_features_:
                warnings.warn(f"Ignoring id-like columns (almost all values unique): "
                              f"{self.ignored_features_}. Set skip_id_columns=False to keep "
                              "them.", stacklevel=2)
        self._skip = {self.feature_names_.index(f) for f in self.ignored_features_}

        idx = np.arange(n)
        self.trunk_ = self._make_node(idx, depth=0)
        if self.max_leaves is None:
            self._grow_depth_first(idx)
        else:
            self._grow_best_first(idx)
        del self._cols, self._y, self._skip
        return self

    def _make_node(self, idx: np.ndarray, depth: int) -> TreeNode:
        counts = np.bincount(self._y[idx], minlength=self.n_classes_)
        return TreeNode(class_counts=counts, level=depth, impurity=float(entropy_of_counts(counts)))

    def _can_split(self, node: TreeNode) -> bool:
        if node.impurity <= _EPS:                       # pure node
            return False
        if self.max_depth is not None and node.level >= self.max_depth:
            return False
        return node.size >= max(self.min_samples_split, 2 * self.min_samples_leaf)

    # -- growth strategies (iterative: no recursion limit on deep trees) -- #
    def _grow_depth_first(self, idx: np.ndarray) -> None:
        stack = [(self.trunk_, idx)]
        while stack:
            node, node_idx = stack.pop()
            if not self._can_split(node):
                continue
            split = self._best_split(node_idx, node)
            if split is not None:
                stack.extend(self._apply_split(node, node_idx, split))

    def _grow_best_first(self, idx: np.ndarray) -> None:
        tie = count()
        heap = []
        n_leaves = 1

        def push(node, node_idx):
            if self._can_split(node):
                split = self._best_split(node_idx, node)
                if split is not None:
                    weighted_gain = split[0] * len(node_idx)
                    heapq.heappush(heap, (-weighted_gain, next(tie), node, node_idx, split))

        push(self.trunk_, idx)
        while heap:
            _, _, node, node_idx, split = heapq.heappop(heap)
            n_children = 2 if split[2] is not None else len(split[3])
            if n_leaves - 1 + n_children > self.max_leaves:
                continue
            n_leaves += n_children - 1
            for child, child_idx in self._apply_split(node, node_idx, split):
                push(child, child_idx)

    # -- split search ----------------------------------------------------- #
    def _best_split(self, idx: np.ndarray, node: TreeNode):
        """Return (gain, feature, cut, categories, nan_low) or None."""
        y = self._y[idx]
        best, best_gain = None, self.min_gain
        for j in range(self.n_features_in_):
            if j in self._skip:
                continue
            if self.categorical_mask_[j]:
                cand = self._best_categorical(idx, y, j, node.impurity)
            else:
                cand = self._best_numeric(idx, y, j)
            if cand is not None and cand[0] > best_gain + _EPS:
                best_gain = cand[0]
                best = (cand[0], j) + cand[1:]
        return best

    def _best_numeric(self, idx, y, j):
        x = self._cols[j][idx]
        known = ~np.isnan(x)
        n, n_known = len(x), int(known.sum())
        if n_known < 2 * self.min_samples_leaf:
            return None
        if n_known < n:
            x, y = x[known], y[known]
        # class counts per distinct value: (u, k) — memory O(distinct * classes)
        vals, inv = np.unique(x, return_inverse=True)
        u, k = len(vals), self.n_classes_
        if u < 2:
            return None
        counts = np.bincount(inv * k + y, minlength=u * k).reshape(u, k)
        total = counts.sum(axis=0)
        low_counts = np.cumsum(counts, axis=0)[:-1]
        high_counts = total - low_counts
        n_low = low_counts.sum(axis=1)
        n_high = n_known - n_low
        valid = (n_low >= self.min_samples_leaf) & (n_high >= self.min_samples_leaf)
        if not valid.any():
            return None
        child_h = (n_low * entropy_of_counts(low_counts)
                   + n_high * entropy_of_counts(high_counts)) / n_known
        gains = (n_known / n) * (float(entropy_of_counts(total)) - child_h)
        gains = np.where(valid, gains, -np.inf)
        i = int(np.argmax(gains))
        lo, hi = vals[i], vals[i + 1]
        with np.errstate(over="ignore", invalid="ignore"):
            t = lo + (hi - lo) / 2.0
        if not (np.isfinite(t) and lo <= t < hi):  # inf values / float rounding
            t = lo
        return float(gains[i]), float(t), None, bool(n_low[i] >= n_high[i])

    def _best_categorical(self, idx, y, j, parent_entropy):
        keys = self._cols[j][idx]
        values, inv = np.unique(keys, return_inverse=True)
        u, k = len(values), self.n_classes_
        if u < 2:
            return None
        counts = np.bincount(inv * k + y, minlength=u * k).reshape(u, k)
        sizes = counts.sum(axis=1)
        if sizes.min() < self.min_samples_leaf:
            return None
        child_h = (sizes * entropy_of_counts(counts)).sum() / len(y)
        return float(parent_entropy - child_h), None, list(values), True

    def _apply_split(self, node: TreeNode, idx: np.ndarray, split):
        gain, j, cut, categories, nan_low = split
        node.split_on, node.info_gain = j, gain
        if cut is not None:
            x = self._cols[j][idx]
            to_low = (x <= cut) | (np.isnan(x) & nan_low)
            node.cut, node.nan_goes_low = cut, nan_low
            node.low = self._make_node(idx[to_low], node.level + 1)
            node.high = self._make_node(idx[~to_low], node.level + 1)
            return [(node.low, idx[to_low]), (node.high, idx[~to_low])]
        keys = self._cols[j][idx]
        out = []
        for value in categories:
            child_idx = idx[keys == value]
            child = self._make_node(child_idx, node.level + 1)
            node.branches[value] = child
            out.append((child, child_idx))
        return out

    # ------------------------------------------------------------------ #
    # Predict (vectorised: whole groups of rows move down the tree)
    # ------------------------------------------------------------------ #
    def _check_fitted(self):
        if not hasattr(self, "trunk_"):
            raise RuntimeError("This GainTreeClassifier is not fitted yet. Call fit first.")

    def _route(self, node: TreeNode, cols, rows):
        """Split `rows` between the children of `node`.
        Returns ([(child, rows)], rows_that_stop_here)."""
        j = node.split_on
        if node.cut is not None:
            x = cols[j][rows]
            to_low = (x <= node.cut) | (np.isnan(x) & node.nan_goes_low)
            return [(node.low, rows[to_low]), (node.high, rows[~to_low])], rows[:0]
        keys = cols[j][rows]
        routed, matched = [], np.zeros(len(rows), bool)
        for value, child in node.branches.items():
            m = keys == value
            matched |= m
            routed.append((child, rows[m]))
        return routed, rows[~matched]            # unseen category -> stop here

    def apply(self, X) -> np.ndarray:
        """Return the TreeNode each sample ends up in (object array)."""
        self._check_fitted()
        cols, n = self._columns(X, fitting=False)
        cols = self._encode(cols, strict=False)
        out = np.empty(n, dtype=object)
        stack = [(self.trunk_, np.arange(n))]
        while stack:
            node, rows = stack.pop()
            if len(rows) == 0:
                continue
            if node.is_terminal:
                out[rows] = node
                continue
            routed, stopped = self._route(node, cols, rows)
            out[stopped] = node
            stack.extend(routed)
        return out

    def predict_proba(self, X) -> np.ndarray:
        nodes = self.apply(X)
        if len(nodes) == 0:
            return np.empty((0, self.n_classes_))
        counts = np.array([n.class_counts for n in nodes], dtype=float)
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        nodes = self.apply(X)
        return self.classes_[np.array([n.majority for n in nodes], dtype=int)]

    def score(self, X, y) -> float:
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if pred.dtype != y.dtype:
            return float(np.mean(pred.astype(str) == y.astype(str)))
        return float(np.mean(pred == y))

    # ------------------------------------------------------------------ #
    # Size / structure
    # ------------------------------------------------------------------ #
    def walk(self, node=None):
        stack = [node or self.trunk_]
        while stack:
            cur = stack.pop()
            yield cur
            if not cur.is_terminal:
                stack.extend(c for _, c in cur.edges())

    @property
    def feature_importances_(self) -> np.ndarray:
        """Share of the total (sample-weighted) information gain brought by each feature."""
        self._check_fitted()
        imp = np.zeros(self.n_features_in_)
        for n in self.walk():
            if not n.is_terminal:
                imp[n.split_on] += n.info_gain * n.size
        total = imp.sum()
        return imp / total if total > 0 else imp

    def depth(self) -> int:
        self._check_fitted()
        return max(n.level for n in self.walk())

    def n_leaves(self) -> int:
        self._check_fitted()
        return sum(n.is_terminal for n in self.walk())

    def n_nodes(self) -> int:
        self._check_fitted()
        return sum(1 for _ in self.walk())

    # ------------------------------------------------------------------ #
    # Reduced-error post-pruning (iterative, bottom-up)
    # ------------------------------------------------------------------ #
    def prune(self, X_val, y_val) -> "GainTreeClassifier":
        """Turn every internal node into a leaf if that does not lower
        accuracy on the validation set (processed from the bottom up)."""
        self._check_fitted()
        cols, n = self._columns(X_val, fitting=False)
        cols = self._encode(cols, strict=False)
        y_val = np.asarray(y_val).ravel()
        lookup = {str(c): i for i, c in enumerate(self.classes_)}
        y_enc = np.array([lookup.get(str(v), -1) for v in y_val])
        if len(y_enc) != n:
            raise ValueError("X_val and y_val have different numbers of rows")

        # top-down: which validation rows reach each node (pre-order list)
        order, rows_at, stopped_at = [], {}, {}
        stack = [(self.trunk_, np.arange(n))]
        while stack:
            node, rows = stack.pop()
            order.append(node)
            rows_at[id(node)] = rows
            if not node.is_terminal:
                routed, stopped = self._route(node, cols, rows)
                stopped_at[id(node)] = stopped
                stack.extend(routed)

        # bottom-up: children are always processed before their parent
        correct = {}
        for node in reversed(order):
            rows = rows_at[id(node)]
            leaf_correct = int(np.sum(y_enc[rows] == node.majority))
            if node.is_terminal:
                correct[id(node)] = leaf_correct
                continue
            sub = sum(correct[id(c)] for _, c in node.edges())
            sub += int(np.sum(y_enc[stopped_at[id(node)]] == node.majority))
            if leaf_correct >= sub:
                node.collapse()
                correct[id(node)] = leaf_correct
            else:
                correct[id(node)] = sub
        return self
