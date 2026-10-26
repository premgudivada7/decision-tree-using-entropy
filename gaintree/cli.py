"""Command line: train the tree on any CSV file.

    python -m gaintree data.csv --target survived --max-depth 4 --plot tree.png
    gaintree data.csv --target survived            (after pip install)
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .columns import is_missing
from .classifier import GainTreeClassifier


def _read_csv(path: str, sep: str):
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=sep)
        return [str(c) for c in df.columns], df
    except ImportError:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f, delimiter=sep))
        header, body = rows[0], [r + [""] * (len(rows[0]) - len(r)) for r in rows[1:] if r]
        data = np.array([[None if v == "" else v for v in r] for r in body], dtype=object)
        return header, data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gaintree", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to a CSV file")
    ap.add_argument("--target", "-t", help="target column (default: last column)")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--drop", nargs="*", default=[], help="columns to ignore (ids, ...)")
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--min-samples-split", type=int, default=2)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--max-leaves", type=int)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-depth", type=int, default=4, help="depth printed in the console")
    ap.add_argument("--plot", help="save a picture of the tree (needs matplotlib)")
    a = ap.parse_args(argv)

    header, data = _read_csv(a.csv, a.sep)
    target = a.target or header[-1]
    if target not in header:
        ap.error(f"target {target!r} not in columns: {header}")
    keep = [c for c in header if c != target and c not in a.drop]
    if hasattr(data, "loc"):
        X, y = data[keep], data[target].to_numpy()
    else:
        X = data[:, [header.index(c) for c in keep]]
        y = data[:, header.index(target)]

    labelled = np.array([not is_missing(v) for v in y], dtype=bool)
    if not labelled.all():
        print(f"Dropping {int((~labelled).sum())} rows with no target value.")
        X = X[labelled] if not hasattr(X, "loc") else X.loc[labelled]
        y = y[labelled]

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(y))
    n_test = int(round(a.test_size * len(y))) if len(y) > 4 else 0
    te, tr = perm[:n_test], perm[n_test:]
    take = (lambda M, i: M.iloc[i]) if hasattr(X, "iloc") else (lambda M, i: M[i])

    tree = GainTreeClassifier(
        max_depth=a.max_depth, min_samples_split=a.min_samples_split,
        min_samples_leaf=a.min_samples_leaf, min_gain=a.min_gain,
        max_leaves=a.max_leaves,
    ).fit(take(X, tr), y[tr], feature_names=keep)

    print(f"rows={len(y)}  features={len(keep)}  classes={tree.n_classes_}")
    cats = [n for n, c in zip(keep, tree.categorical_mask_) if c]
    print(f"categorical features: {cats or 'none'}")
    print(f"depth={tree.depth()}  leaves={tree.n_leaves()}")
    print(f"train accuracy = {tree.score(take(X, tr), y[tr]):.3f}")
    if n_test:
        print(f"test accuracy  = {tree.score(take(X, te), y[te]):.3f}")
    imp = sorted(zip(tree.feature_importances_, keep), reverse=True)[:10]
    print("top features: " + ", ".join(f"{n} ({v:.2f})" for v, n in imp if v > 0))
    print()
    print(tree.to_text(max_depth=a.show_depth))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        ax = tree.draw()
        ax.figure.savefig(a.plot, dpi=120, bbox_inches="tight")
        print(f"\nSaved {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
