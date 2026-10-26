"""Sanity check against scikit-learn's DecisionTreeClassifier(criterion="entropy").

5-fold stratified cross-validation. For each fold we record both accuracies and
the share of test rows where the two trees predict the same class.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as Reference

from gaintree import GainTreeClassifier

SETTINGS = [{"max_depth": 2}, {"min_samples_leaf": 10}, {"max_leaves": 8}, {}]
SK_NAME = {"max_leaves": "max_leaf_nodes"}

print(f"{'dataset':<15}{'setting':<22}{'gaintree':>10}{'sklearn':>10}{'same pred.':>12}")
for name, loader in [("iris", load_iris), ("wine", load_wine),
                     ("breast_cancer", load_breast_cancer), ("digits", load_digits)]:
    X, y = loader(return_X_y=True)
    for kw in SETTINGS:
        ours_acc, ref_acc, agree = [], [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
            ours = GainTreeClassifier(**kw).fit(X[tr], y[tr])
            ref = Reference(criterion="entropy", random_state=0,
                            **{SK_NAME.get(k, k): v for k, v in kw.items()}).fit(X[tr], y[tr])
            p, q = ours.predict(X[te]), ref.predict(X[te])
            ours_acc.append(np.mean(p == y[te])); ref_acc.append(np.mean(q == y[te]))
            agree.append(np.mean(p == q))
        setting = ", ".join(f"{k}={v}" for k, v in kw.items()) or "unlimited"
        print(f"{name:<15}{setting:<22}{np.mean(ours_acc):>10.3f}{np.mean(ref_acc):>10.3f}"
              f"{np.mean(agree):>11.1%}")
