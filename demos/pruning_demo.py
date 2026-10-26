"""Does reduced-error pruning shrink the tree without hurting accuracy?

Split: 60 % train / 20 % validation (used only for pruning) / 20 % test. 10 random splits.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split

from gaintree import GainTreeClassifier

print(f"{'dataset':<15}{'leaves before':>15}{'leaves after':>14}{'test acc before':>17}{'after':>8}")
for name, loader in [("wine", load_wine), ("breast_cancer", load_breast_cancer),
                     ("digits", load_digits)]:
    X, y = loader(return_X_y=True)
    rows = []
    for seed in range(10):
        Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.25, random_state=seed, stratify=ya)
        t = GainTreeClassifier().fit(Xtr, ytr)
        before = (t.n_leaves(), t.score(Xte, yte))
        t.prune(Xva, yva)
        rows.append(before + (t.n_leaves(), t.score(Xte, yte)))
    lb, ab, la, aa = np.mean(rows, axis=0)
    print(f"{name:<15}{lb:>15.1f}{la:>14.1f}{ab:>17.3f}{aa:>8.3f}")
