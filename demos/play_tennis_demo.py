"""The course exercise: build the Play-Tennis tree with entropy + information gain."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaintree import GainTreeClassifier, shannon_entropy, gain_of_split, load_play_tennis

X, y, features = load_play_tennis()
Xa, ya = np.array(X, dtype=object), np.array(y)

print(f"Initial entropy H(T) = {shannon_entropy(ya):.3f}\n")
for j, name in enumerate(features):
    children = [ya[Xa[:, j] == v] for v in np.unique(Xa[:, j])]
    print(f"IG(T, {name:<8}) = {gain_of_split(ya, children):.3f}")

# 1) categorical (multiway) splits — the textbook ID3 tree
tree = GainTreeClassifier().fit(X, y, feature_names=features)
print("\nCategorical splits:\n" + tree.to_text(details=True))

# 2) label encoding -> features become ordinal numbers -> binary threshold splits
enc = np.column_stack([np.unique(Xa[:, j], return_inverse=True)[1] for j in range(4)])
tree_le = GainTreeClassifier().fit(enc, y, feature_names=features)
print(f"\nLabel-encoded: depth={tree_le.depth()}, leaves={tree_le.n_leaves()}")
print(tree_le.to_text())

fig, axes = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={"width_ratios": [1, 1.4]})
tree.draw(ax=axes[0]); axes[0].set_title("Categorical (multiway) splits")
tree_le.draw(ax=axes[1], fontsize=8); axes[1].set_title("Label-encoded (binary) splits")
fig.tight_layout()
fig.savefig("play_tennis_trees.png", dpi=130)
print("\nSaved play_tennis_trees.png")
