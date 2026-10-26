# gaintree — a classification tree grown by information gain

*[AIS] Statistical ML · Session 2 · Exercise 2*

- **Code:** https://github.com/premgudivada7/decision-tree-using-entropy
- **Package:** https://pypi.org/project/gaintree/ — `pip install gaintree`

---

## Summary

`gaintree` is a from-scratch classification tree (NumPy only). At each node it measures the Shannon entropy of the classes, tries every possible split, and keeps the one that removes the most entropy. The tree size is set by five arguments plus a pruning step. It prints itself as a text tree, draws itself with matplotlib, exports Graphviz, and ships as a pip-installable package with a command-line tool.

My main design goal beyond the brief was **"fit on any table without preparing it first"**: the classifier reads text, dates, booleans and missing values directly, which scikit-learn's tree cannot do for text columns.

| Asked for | Delivered |
|---|---|
| Decision tree from scratch | `GainTreeClassifier`, NumPy only |
| Entropy + information gain | `gaintree/info.py`, used for every split |
| Hyperparameters for tree size | `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_gain`, `max_leaves` + `prune()` |
| Bonus — display | `to_text()`, `draw()`, `to_dot()` |
| Bonus — PyPI | `pyproject.toml` builds a wheel that passes `twine check`; a GitHub Action publishes on each release |
| (own goal) Robust to any dataset | `gaintree/columns.py` + 25 tests on messy data + CLI |

---

## 1. Starting from the course exercise

The Play-Tennis table has 9 *Yes* and 5 *No*:

H(T) = −(9/14)·log₂(9/14) − (5/14)·log₂(5/14) = **0.940 bits**

Splitting on each attribute and subtracting the size-weighted entropy of the children gives the information gain IG(T, a) = H(T) − Σₖ (|Tₖ|/|T|)·H(Tₖ):

| Attribute | Children (Yes/No) | IG (bits) |
|---|---|---|
| **Outlook** | Sunny 2/3 · Overcast 4/0 · Rain 3/2 | **0.247** |
| Humidity | High 3/4 · Normal 6/1 | 0.152 |
| Wind | Weak 6/2 · Strong 3/3 | 0.048 |
| Temp | Hot 2/2 · Mild 4/2 · Cool 3/1 | 0.029 |

Outlook wins. Overcast is already pure; repeating the procedure inside Sunny picks Humidity and inside Rain picks Wind (both IG = 0.971, i.e. the child becomes pure). `gaintree` reproduces this tree exactly (left panel):

![trees](demos/play_tennis_trees.png)

**Encoding question from the exercise.** With *label encoding* (Overcast = 0, Rain = 1, Sunny = 2) the tree can only cut on order, so it must split Outlook twice to isolate the three values; the tree grows from **depth 2 / 5 leaves to depth 4 / 7 leaves** (right panel) and the cuts depend on an alphabetical order that means nothing. *One-hot encoding* removes the fake order but every split becomes "this category vs. all others", so isolating 3 values still takes two levels, and a column with many categories becomes many weak binary columns. Splitting natively on the category (what `gaintree` does for text) gives the shortest tree, at the price of fragmenting the data when there are many categories (see §5).

---

## 2. How the classifier is built

```
gaintree/
  info.py         surprise, shannon_entropy, entropy_of_counts, gain_of_split
  columns.py      turns any input into clean per-column arrays
  classifier.py   TreeNode + GainTreeClassifier (fit, predict, prune)
  render.py       to_text, to_dot, draw
  cli.py          `gaintree data.csv --target y`
  play_tennis.py  course dataset
```

**Searching a split on a number column.** Instead of testing thresholds one by one, the rows of the node are grouped by distinct value and a *class-count table* is built (distinct values × classes) with one `bincount`. A cumulative sum over that table gives, for every possible cut at once, the class counts on the low side; the high side is the total minus that. Entropy is computed on the whole table in one vectorised call, so all cuts of a column cost one sort. The cut is the midpoint between the two neighbouring values.

**Searching a split on a text column.** One branch per category; the gain uses the same count table.

**Growing.** The best split over all columns is applied if it gains more than `min_gain`, and its children go on a work list (an explicit stack, not recursion, so a tree thousands of levels deep cannot crash Python). With `max_leaves`, the work list becomes a priority queue ordered by *gain × rows*, so the most useful splits are made first.

**Predicting.** Rows travel down the tree as groups (boolean masks per node), not one by one, so predicting 30 000 rows takes about 0.1 s. A leaf predicts its most frequent class; `predict_proba` returns the class shares in the leaf.

**Pruning.** `prune(X_val, y_val)` sends a validation set through the tree, then visits nodes from the bottom up and turns a subtree into a leaf whenever the leaf would classify the validation rows at least as well.

---

## 3. Making it work on any table

Real data is rarely a clean float matrix, so `columns.py` inspects every column before training:

| Situation | Decision |
|---|---|
| Value is missing (`None`, `NaN`, `pd.NA`, `NaT`) in a number column | Cut is searched on the known values only; the gain is multiplied by the share of known values (C4.5's rule), and missing rows follow the bigger branch |
| Missing in a text column | `<missing>` becomes a category of its own |
| Numbers or ISO dates stored as text, real dates, durations, booleans | Converted to numbers |
| Anything else non-numeric | Treated as a category |
| Text column unique on (almost) every row — names, ids, e-mails | Skipped, with a warning. Information gain *loves* these (each row becomes a pure leaf, so the gain is maximal) but they predict nothing |
| `inf`, extreme floats | Cut computed as `lo + (hi − lo)/2`, falls back to `lo` if that overflows |
| Unknown category or unreadable value at prediction | Row stops at that node and takes its majority class |
| Labels missing | Clear error (cannot learn from them) |

The ID-column rule came from testing, not theory: on a synthetic "passengers" table, the first version split on `passenger_id` at the root and produced 2 100 single-row leaves with **55 %** test accuracy. Skipping it brought the same tree to 69 %, and capping depth at 4 to **74.5 %** with 19 leaves, using sex, class and age — the variables that actually generated the labels.

**Torture test** (`demos/torture_test.py`, 70/30 split, no size limit):

| Dataset | Rows | Fit time | Depth | Leaves | Test acc. |
|---|---|---|---|---|---|
| Iris | 150 | < 0.01 s | 4 | 7 | 0.978 |
| Wine | 178 | 0.02 s | 4 | 7 | 0.926 |
| Breast cancer | 569 | 0.07 s | 7 | 15 | 0.971 |
| Digits (64 pixels, 10 classes) | 1 797 | 0.7 s | 10 | 112 | 0.863 |
| Passengers (text, NaN, `inf`, dates, ids) | 3 000 | 0.3 s | 24 | 530 | 0.692 |
| Synthetic, 20 features | 100 000 | 8.6 s | 36 | 1 285 | 0.971 |
| Synthetic, 50 classes | 20 000 | 16.4 s | 19 | 6 088 | 0.274 |

The 50-class score is low because the data is hard: scikit-learn reaches 0.263 on it with the same 6 088 leaves.

---

## 4. Checking correctness

**Against scikit-learn** (`criterion="entropy"`, 5-fold stratified cross-validation; "same pred." is the share of test rows where both trees output the same class):

| Dataset | Setting | gaintree | sklearn | Same pred. |
|---|---|---|---|---|
| Iris | `max_depth=2` | 0.940 | 0.940 | 98.7 % |
| Wine | `max_leaves=8` | 0.938 | 0.938 | 98.9 % |
| Breast cancer | `min_samples_leaf=10` | 0.917 | 0.917 | 100 % |
| Breast cancer | unlimited | 0.912 | 0.928 | 97.0 % |
| Digits | `min_samples_leaf=10` | 0.813 | 0.814 | 99.2 % |
| Digits | unlimited | 0.852 | 0.850 | 92.5 % |

(16 configurations in `demos/vs_sklearn.py`, all within 0.02.) Remaining differences are **ties**: when two columns give exactly the same gain, scikit-learn chooses through a random column order while `gaintree` keeps the first; deep unlimited trees accumulate more ties, hence lower agreement there.

**Unit tests:** 50 `pytest` tests — entropy and gain values of the course table, the expected ID3 tree, every size argument, pruning, probabilities, importances, displays, and 25 tests that feed deliberately broken data (listed in §3). A GitHub Action runs them on Python 3.9, 3.11 and 3.13.

---

## 5. Controlling the size of the tree

Each argument stops growth at a different place:

| Argument | Stops a split when… |
|---|---|
| `max_depth` | the node is already at that level |
| `min_samples_split` | the node has too few rows |
| `min_samples_leaf` | one of the branches would be too small |
| `min_gain` | the split gains too few bits |
| `max_leaves` | the leaf budget is spent (best splits are made first) |

These are pre-pruning. **Post-pruning** experiment (`demos/pruning_demo.py`; 60 % train / 20 % validation / 20 % test, averaged over 10 random splits):

| Dataset | Leaves before → after | Test accuracy before → after |
|---|---|---|
| Wine | 6.8 → 4.1 | 0.914 → 0.883 |
| Breast cancer | 13.0 → 6.1 | 0.931 → 0.936 |
| Digits | 109 → 47 | 0.836 → 0.822 |

Pruning roughly **halves the tree**. On breast cancer this is free (accuracy even rises slightly). On wine and digits it costs 1–3 points, because the validation sets are small (36 and 360 rows) and the pruning decision overfits *them* — the same bias/variance trade-off from Session 1, moved to another dataset.

---

## 6. Limits

- Greedy: each split is the best *locally*; the tree is not globally optimal.
- Classification only (regression would replace entropy by variance).
- A text column with many — but not unique — categories still creates a very wide split and can overfit; C4.5's *gain ratio* or binary "category vs. rest" splits would fix this.
- About 5–10× slower than scikit-learn (pure Python/NumPy vs. compiled code), still interactive up to ~100 k rows.
- Natural extension for Session 3: bagging these trees into a random forest.

---

## 7. Reproducing everything

```bash
git clone https://github.com/premgudivada7/decision-tree-using-entropy && cd decision-tree-using-entropy
pip install -e ".[test]"
pytest -q
python demos/play_tennis_demo.py
python demos/vs_sklearn.py
python demos/pruning_demo.py
python demos/torture_test.py
gaintree demos/messy_passengers.csv --target survived --max-depth 4
```
