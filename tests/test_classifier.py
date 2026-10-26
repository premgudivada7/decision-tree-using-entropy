import numpy as np
import pytest

from gaintree import GainTreeClassifier, load_play_tennis


# ---------- tree on the course example -------------------------------------- #
def test_tennis_tree_is_the_classic_id3_tree():
    X, y, f = load_play_tennis()
    t = GainTreeClassifier().fit(X, y, feature_names=f)
    assert f[t.trunk_.split_on] == "Outlook"
    assert t.trunk_.branches["Overcast"].is_terminal
    assert f[t.trunk_.branches["Sunny"].split_on] == "Humidity"
    assert f[t.trunk_.branches["Rain"].split_on] == "Wind"
    assert t.depth() == 2 and t.n_leaves() == 5
    assert t.score(X, y) == 1.0


def test_label_encoded_tennis_uses_binary_splits():
    X, y, f = load_play_tennis()
    X = np.array(X)
    enc = np.column_stack([np.unique(X[:, j], return_inverse=True)[1] for j in range(4)])
    t = GainTreeClassifier().fit(enc, y, feature_names=f)
    assert t.trunk_.cut is not None
    assert t.score(enc, y) == 1.0


def test_unseen_category_falls_back_to_node_majority():
    X, y, f = load_play_tennis()
    t = GainTreeClassifier().fit(X, y, feature_names=f)
    assert t.predict([["Snow", "Hot", "High", "Weak"]])[0] == "Yes"  # root majority


# ---------- size hyperparameters -------------------------------------------- #
@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] > 0.5)).astype(int)
    return X, y


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5])
def test_depth_never_exceeds_cap(data, depth):
    t = GainTreeClassifier(max_depth=depth).fit(*data)
    assert t.depth() <= depth


def test_every_leaf_keeps_enough_rows(data):
    t = GainTreeClassifier(min_samples_leaf=25).fit(*data)
    assert all(n.size >= 25 for n in t.walk() if n.is_terminal)


def test_small_nodes_stay_unsplit(data):
    t = GainTreeClassifier(min_samples_split=60).fit(*data)
    assert all(n.size >= 60 for n in t.walk() if not n.is_terminal)


@pytest.mark.parametrize("k", [1, 2, 5, 12])
def test_leaf_budget_is_respected(data, k):
    t = GainTreeClassifier(max_leaves=k).fit(*data)
    assert t.n_leaves() <= k


def test_weak_splits_are_rejected(data):
    t = GainTreeClassifier(min_gain=0.05).fit(*data)
    assert all(n.info_gain > 0.05 for n in t.walk() if not n.is_terminal)


def test_unlimited_tree_fits_training_set(data):
    X, y = data
    assert GainTreeClassifier().fit(X, y).score(X, y) == 1.0


def test_pruning_shrinks_without_hurting_validation(data):
    X, y = data
    rng = np.random.default_rng(1)
    y_noisy = np.where(rng.random(len(y)) < 0.15, 1 - y, y)
    Xtr, ytr, Xva, yva = X[:250], y_noisy[:250], X[250:], y_noisy[250:]
    t = GainTreeClassifier().fit(Xtr, ytr)
    before_leaves, before_acc = t.n_leaves(), t.score(Xva, yva)
    t.prune(Xva, yva)
    assert t.n_leaves() < before_leaves
    assert t.score(Xva, yva) >= before_acc


# ---------- API ------------------------------------------------------------- #
def test_predict_proba_and_importances(data):
    X, y = data
    t = GainTreeClassifier(max_depth=3).fit(X, y)
    p = t.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(axis=1), 1)
    assert t.feature_importances_.sum() == pytest.approx(1.0)
    assert t.feature_importances_[3] == pytest.approx(0.0)   # pure noise feature


def test_string_labels_and_pandas():
    pd = pytest.importorskip("pandas")
    X, y, f = load_play_tennis()
    df = pd.DataFrame(X, columns=f)
    t = GainTreeClassifier(max_depth=1).fit(df, y)
    assert t.feature_names_ == f
    assert set(t.predict(df)) <= {"Yes", "No"}


def test_errors():
    with pytest.raises(ValueError):
        GainTreeClassifier(min_samples_leaf=0)
    with pytest.raises(RuntimeError):
        GainTreeClassifier().predict([[1.0]])
    with pytest.raises(ValueError):   # string column declared numeric
        GainTreeClassifier(categorical="none").fit([["a"], ["b"]], [0, 1])


def test_displays():
    X, y, f = load_play_tennis()
    t = GainTreeClassifier().fit(X, y, feature_names=f)
    txt = t.to_text()
    assert txt.splitlines()[0] == "[Outlook]" and "Overcast ⇒ Yes  (4/4)" in txt
    assert t.to_dot().startswith("digraph Tree")
    import matplotlib
    matplotlib.use("Agg")
    assert t.draw() is not None
