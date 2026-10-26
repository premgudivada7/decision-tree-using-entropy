"""Information theory building blocks."""
import numpy as np
import pytest

from gaintree import entropy_of_counts, gain_of_split, load_play_tennis, shannon_entropy, surprise


def test_surprise_is_log2_of_inverse_probability():
    assert surprise(0.25) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        surprise(0)


def test_entropy_of_counts_is_vectorised():
    out = entropy_of_counts(np.array([[5, 5], [10, 0], [0, 0]]))
    assert out.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_information_and_entropy_basics():
    assert surprise(0.5) == pytest.approx(1.0)
    assert surprise(1.0) == 0.0
    assert shannon_entropy(["a", "b"]) == pytest.approx(1.0)       # fair coin
    assert shannon_entropy(["a"] * 5) == 0.0                         # pure
    assert shannon_entropy([0, 1, 2, 3]) == pytest.approx(2.0)


def test_tennis_root_entropy_and_gains():
    X, y, f = load_play_tennis()
    X = np.array(X, dtype=object)
    y = np.array(y)
    assert shannon_entropy(y) == pytest.approx(0.940, abs=1e-3)      # 9 Yes / 5 No
    expected = {"Outlook": 0.247, "Temp": 0.029, "Humidity": 0.152, "Wind": 0.048}
    for j, name in enumerate(f):
        children = [y[X[:, j] == v] for v in np.unique(X[:, j])]
        assert gain_of_split(y, children) == pytest.approx(expected[name], abs=1e-3)


