"""Run the tree on many kinds of datasets and report time + accuracy."""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from gaintree import GainTreeClassifier

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def messy_passengers(n=3000):
    """Titanic-like table: text, NaN, booleans, dates, ids, outliers."""
    sex = rng.choice(["male", "female"], n)
    cls = rng.choice([1, 2, 3], n, p=[.25, .25, .5])
    age = rng.normal(30, 14, n).clip(0.5, 80)
    fare = np.exp(rng.normal(3, 1, n)) * (4 - cls)
    p = 0.15 + 0.55 * (sex == "female") + 0.12 * (3 - cls) - 0.003 * age
    df = pd.DataFrame({
        "passenger_id": [f"P{i:05d}" for i in range(n)],
        "sex": sex, "pclass": cls, "age": age.round(1), "fare": fare.round(2),
        "port": rng.choice(["S", "C", "Q", None], n, p=[.6, .2, .15, .05]),
        "alone": rng.random(n) < .4,
        "booked": pd.Timestamp("1912-01-01") + pd.to_timedelta(rng.integers(0, 90, n), "D"),
        "survived": np.where(rng.random(n) < p, "yes", "no"),
    })
    df.loc[rng.random(n) < .2, "age"] = np.nan          # 20 % missing ages
    df.loc[rng.random(n) < .01, "fare"] = np.inf        # broken values
    return df.drop(columns="survived"), df["survived"]


cases = {
    "iris (numeric, 3 cls)": datasets.load_iris(return_X_y=True),
    "wine (numeric, 3 cls)": datasets.load_wine(return_X_y=True),
    "breast cancer (30 feat)": datasets.load_breast_cancer(return_X_y=True),
    "digits (64 feat, 10 cls)": datasets.load_digits(return_X_y=True),
    "messy passengers (mixed)": messy_passengers(),
    "100k rows x 20 feat": datasets.make_classification(100_000, 20, n_informative=8,
                                                        random_state=0),
    "20k rows, 50 classes": datasets.make_classification(20_000, 15, n_informative=10,
                                                         n_classes=50, n_clusters_per_class=1,
                                                         random_state=0),
}

print(f"{'dataset':<28}{'rows':>8}{'depth':>7}{'leaves':>8}{'fit s':>8}{'pred s':>8}{'test acc':>10}")
for name, (X, y) in cases.items():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for depth in [None]:
        t0 = time.perf_counter()
        tree = GainTreeClassifier(max_depth=depth, min_samples_leaf=1).fit(Xtr, ytr)
        t1 = time.perf_counter()
        acc = tree.score(Xte, yte)
        t2 = time.perf_counter()
        print(f"{name:<28}{len(y):>8}{tree.depth():>7}{tree.n_leaves():>8}"
              f"{t1 - t0:>8.2f}{t2 - t1:>8.2f}{acc:>10.3f}")

Xm, ym = messy_passengers()
Xm.assign(survived=ym).to_csv(__file__.replace("torture_test.py", "messy_passengers.csv"), index=False)
print("\nWrote messy_passengers.csv  ->  try:  gaintree messy_passengers.csv "
      "--target survived --max-depth 4")
