"""
Engine health analytics on the NASA C-MAPSS turbofan dataset (FD001).

Two parts:
  A) Remaining Useful Life (RUL) prediction WITH uncertainty
     -> quantile gradient boosting gives a 10-90 % prediction interval,
        and we check how often the true value really falls inside it.
  B) Virtual sensor
     -> estimate one physical sensor from the others, then use the
        residual (measured - estimated) as a simple health/fault indicator.

Validation is done per ENGINE (GroupKFold), never per row, so the model is
always tested on engines it has never seen. This avoids data leakage.

Usage:
    python engine_health.py --data data            # real NASA files
    python engine_health.py --synthetic            # quick test without data
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125          # common practice: RUL is flat early in life
VIRTUAL_TARGET = "s9"  # physical core speed, estimated from the other sensors
OUT = "results"


# ---------------------------------------------------------------- data
def load_cmapss(folder, fd="FD001"):
    read = lambda f: pd.read_csv(os.path.join(folder, f), sep=r"\s+", header=None, names=COLS)
    train = read(f"train_{fd}.txt")
    test = read(f"test_{fd}.txt")
    rul_true = pd.read_csv(os.path.join(folder, f"RUL_{fd}.txt"), header=None)[0].values
    return train, test, rul_true


def make_synthetic(n_units=100, seed=0):
    """Fake data with the same columns, only for testing that the code runs."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = rng.integers(130, 330)
        for c in range(1, life + 1):
            wear = (c / life) ** 2
            base = rng.normal(0, 1, 21) * 0.3
            sens = 10 + base + wear * rng.uniform(0.5, 3, 21)
            sens[8] = 9000 + 40 * wear + 0.8 * (sens[1] + sens[2]) + rng.normal(0, 2)
            rows.append([u, c, *rng.normal(0, 0.002, 3), *sens])
    train = pd.DataFrame(rows, columns=COLS)
    test_rows, rul_true = [], []
    for u, g in train.groupby("unit"):
        cut = rng.integers(30, len(g) - 5)
        test_rows.append(g.iloc[:cut])
        rul_true.append(len(g) - cut)
    return train, pd.concat(test_rows), np.array(rul_true)


def add_rul(df):
    max_c = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = np.minimum(max_c - df["cycle"], RUL_CAP)
    return df


def useful_sensors(df):
    """Drop sensors that are (almost) constant - they carry no information."""
    return [c for c in df.columns if c.startswith("s") and df[c].std() > 1e-3]


def add_rolling(df, sensors, w=10):
    """Smooth noisy sensors per engine with a rolling mean."""
    df = df.sort_values(["unit", "cycle"]).copy()
    for s in sensors:
        df[f"{s}_rm"] = df.groupby("unit")[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
    return df


# ---------------------------------------------------------------- part A
def quantile_model(alpha):
    return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=200,
                                         max_depth=4, learning_rate=0.05, random_state=0)


def part_a_rul(train, test, rul_true, feats):
    X, y, groups = train[feats], train["RUL"], train["unit"]

    # cross-validation per engine
    cv_rmse, cv_cov = [], []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        lo, md, hi = (quantile_model(a).fit(X.iloc[tr], y.iloc[tr]) for a in (0.1, 0.5, 0.9))
        p_lo, p_md, p_hi = lo.predict(X.iloc[va]), md.predict(X.iloc[va]), hi.predict(X.iloc[va])
        cv_rmse.append(mean_squared_error(y.iloc[va], p_md) ** 0.5)
        cv_cov.append(np.mean((y.iloc[va] >= p_lo) & (y.iloc[va] <= p_hi)))

    # final model on all training engines, evaluated on the official test set
    lo, md, hi = (quantile_model(a).fit(X, y) for a in (0.1, 0.5, 0.9))
    last = test.groupby("unit").tail(1)
    p_lo, p_md, p_hi = lo.predict(last[feats]), md.predict(last[feats]), hi.predict(last[feats])
    y_t = np.minimum(rul_true, RUL_CAP)
    res = {
        "cv_rmse": np.mean(cv_rmse),
        "cv_interval_coverage": np.mean(cv_cov),
        "test_rmse": mean_squared_error(y_t, p_md) ** 0.5,
        "test_mae": mean_absolute_error(y_t, p_md),
        "test_interval_coverage": np.mean((y_t >= p_lo) & (y_t <= p_hi)),
        "mean_interval_width": np.mean(p_hi - p_lo),
    }

    # plot: prediction with uncertainty band, sorted by true RUL
    o = np.argsort(y_t)
    plt.figure(figsize=(9, 4.5))
    plt.fill_between(range(len(o)), p_lo[o], p_hi[o], alpha=0.3, label="10-90 % interval")
    plt.plot(p_md[o], label="Predicted RUL (median)")
    plt.plot(y_t[o], "k.", label="True RUL")
    plt.xlabel("Test engines (sorted by true RUL)")
    plt.ylabel("Remaining useful life [cycles]")
    plt.title("RUL prediction with uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rul_prediction_interval.png"), dpi=150)
    plt.close()

    # importance (which sensors drive the prediction)
    pi = permutation_importance(md, last[feats], y_t, n_repeats=5, random_state=0)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False).head(10)
    imp.to_csv(os.path.join(OUT, "rul_feature_importance.csv"))
    return res


# ---------------------------------------------------------------- part B
def part_b_virtual_sensor(train, sensors):
    target = VIRTUAL_TARGET
    inputs = [s for s in sensors if s != target]
    healthy = train[train["RUL"] >= RUL_CAP]           # early life = healthy reference
    X, y, groups = healthy[inputs], healthy[target], healthy["unit"]

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    errs = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        errs.append(mean_absolute_error(y.iloc[va], model.predict(X.iloc[va])))
    model.fit(X, y)

    # residual over the whole life of each engine = health indicator
    df = train.copy()
    df["residual"] = df[target] - model.predict(df[inputs])
    sigma = (healthy[target] - model.predict(X)).std()
    df["life_frac"] = df["cycle"] / df.groupby("unit")["cycle"].transform("max")
    bins = pd.cut(df["life_frac"], np.linspace(0, 1, 11))
    trend = df.groupby(bins, observed=True)["residual"].mean()

    plt.figure(figsize=(9, 4.5))
    for u in df["unit"].unique()[:8]:
        g = df[df["unit"] == u]
        plt.plot(g["life_frac"], g["residual"].rolling(10, min_periods=1).mean(), alpha=0.7)
    plt.axhline(3 * sigma, color="r", ls="--", label="+3 sigma (healthy)")
    plt.axhline(-3 * sigma, color="r", ls="--")
    plt.xlabel("Fraction of engine life")
    plt.ylabel(f"{target}: measured - virtual sensor")
    plt.title("Virtual sensor residual grows as the engine degrades")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "virtual_sensor_residual.png"), dpi=150)
    plt.close()

    return {
        "virtual_sensor_target": target,
        "healthy_cv_mae": np.mean(errs),
        "healthy_residual_sigma": sigma,
        "residual_early_life": trend.iloc[0],
        "residual_end_of_life": trend.iloc[-1],
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    train, test, rul_true = make_synthetic() if a.synthetic else load_cmapss(a.data)
    train = add_rul(train)
    sensors = useful_sensors(train)
    train, test = add_rolling(train, sensors), add_rolling(test, sensors)
    feats = sensors + [f"{s}_rm" for s in sensors]

    res = {**part_a_rul(train, test, rul_true, feats), **part_b_virtual_sensor(train, sensors)}
    lines = [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in res.items()]
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nPlots and summary saved in ./{OUT}/")


if __name__ == "__main__":
    main()
