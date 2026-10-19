import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiment import (AMPUTE_COL, FEATURES, N_REPEATS, SEED, TARGET, TEST_SIZE, U,
                        UTILITY_THRESHOLD, fit_H_model, fit_Y_model,
                        load_data, logreg, mar_mask, mar_score, p0_p1, q_of)

TRAIN_MISSING = [0.10, 0.30, 0.50]      # HighChol missing in the training set
TEST_MISSING = 0.30                     # HighChol missing in the test set
RHOS = [0.7, 0.9]                       # optional E2M rows: reliability of the source ([] = EM only)
TRAIN_SIZES = [7000, 500]               # all training rows, and a scarce setting
rng = np.random.default_rng(SEED + 7)


# EVIDENCE ON A MISSING VALUE as a contour function pl(h) on {0, 1}

def vacuous_evidence(n):
    """m({0,1}) = 1: nothing is known (a plain missing value)."""
    return np.ones(n), np.ones(n)                       # pl(0), pl(1)


def source_evidence(h_true, rho):
    """A source reports h_hat, correct with probability rho.
    Simple support function m({h_hat}) = s, m({0,1}) = 1 - s with
    s = (2 rho - 1) / rho, so that pl(other) / pl(h_hat) = (1 - rho) / rho.
    rho = 0.5 gives the vacuous mass, rho = 1 the categorical (observed) value."""
    h_hat = np.where(rng.random(len(h_true)) < rho, h_true, 1 - h_true)
    s = (2 * rho - 1) / rho
    pl_other = np.full(len(h_true), 1 - s)
    pl1 = np.where(h_hat == 1, 1.0, pl_other)
    pl0 = np.where(h_hat == 0, 1.0, pl_other)
    return pl0, pl1


# EM (and E2M) for the two logistic models

def fit_em(train, miss, pl0, pl1, max_iter=200, tol=1e-6):
    """Returns (H_model, Y_model, iterations, log-likelihood per iteration).
    miss: boolean mask of training rows whose HighChol is missing;
    pl0, pl1: evidence on those rows; pl0 = pl1 = 1 (vacuous) gives classical EM."""
    X, y, h = train[FEATURES].values, train[TARGET].values, train[AMPUTE_COL].values.astype(float)
    obs = ~miss
    Xm, ym = X[miss], y[miss]
    # stacked data: an observed row once, a missing row twice (as H=0 and as H=1)
    XH = np.vstack([X[obs], Xm, Xm])
    hH = np.r_[h[obs], np.zeros(len(Xm)), np.ones(len(Xm))]
    XY, yY = np.column_stack([XH, hH]), np.r_[y[obs], ym, ym]
    # start from the complete-case fit
    H_model, Y_model = logreg(X[obs], h[obs]), logreg(np.column_stack([X[obs], h[obs]]), y[obs])
    trace = []
    for it in range(1, max_iter + 1):
        # E-step: posterior weight of H = 1 (times the evidence pl, for E2M)
        q = H_model.predict_proba(Xm)[:, 1]
        p0, p1 = p0_p1(Y_model, train[miss])
        f1 = q * np.where(ym == 1, p1, 1 - p1) * pl1
        f0 = (1 - q) * np.where(ym == 1, p0, 1 - p0) * pl0
        w1 = f1 / (f0 + f1)
        # observed-data log-likelihood (generalised likelihood for E2M)
        qo = H_model.predict_proba(X[obs])[:, 1]
        po = Y_model.predict_proba(np.column_stack([X[obs], h[obs]]))[:, 1]
        ll = (np.log(np.where(h[obs] == 1, qo, 1 - qo)).sum()
              + np.log(np.where(y[obs] == 1, po, 1 - po)).sum() + np.log(f0 + f1).sum())
        trace.append(ll)
        if it > 1 and ll - trace[-2] < tol:
            break
        # M-step: weighted logistic regressions
        wts = np.r_[np.ones(obs.sum()), 1 - w1, w1]
        H_model, Y_model = logreg(XH, hH, wts), logreg(XY, yY, wts)
    return H_model, Y_model, it, np.array(trace)


def coefs(H_model, Y_model):
    return np.r_[H_model.intercept_, H_model.coef_.ravel(), Y_model.intercept_, Y_model.coef_.ravel()]



# Evaluate models on test records with missing HighChol 
def evaluate_models(H_model, Y_model, test_m, theta_full):
    y = test_m[TARGET].values
    q = q_of(H_model, test_m)
    p0, p1 = p0_p1(Y_model, test_m)
    p = (1 - q) * p0 + q * p1                                
    # test iff p > 1/8       
    d = (p > UTILITY_THRESHOLD).astype(int)                        
    th = coefs(H_model, Y_model)
    return {"coef_error": np.sqrt(np.mean((th - theta_full) ** 2)),  # RMSE vs full-data fit
            "beta_HighChol": Y_model.coef_.ravel()[-1],
            "log_loss": -np.mean(np.where(y == 1, np.log(p), np.log(1 - p))),
            "accuracy": np.mean((p > 0.5) == y),
            "utility": U[d, y].mean()}


# Implementation checks
def checks(df):
    train, _ = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET])
    train = train.reset_index(drop=True)
    miss = mar_mask(mar_score(train, train), 0.3, rng.random(len(train)))
    h = train[AMPUTE_COL].values[miss]
    full = coefs(fit_H_model(train), fit_Y_model(train))
    # (a) categorical evidence (rho = 1) must reproduce the full-data fit
    Hc, Yc, _, _ = fit_em(train, miss, *source_evidence(h, 1.0))
    # (b) EM never decreases the observed-data likelihood
    _, _, it, trace = fit_em(train, miss, *vacuous_evidence(miss.sum()))
    print(f"check (a) categorical evidence vs full data: max |diff| = {np.abs(coefs(Hc, Yc) - full).max():.1e}")
    print(f"check (b) EM: {it} iterations, smallest likelihood step = {np.diff(trace).min():.1e} (>= 0 up to rounding)")


# Experiment C: HighChol also missing in the training data, fitted with EM
def experiment_em(df):
    rows = []
    for rep in range(N_REPEATS):
        train, test = train_test_split(df, test_size=TEST_SIZE, random_state=int(rng.integers(1e9)),
                                       stratify=df[TARGET])
        test = test.reset_index(drop=True)
        test_m = test[mar_mask(mar_score(train, test), TEST_MISSING, rng.random(len(test)))]
        perm = rng.permutation(len(train))
        for n_train in TRAIN_SIZES:
            tr = train.iloc[perm[:n_train]].reset_index(drop=True)
            score, r = mar_score(tr, tr), rng.random(len(tr))       
            flips = rng.random(len(tr))                          
            full = (fit_H_model(tr), fit_Y_model(tr))
            theta_full = coefs(*full)
            for level in TRAIN_MISSING:
                miss = mar_mask(score, level, r)
                h = tr[AMPUTE_COL].values[miss]
                fits = {}
                t0 = time.perf_counter()
                fits["complete cases"] = (fit_H_model(tr[~miss]), fit_Y_model(tr[~miss]), 1)
                t_cc = time.perf_counter() - t0
                fits["full data"] = (*full, 1)
                times = {"complete cases": t_cc, "full data": np.nan}
                runs = [("EM", vacuous_evidence(miss.sum()))]
                for rho in RHOS:                                      
                    h_hat = np.where(flips[miss] < rho, h, 1 - h)     
                    s = (2 * rho - 1) / rho
                    runs.append((f"E2M rho={rho}", (np.where(h_hat == 0, 1.0, 1 - s),
                                                    np.where(h_hat == 1, 1.0, 1 - s))))
                for name, (pl0, pl1) in runs:
                    t0 = time.perf_counter()
                    Hm, Ym, it, _ = fit_em(tr, miss, pl0, pl1)
                    times[name] = time.perf_counter() - t0
                    fits[name] = (Hm, Ym, it)
                for name, (Hm, Ym, it) in fits.items():
                    rows.append({"n_train": n_train, "train_missing": level, "method": name,
                                 "repeat": rep, "iterations": it, "time_s": times[name],
                                 **evaluate_models(Hm, Ym, test_m, theta_full)})
    res = pd.DataFrame(rows)
    # paired differences against complete cases (same split, same masks)
    key = ["n_train", "train_missing", "repeat"]
    base = res[res.method == "complete cases"].set_index(key)
    idx = list(zip(*(res[k] for k in key)))
    res["logloss_gain_vs_cc"] = base.loc[idx, "log_loss"].values - res["log_loss"].values
    res["utility_gain_vs_cc"] = res["utility"].values - base.loc[idx, "utility"].values
    g = res.groupby(["n_train", "train_missing", "method"], sort=False)
    table = g.mean().drop(columns="repeat")
    table["coef_error_se"] = g["coef_error"].std() / np.sqrt(N_REPEATS)
    table["logloss_gain_se"] = g["logloss_gain_vs_cc"].std() / np.sqrt(N_REPEATS)
    table["utility_gain_se"] = g["utility_gain_vs_cc"].std() / np.sqrt(N_REPEATS)
    return table.reset_index()


def plot(table, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"complete cases": "#898781", "EM": "#2a78d6", "E2M rho=0.7": "#eb6834",
              "E2M rho=0.9": "#1baf7a"}
    fig, axes = plt.subplots(1, len(TRAIN_SIZES), figsize=(10, 3.4), sharey=False)
    for ax, n in zip(axes, TRAIN_SIZES):
        t = table[table.n_train == n]
        for name, g in t.groupby("method", sort=False):
            if name == "full data":
                continue
            ax.errorbar(100 * g.train_missing, g.coef_error, yerr=g.coef_error_se, marker="o", ms=4,
                        lw=1.8, capsize=3, label=name, color=colors[name])
        ax.set_title(f"{n:,} training rows", loc="left", fontsize=10)
        ax.set_xlabel("HighChol missing in the training set (%)")
        ax.set_ylabel("coefficient RMSE vs full-data fit")
        ax.set_xticks([100 * x for x in TRAIN_MISSING])
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(filename, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
    df = load_data()
    checks(df)
    table = experiment_em(df)
    table.to_csv("results/table_em.csv", index=False)
    print("\nEXPERIMENT C (test records with missing HighChol, probability branch, mean over 10 repeats)")
    print(table.to_string(index=False))
    plot(table, "results/fig_em.png")
    print("\nSaved results/table_em.csv and results/fig_em.png")
