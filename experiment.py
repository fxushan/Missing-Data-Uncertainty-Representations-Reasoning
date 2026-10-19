import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Setup
FEATURES = ["HighBP", "Smoker", "PhysActivity", "GenHlth"]  # X_i
AMPUTE_COL = "HighChol"                                     # H_i (made missing)
TARGET = "Diabetes_binary"                                  # Y_i
LOCAL_CSV = "diabetes_binary_health_indicators_BRFSS2015.csv"

MISSING_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
N_BOOTSTRAP = 200          # B_boot
N_REPEATS = 10
TEST_SIZE = 0.30
SEED = 2025
MAX_ROWS = 10000
MAR_WEIGHTS = np.array([1.0, 0.6, -0.5, 0.8])   # HighBP, Smoker, PhysActivity, GenHlth
TRAIN_SIZES = (150, 300, 600, 1200, 2500, 5000)  # experiment B

U = np.array([[0.0, -5.0],    # U[d, y]: d = 0 no test
              [-1.0, 2.0]])   #          d = 1 test        (report Eq. 16)
UTILITY_THRESHOLD = 1 / 8     # test iff P(Y=1) > 1/8         (report Eq. 33)
CLASS_THRESHOLD = 0.5
MEASURE_COST = 0.05           # cost of measuring HighChol when a method abstains
ABSTAIN = -1

FCM_CLUSTERS = 6              # K
FCM_FUZZIFIER = 2.0           # a

rng = np.random.default_rng(SEED)   # same random-number order as prob_bf.py
os.makedirs("results", exist_ok=True)


#data
def load_data():
    try:
        df = pd.read_csv(LOCAL_CSV)
    except FileNotFoundError:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=891)
        df = ds.data.features.copy()
        df[TARGET] = ds.data.targets.iloc[:, 0].values
    df = df[FEATURES + [AMPUTE_COL, TARGET]].dropna()
    df = df.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return df.astype(int)


# MAR missingness
def mar_score(train, test):
    mu, sigma = train[FEATURES].mean(), train[FEATURES].std() + 1e-9
    return ((test[FEATURES] - mu) / sigma).values @ MAR_WEIGHTS


def calibrate_intercept(score, rate):
    """Bisection for b such that mean(sigmoid(b + score)) == rate."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + score)))).mean() < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mar_mask(score, rate, r):
    """True = HighChol missing. The same uniform draws r give nested masks."""
    p = 1 / (1 + np.exp(-(calibrate_intercept(score, rate) + score)))
    return r < p


# shared models and bootstrap refits
def logreg(X, y, weights=None):
    return LogisticRegression(C=np.inf, max_iter=3000).fit(X, y, sample_weight=weights)


def fit_H_model(df):              # P(H = 1 | X)
    return logreg(df[FEATURES].values, df[AMPUTE_COL].values)


def fit_Y_model(df):              # P(Y = 1 | X, H)
    return logreg(df[FEATURES + [AMPUTE_COL]].values, df[TARGET].values)


def q_of(H_model, df):
    return H_model.predict_proba(df[FEATURES].values)[:, 1]


def p0_p1(Y_model, df):
    """p_i0 and p_i1: Diabetes probability if HighChol were 0 or 1."""
    X = df[FEATURES].values
    p0 = Y_model.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1]
    p1 = Y_model.predict_proba(np.column_stack([X, np.ones(len(X))]))[:, 1]
    return p0, p1


def bootstrap_Q(train, test, n_boot):
    """Q[i, b] = q_i^(b): HighChol probability of test record i under refit b."""
    n, cols = len(train), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = train.iloc[idx]
        if sample[AMPUTE_COL].nunique() < 2:          # skip a degenerate resample
            continue
        cols.append(q_of(fit_H_model(sample), test))
    return np.column_stack(cols)


# four representations
def probability_rep(q):                      
    return q, q


def belief_rep(q, Q):                        
    u = np.percentile(Q, 95, axis=1) - np.percentile(Q, 5, axis=1)   
    m1 = (1 - u) * q                                                
    return m1, m1 + u                                             

#BetP(H=1) = m({1}) + u/2
def pignistic_rep(q, Q):                     
    bel, pl = belief_rep(q, Q)
    betp = (bel + pl) / 2
    return betp, betp

# convex hull of the refits
def credal_rep(Q):                            
    return Q.min(axis=1), Q.max(axis=1)


def fcm_fit(Z, rng_fcm, n_iter=100):
    """Fuzzy c-means on complete training rows; returns the prototypes v_k."""
    Um = rng_fcm.dirichlet(np.ones(FCM_CLUSTERS), size=len(Z))
    for _ in range(n_iter):
        W = Um ** FCM_FUZZIFIER
        V = (W.T @ Z) / W.sum(axis=0)[:, None]
        D = ((Z[:, None, :] - V[None, :, :]) ** 2).sum(-1) + 1e-12
        Um = memberships(D)
    return V


def memberships(D):                        
    inv = D ** (-1 / (FCM_FUZZIFIER - 1))
    return inv / inv.sum(axis=1, keepdims=True)


def fuzzy_rep(train, test, rng_fcm):         
    cols = FEATURES + [AMPUTE_COL]
    Z = train[cols].values.astype(float)
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    V = fcm_fit((Z - mu) / sd, rng_fcm)
    Zt = (test[FEATURES].values - mu[:-1]) / sd[:-1]          
    D = len(cols) / len(FEATURES) * ((Zt[:, None, :] - V[None, :, :-1]) ** 2).sum(-1) + 1e-12
    W = memberships(D) ** FCM_FUZZIFIER                          
    h_proto = V[:, -1] * sd[-1] + mu[-1]                         
    H_hat = (W @ h_proto) / W.sum(axis=1)                      
    H_bar = (H_hat >= 0.5).astype(float)                         
    return H_bar, H_bar


def to_diabetes(q_lo, q_hi, p0, p1):
    """[q_lo, q_hi] for P(H=1)  ->  [p_lo, p_hi] for P(Y=1) (Eq. 28, 37-38)."""
    a, b = (1 - q_lo) * p0 + q_lo * p1, (1 - q_hi) * p0 + q_hi * p1
    return np.minimum(a, b), np.maximum(a, b)


# Decision and Eval

def classify(p_lo, p_hi, rule):
    if rule == "gamma":                       # larger lower probability wins
        return (p_lo > 1 - p_hi).astype(int)
    return np.where(p_lo > CLASS_THRESHOLD, 1, np.where(p_hi < CLASS_THRESHOLD, 0, ABSTAIN))


def decide_test(p_lo, p_hi, rule):
    if rule == "gamma":                      
        return (3 * p_lo + 5 * p_hi > 1).astype(int)
    return np.where(p_lo > UTILITY_THRESHOLD, 1, np.where(p_hi < UTILITY_THRESHOLD, 0, ABSTAIN))


def evaluate(rep, rule, p0, p1, y, h_true):
    p_lo, p_hi = to_diabetes(*rep, p0, p1)
    c = classify(p_lo, p_hi, rule)
    d = decide_test(p_lo, p_hi, rule)
    c_ok, d_ok = c != ABSTAIN, d != ABSTAIN
    # if the method abstains on the test decision: measure HighChol, then decide
    p_true = np.where(h_true == 1, p1, p0)
    u_measured = U[(p_true > UTILITY_THRESHOLD).astype(int), y] - MEASURE_COST
    u_decided = U[np.where(d_ok, d, 0), y]
    return {
        "accuracy": np.mean(c[c_ok] == y[c_ok]),            
        "determinacy": c_ok.mean(),
        "test_determinacy": d_ok.mean(),
        "utility_decided": u_decided[d_ok].mean(),
        "utility_with_measuring": np.where(d_ok, u_decided, u_measured).mean(),
        "width_H": np.mean(rep[1] - rep[0]),
        "width_Y": np.mean(p_hi - p_lo),
    }


def all_methods(q, Q, fuzzy, h_true):
    """name -> (representation, decision rule)."""
    return {
        "Probability": (probability_rep(q), "interval"),
        "BF pignistic": (pignistic_rep(q, Q), "interval"),
        "BF cautious": (belief_rep(q, Q), "interval"),
        "Credal (dominance)": (credal_rep(Q), "interval"),
        "Credal (Gamma-maximin)": (credal_rep(Q), "gamma"),
        "Fuzzy": (fuzzy, "interval"),
        "Oracle (HighChol known)": ((h_true.astype(float), h_true.astype(float)), "interval"),
    }


# Experiment A
def experiment_missing_rate(df):
    rows, timing = [], []
    for rep in range(N_REPEATS):
        train, test = train_test_split(df, test_size=TEST_SIZE,
                                       random_state=int(rng.integers(1e9)), stratify=df[TARGET])
        test = test.reset_index(drop=True)

        t0 = time.perf_counter()
        H_model, Y_model = fit_H_model(train), fit_Y_model(train)
        t_models = time.perf_counter() - t0
        t0 = time.perf_counter()
        Q_all = bootstrap_Q(train, test, N_BOOTSTRAP)
        t_boot = time.perf_counter() - t0
        r = rng.random(len(test))                               # shared draws -> nested masks
        score = mar_score(train, test)
        t0 = time.perf_counter()
        fuzzy_all = fuzzy_rep(train, test, np.random.default_rng(SEED + rep))
        t_fcm = time.perf_counter() - t0
        timing.append({"logistic models (s)": t_models,
                       f"{N_BOOTSTRAP} bootstrap refits (s)": t_boot,
                       "fuzzy c-means (s)": t_fcm})

        q_all = q_of(H_model, test)
        p0_all, p1_all = p0_p1(Y_model, test)
        for level in MISSING_LEVELS:
            m = mar_mask(score, level, r)
            y, h = test[TARGET].values[m], test[AMPUTE_COL].values[m]
            fz = (fuzzy_all[0][m], fuzzy_all[1][m])
            for name, (repr_, rule) in all_methods(q_all[m], Q_all[m], fz, h).items():
                rows.append({"missing": level, "method": name, "repeat": rep,
                             **evaluate(repr_, rule, p0_all[m], p1_all[m], y, h)})
    res = pd.DataFrame(rows)
    # paired comparison: same split and same masked records as "Probability"
    base = res[res.method == "Probability"].set_index(["missing", "repeat"])["utility_with_measuring"]
    res["gain_vs_probability"] = (res["utility_with_measuring"].values
                                  - base.loc[list(zip(res.missing, res.repeat))].values)
    groups = res.groupby(["missing", "method"], sort=False)
    table = groups.mean().drop(columns="repeat")
    table["gain_se"] = groups["gain_vs_probability"].std() / np.sqrt(N_REPEATS)
    return table.reset_index(), pd.DataFrame(timing).mean()


#  EXPERIMENT B: less training evidence for HighChol 

def experiment_scarcity(df, missing=0.30):
    rows = []
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(df))
        pool, test = df.iloc[perm[:5000]], df.iloc[perm[5000:7000]].reset_index(drop=True)
        Y_model = fit_Y_model(pool)                              # fixed Diabetes model
        m = mar_mask(mar_score(pool, test), missing, rng.random(len(test)))
        test_m = test[m]
        y, h = test_m[TARGET].values, test_m[AMPUTE_COL].values
        p0, p1 = p0_p1(Y_model, test_m)
        h_perm = rng.permutation(len(pool))
        for n_train in TRAIN_SIZES:
            sub = pool.iloc[h_perm[:n_train]]                    # nested subsets
            q, Q = q_of(fit_H_model(sub), test_m), bootstrap_Q(sub, test_m, N_BOOTSTRAP)
            for name in ["Probability", "BF cautious", "Credal (dominance)"]:
                repr_, rule = all_methods(q, Q, (q, q), h)[name]
                rows.append({"n_train": n_train, "method": name,
                             **evaluate(repr_, rule, p0, p1, y, h)})
    return pd.DataFrame(rows).groupby(["n_train", "method"], sort=False).mean().reset_index()


# Figures
COLORS = {"Probability": "#2a78d6", "BF pignistic": "#eb6834", "BF cautious": "#eb6834",
          "Credal (dominance)": "#1baf7a", "Credal (Gamma-maximin)": "#1baf7a",
          "Fuzzy": "#eda100", "Oracle (HighChol known)": "#898781"}
STYLES = {"BF pignistic": "--", "Credal (Gamma-maximin)": "--"}


def plot(table, x, panels, xlabel, filename, logx=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.4))
    for ax, (col, title) in zip(axes, panels):
        for name, g in table.groupby("method", sort=False):
            ax.plot(g[x], g[col], marker="o", ms=4, lw=1.8, label=name,
                    color=COLORS[name], ls=STYLES.get(name, "-"))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(TRAIN_SIZES, [str(s) for s in TRAIN_SIZES])
            ax.minorticks_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(filename, dpi=160)
    plt.close(fig)

# Main
if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 20,
                  "display.float_format", "{:.3f}".format)
    df = load_data()

    main, timing = experiment_missing_rate(df)
    main.to_csv("results/table_main.csv", index=False)
    timing.to_frame("mean per repeat").to_csv("results/table_timing.csv")
    print("EXPERIMENT A (masked records only, mean over 10 repeats)\n", main.to_string(index=False))
    print("\nCONSTRUCTION TIME PER REPEAT\n", timing.to_string())

    scarcity = experiment_scarcity(df)
    scarcity.to_csv("results/table_scarcity.csv", index=False)
    print("\nEXPERIMENT B (30% missing)\n", scarcity.to_string(index=False))

    main["missing_pct"] = 100 * main["missing"]
    plot(main, "missing_pct",
         [("test_determinacy", "Share of test/no-test decisions made"),
          ("gain_vs_probability", f"Utility gain over Probability (abstain = measure, cost {MEASURE_COST})")],
         "HighChol missing in the test set (%)", "results/fig_main.png")
    plot(scarcity, "n_train",
         [("width_H", "Width of the HighChol interval"),
          ("test_determinacy", "Share of test/no-test decisions made")],
         "records used to fit the HighChol model (log scale)", "results/fig_scarcity.png", logx=True)
    print("\nSaved results/table_*.csv and results/fig_*.png")
