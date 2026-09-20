"""
========================================================================
 Missing data: Probability theory  vs  Belief functions (Dempster-Shafer)
 on 5 variables from the UCI CDC Diabetes Health Indicators data set.
========================================================================

WHAT THIS SCRIPT DOES
---------------------
1. Loads the CDC Diabetes Health Indicators data set (5 chosen variables +
   binary target). Tries, in order:
      (a) the `ucimlrepo` package  (pip install ucimlrepo),
      (b) a local CSV given in CONFIG,
      (c) a faithful SYNTHETIC stand-in, so the script always runs.
2. Amputes one variable (HighChol) under a MAR mechanism (Schouten-style
   weighted-sum-score logistic), sweeping the missing fraction 10% -> 50%.
3. Builds the SAME per-feature evidence  p_j(class | X_j = v)  from training,
   then predicts the class two ways that differ ONLY in representation:
      * PROBABILITY  -> naive-Bayes combination (product of p_j, skip the
        missing feature = the standard NB way of marginalising it out).
      * BELIEF       -> each feature's evidence is DISCOUNTED into a mass
        function m_j (mass on singletons + mass on the whole frame Omega);
        a MISSING feature contributes a VACUOUS mass (all mass on Omega);
        the m_j are fused with DEMPSTER'S RULE, tracking conflict K.
4. Reports, per method and per missing level:
      accuracy, wall-clock time (a proxy for time complexity),
      and for the belief representation the four uncertainty components:
        - randomness      = Shannon entropy of the pignistic BetP
        - imprecision     = non-specificity  N(m) = sum_A m(A) log2|A|
        - ignorance       = mass on the frame m(Omega)   (== Pl - Bel here)
        - conflict        = Dempster conflict K aggregated over the fusion
   (Probability has, by construction, imprecision = ignorance = 0 and no
    tracked conflict -- that ZERO is itself the key representational result.)

NOTE ON THE MEASURES (read this before presenting numbers):
   For a BINARY class the only non-singleton focal set is the frame, so
   non-specificity N(m) == m(Omega) == Pl-Bel.  Hence "imprecision" and
   "ignorance" are the SAME number here; report them as one and say so.
   Randomness (H of BetP) and conflict (K) are genuinely distinct.
   Set TARGET = "Diabetes_012" (3 classes) if your CSV has it, to make the
   components a little richer -- but they stay proportional unless you also
   enrich the mass model (see comments in feature_mass()).

Author: Group [NN] -- 2AMU30
"""

import time
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# CONFIG -- edit here
# ----------------------------------------------------------------------
FEATURES     = ["HighBP", "HighChol", "Smoker", "PhysActivity", "GenHlth"]  # 5 variables
AMPUTE_COL   = "HighChol"          # the variable we make missing (must be in FEATURES)
TARGET       = "Diabetes_binary"   # or "Diabetes_012" for the 3-class version
LOCAL_CSV    = "diabetes_binary_health_indicators_BRFSS2015.csv"  # used if ucimlrepo absent
MISSING_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
DISCOUNT     = 0.55                 # alpha: belief keeps (1-alpha) mass as "don't know".
                                   # With ~5 features, Dempster fusion sharpens fast, so keep
                                   # alpha modest (0.5-0.7) or the frame mass vanishes and the
                                   # belief model collapses onto probability. Try a few values.
N_REPEATS    = 10                  # repeats per level (different random masks)
TEST_SIZE    = 0.30
SEED         = 2025
MAX_ROWS     = 20000               # subsample big data sets for speed (None = all)

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# 1. DATA LOADING (real UCI -> local CSV -> synthetic fallback)
# ----------------------------------------------------------------------
def load_data():
    # (a) try ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=891)                       # CDC Diabetes Health Indicators
        df = ds.data.features.copy()
        df[TARGET] = ds.data.targets.iloc[:, 0].values
        print("Loaded data from ucimlrepo (id=891).")
        return df
    except Exception as e:
        print(f"ucimlrepo not used ({e.__class__.__name__}). Trying local CSV...")
    # (b) try local CSV
    try:
        df = pd.read_csv(LOCAL_CSV)
        print(f"Loaded data from local CSV: {LOCAL_CSV}")
        return df
    except Exception as e:
        print(f"Local CSV not found ({e.__class__.__name__}). Using SYNTHETIC data.")
    # (c) synthetic fallback with the same structure
    return make_synthetic()

def make_synthetic(n=12000):
    age   = rng.integers(1, 14, n)
    highbp = (rng.random(n) < 0.05 + 0.03 * age).astype(int)
    p_chol = 1 / (1 + np.exp(-(-2.2 + 0.18 * age + 0.9 * highbp)))
    highchol = (rng.random(n) < p_chol).astype(int)
    smoker = (rng.random(n) < 0.45).astype(int)
    physact = (rng.random(n) < 0.75).astype(int)
    genhlth = rng.integers(1, 6, n)                      # 1..5
    lin = (-3.0 + 0.9 * highbp + 1.1 * highchol + 0.25 * smoker
           - 0.4 * physact + 0.5 * genhlth + 0.05 * age)
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    return pd.DataFrame(dict(HighBP=highbp, HighChol=highchol, Smoker=smoker,
                             PhysActivity=physact, GenHlth=genhlth,
                             Diabetes_binary=y))

def prepare(df):
    keep = FEATURES + [TARGET]
    df = df[keep].dropna().copy()
    if MAX_ROWS and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
    # bin any feature with too many distinct values (e.g. continuous BMI)
    for c in FEATURES:
        if df[c].nunique() > 12:
            df[c] = pd.qcut(df[c], 5, labels=False, duplicates="drop")
    for c in keep:
        df[c] = df[c].astype(int)
    return df

# ----------------------------------------------------------------------
# 2. MAR amputation (weighted-sum-score logistic, after Schouten et al. 2018)
#    Missingness of AMPUTE_COL depends only on the OTHER (observed) features.
# ----------------------------------------------------------------------
def mar_mask(X, rate):
    others = [c for c in FEATURES if c != AMPUTE_COL]
    Z = X[others].apply(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    w = rng.normal(1.0, 0.3, len(others))                # random-ish weights
    score = Z.values @ w
    # calibrate logistic intercept so that mean(missing prob) ~= rate
    def mean_p(b): return (1 / (1 + np.exp(-(b + score)))).mean()
    lo, hi = -20, 20
    for _ in range(60):
        mid = (lo + hi) / 2
        if mean_p(mid) < rate: lo = mid
        else: hi = mid
    p = 1 / (1 + np.exp(-((lo + hi) / 2 + score)))
    return rng.random(len(X)) < p                        # True = missing

# ----------------------------------------------------------------------
# 3a. Per-feature evidence  p_j(class | X_j = value)   (Laplace-smoothed)
# ----------------------------------------------------------------------
def fit_feature_evidence(Xtr, ytr, classes):
    ev = {}
    for c in FEATURES:
        tab = {}
        for v in np.unique(Xtr[c]):
            sel = Xtr[c].values == v
            counts = np.array([(ytr[sel] == k).sum() for k in classes], float)
            counts += 1.0                                # Laplace
            tab[v] = counts / counts.sum()
        # fallback distribution for unseen values = class prior
        prior = np.array([(ytr == k).mean() for k in classes], float)
        ev[c] = (tab, prior)
    return ev

def feat_dist(ev, feature, value):
    tab, prior = ev[feature]
    return tab.get(value, prior)

# ----------------------------------------------------------------------
# 3b. PROBABILITY method : naive-Bayes product over OBSERVED features
# ----------------------------------------------------------------------
def predict_probability(row, missing, ev, classes, prior):
    post = prior.copy().astype(float)
    for c in FEATURES:
        if missing and c == AMPUTE_COL:
            continue                                     # skip = marginalise out
        post *= feat_dist(ev, c, row[c])
    post /= post.sum()
    return post                                          # a proper distribution

# ----------------------------------------------------------------------
# 3c. BELIEF method : discount each feature to a mass fn, fuse w/ Dempster
#     Mass fn stored as dict {frozenset(class subset): mass}.
# ----------------------------------------------------------------------
def feature_mass(dist, classes, alpha):
    """Discounted Bayesian mass: alpha on singletons, (1-alpha) on the frame.
       --- To make imprecision differ from ignorance, replace this with a
           model that puts mass on 2-element subsets (e.g. consonant/k-NN
           evidential). With singletons+frame only, N(m) is proportional to
           m(frame), so imprecision == ignorance (see header note)."""
    m = {frozenset([c]): alpha * p for c, p in zip(classes, dist)}
    m[frozenset(classes)] = 1.0 - alpha                  # the "don't know" part
    return m

def vacuous(classes):
    return {frozenset(classes): 1.0}                     # total ignorance

def dempster(m1, m2):
    """Return (combined mass, conflict K)."""
    comb, K = {}, 0.0
    for A, a in m1.items():
        for B, b in m2.items():
            C = A & B
            if not C:
                K += a * b                               # conflicting mass
            else:
                comb[C] = comb.get(C, 0.0) + a * b
    if K >= 1.0 - 1e-12:                                 # total conflict guard
        return {frozenset().union(*m1.keys()): 1.0}, 1.0
    for C in comb:                                       # normalise
        comb[C] /= (1.0 - K)
    return comb, K

def predict_belief(row, missing, ev, classes, alpha):
    frame = frozenset(classes)
    m = {frame: 1.0}                                     # start vacuous
    total_conf = 0.0
    for c in FEATURES:
        if missing and c == AMPUTE_COL:
            mj = vacuous(classes)                        # missing -> contribute ignorance only
        else:
            mj = feature_mass(feat_dist(ev, c, row[c]), classes, alpha)
        m, K = dempster(m, mj)
        total_conf = 1.0 - (1.0 - total_conf) * (1.0 - K)  # aggregate conflict in [0,1]
    return m, total_conf

# ----------------------------------------------------------------------
# 3d. Belief -> Bel, Pl, pignistic BetP, and the uncertainty measures
# ----------------------------------------------------------------------
def bel_pl_betp(m, classes):
    bel, pl, betp = {}, {}, {}
    for c in classes:
        s = frozenset([c])
        bel[c] = sum(v for A, v in m.items() if A <= s)
        pl[c]  = sum(v for A, v in m.items() if (A & s))
    for c in classes:                                   # pignistic transform
        betp[c] = sum(v / len(A) for A, v in m.items() if c in A)
    return bel, pl, betp

def measures(m, classes):
    # randomness : Shannon entropy of BetP
    _, _, betp = bel_pl_betp(m, classes)
    p = np.array([betp[c] for c in classes]); p = p[p > 0]
    randomness = float(-(p * np.log2(p)).sum())
    # imprecision : non-specificity N(m) = sum_A m(A) log2|A|
    nonspec = float(sum(v * np.log2(len(A)) for A, v in m.items() if len(A) > 1))
    # ignorance : mass on the full frame (== Pl-Bel for binary)
    ignorance = float(m.get(frozenset(classes), 0.0))
    return randomness, nonspec, ignorance

# ----------------------------------------------------------------------
# 4. EXPERIMENT LOOP
# ----------------------------------------------------------------------
def run():
    df = prepare(load_data())
    classes = sorted(df[TARGET].unique())
    print(f"Data: {df.shape[0]} rows, features={FEATURES}, "
          f"target={TARGET} ({len(classes)} classes), amputing '{AMPUTE_COL}'.\n")

    rows = []
    for lvl in MISSING_LEVELS:
        acc_p, acc_b_pig, acc_b_cau, det = [], [], [], []
        t_p, t_b = [], []
        rand_, imp_, ign_, conf_ = [], [], [], []
        for _ in range(N_REPEATS):
            Xtr, Xte, ytr, yte = train_test_split(
                df[FEATURES], df[TARGET].values, test_size=TEST_SIZE,
                random_state=int(rng.integers(1e9)), stratify=df[TARGET])
            prior = np.array([(ytr == k).mean() for k in classes], float)
            ev = fit_feature_evidence(Xtr, ytr, classes)
            miss = mar_mask(Xte, lvl)

            # ---- probability ----
            t0 = time.perf_counter()
            pp = []
            for i, (_, r) in enumerate(Xte.iterrows()):
                pp.append(predict_probability(r, miss[i], ev, classes, prior))
            t_p.append(time.perf_counter() - t0)
            pp = np.array(pp)
            pred_p = np.array([classes[j] for j in pp.argmax(1)])
            acc_p.append((pred_p == yte).mean())

            # ---- belief ----
            t0 = time.perf_counter()
            pig, cau_pred, cau_ok, rr, ii, gg, cc = [], [], [], [], [], [], []
            for i, (_, r) in enumerate(Xte.iterrows()):
                m, K = predict_belief(r, miss[i], ev, classes, DISCOUNT)
                bel, pl, betp = bel_pl_betp(m, classes)
                # pignistic decision
                pig.append(classes[int(np.argmax([betp[c] for c in classes]))])
                # cautious / interval-dominance decision (may abstain)
                nondom = [c for c in classes
                          if not any(bel[d] > pl[c] for d in classes if d != c)]
                cau_pred.append(nondom)
                rmes, imes, gmes = measures(m, classes)
                rr.append(rmes); ii.append(imes); gg.append(gmes); cc.append(K)
            t_b.append(time.perf_counter() - t0)
            pig = np.array(pig)
            acc_b_pig.append((pig == yte).mean())
            # cautious: determinate rows only, plus determinacy rate & set-accuracy
            determinate = np.array([len(s) == 1 for s in cau_pred])
            det.append(determinate.mean())
            if determinate.sum():
                dp = np.array([cau_pred[i][0] for i in range(len(cau_pred)) if determinate[i]])
                acc_b_cau.append((dp == yte[determinate]).mean())
            else:
                acc_b_cau.append(np.nan)
            rand_.append(np.mean(rr)); imp_.append(np.mean(ii))
            ign_.append(np.mean(gg));  conf_.append(np.mean(cc))

        rows.append(dict(
            missing=lvl,
            acc_prob=np.mean(acc_p),
            acc_belief_pignistic=np.mean(acc_b_pig),
            acc_belief_cautious=np.nanmean(acc_b_cau),
            determinacy=np.mean(det),
            time_prob_s=np.mean(t_p),
            time_belief_s=np.mean(t_b),
            randomness=np.mean(rand_),
            imprecision=np.mean(imp_),      # == ignorance for binary target
            ignorance=np.mean(ign_),
            conflict=np.mean(conf_),
        ))

    res = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("RESULTS\n-------")
    print(res.to_string(index=False))
    res.to_csv("results_prob_vs_belief.csv", index=False)
    print("\nSaved results_prob_vs_belief.csv")

    # ---- qualitative rows the numbers don't capture (for your comparison table) ----
    print("\nQUALITATIVE (fill into your comparison table):")
    print("  Representation     : probability = distribution p on Omega (additive)")
    print("                       belief      = mass m on 2^Omega (mass may sit on the frame)")
    print("  Expressiveness     : belief > probability (belief contains probability as the")
    print("                       special case where every focal set is a singleton)")
    print("  Randomness         : both (entropy of the class distribution / BetP)")
    print("  Imprecision        : probability = 0 by construction; belief > 0")
    print("  Ignorance          : probability = 0 by construction; belief > 0 (mass on frame)")
    print("  Conflict           : probability = folded into normalisation; belief = Dempster K")

    make_figures(res, classes)
    return res

# ----------------------------------------------------------------------
# 5. FIGURES
# ----------------------------------------------------------------------
def make_figures(res, classes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figures.")
        return
    x = res.missing * 100
    # accuracy
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, res.acc_prob, "s-", label="Probability")
    ax.plot(x, res.acc_belief_pignistic, "^-", label="Belief (pignistic)")
    ax.plot(x, res.acc_belief_cautious, "D-", label="Belief (cautious, determinate)")
    ax.set_xlabel("% of HighChol missing (MAR)"); ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs. missingness"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("fig_accuracy.png", dpi=150)

    # four uncertainty components (belief representation)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, res.randomness, "o-", label="randomness  H(BetP)")
    ax.plot(x, res.imprecision, "s-", label="imprecision  N(m)")
    ax.plot(x, res.ignorance, "^--", label="ignorance  m(Ω)")
    ax.plot(x, res.conflict, "D-", label="conflict  K")
    ax.set_xlabel("% of HighChol missing (MAR)"); ax.set_ylabel("measure (bits / mass)")
    ax.set_title("Uncertainty components of the belief representation")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("fig_uncertainty_components.png", dpi=150)
    print("Saved fig_accuracy.png, fig_uncertainty_components.png")

if __name__ == "__main__":
    run()
