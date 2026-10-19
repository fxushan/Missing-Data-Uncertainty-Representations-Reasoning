import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import experiment as E

LEVEL = 0.30                          # missing rate used for D2-D7
REFITS = [10, 25, 50, 100, 200]       # D5
COSTS = np.linspace(0, 0.30, 31)      # D4: cost of measuring HighChol


# helpers
def fuzzy_scores(train, test, rng_fcm):
    """Same steps as experiment.fuzzy_rep, but returns H-hat before rounding."""
    cols = E.FEATURES + [E.AMPUTE_COL]
    Z = train[cols].values.astype(float)
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    V = E.fcm_fit((Z - mu) / sd, rng_fcm)
    Zt = (test[E.FEATURES].values - mu[:-1]) / sd[:-1]
    D = len(cols) / len(E.FEATURES) * ((Zt[:, None, :] - V[None, :, :-1]) ** 2).sum(-1) + 1e-12
    W = E.memberships(D) ** E.FCM_FUZZIFIER
    h_proto = V[:, -1] * sd[-1] + mu[-1]
    return (W @ h_proto) / W.sum(axis=1)


def methods(q, Q, H_hat, h):
    out = E.all_methods(q, Q, ((H_hat >= 0.5).astype(float),) * 2, h)
    out["Credal (90% hull)"] = ((np.percentile(Q, 5, axis=1), np.percentile(Q, 95, axis=1)), "interval")
    out["Fuzzy (soft)"] = ((H_hat, H_hat), "interval")
    out["Credal (vacuous)"] = ((np.zeros_like(q), np.ones_like(q)), "interval")
    return out


def point_estimate(name, rep):
    """P(H = 1) that a method commits to, for the Brier score (D2)."""
    lo, hi = rep
    return (lo + hi) / 2


#  identical random-number order to experiment.experiment_missing_rate

def run(df):
    rows, refit_rows, t_refit, t_fcm, t_models, t_decide, example = [], [], [], [], [], [], None
    for rep in range(E.N_REPEATS):
        train, test = train_test_split(df, test_size=E.TEST_SIZE,
                                       random_state=int(E.rng.integers(1e9)), stratify=df[E.TARGET])
        test = test.reset_index(drop=True)
        t0 = time.perf_counter()
        H_model, Y_model = E.fit_H_model(train), E.fit_Y_model(train)
        t_models.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        Q_all = E.bootstrap_Q(train, test, E.N_BOOTSTRAP)
        t_refit.append((time.perf_counter() - t0) / Q_all.shape[1])
        r = E.rng.random(len(test))
        score = E.mar_score(train, test)
        t0 = time.perf_counter()
        H_hat_all = fuzzy_scores(train, test, np.random.default_rng(E.SEED + rep))
        t_fcm.append(time.perf_counter() - t0)
        q_all = E.q_of(H_model, test)
        p0_all, p1_all = E.p0_p1(Y_model, test)

        for level in E.MISSING_LEVELS:
            m = E.mar_mask(score, level, r)
            y, h = test[E.TARGET].values[m], test[E.AMPUTE_COL].values[m]
            q, Q, H_hat, p0, p1 = q_all[m], Q_all[m], H_hat_all[m], p0_all[m], p1_all[m]
            p_true = np.where(h == 1, p1, p0)
            d_oracle = (p_true > E.UTILITY_THRESHOLD).astype(int)
            d_prob = E.decide_test(*E.to_diabetes(q, q, p0, p1), "interval")
            d_vac = E.decide_test(*E.to_diabetes(np.zeros_like(q), np.ones_like(q), p0, p1), "interval")
            t0 = time.perf_counter()
            for name, (rp, rule) in methods(q, Q, H_hat, h).items():
                res = E.evaluate(rp, rule, p0, p1, y, h)
                d = E.decide_test(*E.to_diabetes(*rp, p0, p1), rule)
                decided = d != E.ABSTAIN
                res.update({
                    "missing": level, "method": name, "repeat": rep,
                    "brier_H": np.mean((point_estimate(name, rp) - h) ** 2),
                    "agree_oracle": np.mean(d[decided] == d_oracle[decided]),
                    "differs_from_prob": np.mean(d[decided] != d_prob[decided]),
                    # D3: of the records this method abstains on, how many did probability get wrong?
                    "prob_wrong_where_abstained": (np.mean(d_prob[~decided] != d_oracle[~decided])
                                                   if (~decided).any() else np.nan),
                    "prob_wrong_overall": np.mean(d_prob != d_oracle),
                    # where a method overrules probability: is it right, and is it inside the region
                    # where HighChol can change the decision at all (vacuous credal set abstains)?
                    "right_where_differs": (np.mean(d[changed] == d_oracle[changed])
                                            if (changed := decided & (d != d_prob)).any() else np.nan),
                    "differs_inside_ambiguity": (np.mean(d_vac[changed] == E.ABSTAIN) if changed.any() else np.nan),
                    # D4: utility if measuring were free (then subtract cost x abstention rate)
                    "utility_free_measuring": res["utility_with_measuring"] + E.MEASURE_COST * (1 - res["test_determinacy"]),
                })
                rows.append(res)
            t_decide.append(time.perf_counter() - t0)

            if abs(level - LEVEL) < 1e-9:
                # D5: fewer refits = first B columns of the same refits
                for B in REFITS:
                    for name, rp in [("BF cautious", E.belief_rep(q, Q[:, :B])), ("Credal (dominance)", E.credal_rep(Q[:, :B]))]:
                        res = E.evaluate(rp, "interval", p0, p1, y, h)
                        refit_rows.append({"B": B, "method": name, "repeat": rep, "width_H": res["width_H"],
                                           "test_determinacy": res["test_determinacy"],
                                           "utility_with_measuring": res["utility_with_measuring"],
                                           "utility_prob": E.evaluate((q, q), "interval", p0, p1, y, h)["utility_with_measuring"]})
                # D7: worked example from the first split where the BF rule abstains
                if example is None:
                    example = pick_example(test[m].reset_index(drop=True), q, Q, H_hat, p0, p1, h, y, d_prob, d_oracle, rep + 1)
    timing = {"models": np.mean(t_models), "refit": np.mean(t_refit), "fcm": np.mean(t_fcm),
              "decide_all_methods": np.mean(t_decide)}
    return pd.DataFrame(rows), pd.DataFrame(refit_rows), timing, example


def pick_example(test_m, q, Q, H_hat, p0, p1, h, y, d_prob, d_oracle, rep_now=0):
    """A record where HighChol matters: the interval methods abstain on the test
    decision, probability decides wrongly, and fuzzy rounding disagrees with probability."""
    bel, pl = E.belief_rep(q, Q)
    d_bf = E.decide_test(*E.to_diabetes(bel, pl, p0, p1), "interval")
    d_fz = E.decide_test(*E.to_diabetes(*((H_hat >= 0.5).astype(float),) * 2, p0, p1), "interval")
    for cond in [(d_bf == E.ABSTAIN) & (d_prob != d_oracle) & (y == 1),
                 (d_bf == E.ABSTAIN) & (d_prob != d_oracle) & (d_fz != d_prob),
                 (d_bf == E.ABSTAIN) & (d_prob != d_oracle),
                 (d_bf == E.ABSTAIN) & (d_fz != d_prob),
                 d_bf == E.ABSTAIN]:
        cand = np.where(cond)[0]
        if len(cand):
            break
    if len(cand) == 0:
        return None
    i = cand[np.argmin(np.abs(q[cand] - 0.5))]          # the most balanced one
    same = np.all(test_m[E.FEATURES].values == test_m[E.FEATURES].values[i], axis=1)
    u = np.percentile(Q[i], 95) - np.percentile(Q[i], 5)
    reps = {"Probability": (q[i], q[i]), "BF pignistic": E.pignistic_rep(q[i:i + 1], Q[i:i + 1]),
            "BF cautious": (bel[i], pl[i]), "Credal (dominance)": (Q[i].min(), Q[i].max()),
            "Credal (Gamma-maximin)": (Q[i].min(), Q[i].max()), "Fuzzy": (float(H_hat[i] >= 0.5),) * 2}
    rows = []
    for name, (lo, hi) in reps.items():
        lo, hi = float(np.ravel(lo)[0]), float(np.ravel(hi)[0])
        rule = "gamma" if "Gamma" in name else "interval"
        plo, phi = E.to_diabetes(np.array([lo]), np.array([hi]), np.array([p0[i]]), np.array([p1[i]]))
        d = E.decide_test(plo, phi, rule)[0]
        d_final = d if d != E.ABSTAIN else int(d_oracle[i])          # after measuring HighChol
        rows.append({"method": name, "H_lo": lo, "H_hi": hi, "Y_lo": plo[0], "Y_hi": phi[0],
                     "test_decision": {1: "test", 0: "no test", E.ABSTAIN: "measure HighChol"}[int(d)],
                     "utility": E.U[d_final, y[i]] - (E.MEASURE_COST if d == E.ABSTAIN else 0)})
    ex = pd.DataFrame(rows)
    ex.attrs = {"repeat": rep_now, **{c: int(test_m.loc[i, c]) for c in E.FEATURES}, "HighChol_true": int(h[i]), "Diabetes": int(y[i]),
                "q": q[i], "Q_min": Q[i].min(), "Q05": np.percentile(Q[i], 5), "Q95": np.percentile(Q[i], 95),
                "Q_max": Q[i].max(), "u": u, "m0": (1 - u) * (1 - q[i]), "m1": (1 - u) * q[i], "H_hat": H_hat[i],
                "p0": p0[i], "p1": p1[i], "oracle_decision": "test" if d_oracle[i] else "no test",
                "records_with_same_X": int(same.sum()), "masked_records": len(q)}
    return ex


# summaries
def summarise(raw):
    base = raw[raw.method == "Probability"].set_index(["missing", "repeat"])
    idx = list(zip(raw.missing, raw.repeat))
    raw["gain_vs_probability"] = raw["utility_with_measuring"].values - base.loc[idx, "utility_with_measuring"].values
    raw["gain_free"] = raw["utility_free_measuring"].values - base.loc[idx, "utility_free_measuring"].values
    g = raw.groupby(["missing", "method"], sort=False)
    table = g.mean(numeric_only=True).drop(columns="repeat")
    table["gain_se"] = g["gain_vs_probability"].std() / np.sqrt(E.N_REPEATS)
    return table.reset_index()


def breakeven(table):
    t = table[np.isclose(table.missing, LEVEL)].set_index("method")
    rows = []
    for name in ["BF cautious", "Credal (dominance)", "Credal (90% hull)", "Credal (vacuous)", "Oracle (HighChol known)"]:
        a = 1.0 if name.startswith("Oracle") else 1 - t.loc[name, "test_determinacy"]   # share measured
        g0 = t.loc[name, "gain_free"]
        rows.append({"method": name, "share_measured": a, "gain_if_free": g0,
                     "gain_at_0.05": g0 - 0.05 * a, "breakeven_cost": g0 / a if a > 0 else np.nan})
    return pd.DataFrame(rows)


def plot(table, be, refits, timing, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.axhline(0, color="#2a78d6", lw=1.4, label="probability (never measures)")
    for name, color, ls, label in [("BF cautious", "#eb6834", "-", "BF cautious"),
                                   ("Credal (dominance)", "#1baf7a", "-", "credal set"),
                                   ("Oracle (HighChol known)", "#898781", "--", "measure everyone")]:
        r = be.set_index("method").loc[name]
        share = 100 * r.share_measured
        ax.plot(COSTS, r.gain_if_free - COSTS * r.share_measured, color=color, ls=ls, lw=1.8,
                label=f"{label} (measures {share:.0f}%)" if share < 100 else label)
        ax.plot([r.breakeven_cost], [0], "o", color=color, ms=5)
        ax.annotate(f"{r.breakeven_cost:.2f}", (r.breakeven_cost, 0), textcoords="offset points",
                    xytext=(0, -13 if name != "Credal (dominance)" else 6), ha="center", fontsize=8, color=color)
    ax.set_xlabel("cost of measuring HighChol (utility units)")
    ax.set_title("Utility gain over probability (30% missing)", loc="left", fontsize=10)
    ax.set_ylim(-0.012, 0.022)
    ax.set_xlim(0, 0.30)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax = axes[1]
    for name, color, label in [("BF cautious", "#eb6834", "BF: 5-95% range of the refits"),
                               ("Credal (dominance)", "#1baf7a", "credal set: min-max of the refits")]:
        g = refits[refits.method == name]
        ax.plot(g.B, g.width_H, marker="o", ms=4, lw=1.8, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks(REFITS, [f"{b}\n{b * timing['refit']:.2f} s" for b in REFITS])
    ax.minorticks_off()
    ax.set_xlabel("number of bootstrap refits B (and time per split)")
    ax.set_title("Width of the HighChol interval vs number of refits", loc="left", fontsize=10)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
    df = E.load_data()
    raw, refit_raw, timing, example = run(df)
    table = summarise(raw)
    table.to_csv("results/table_compare.csv", index=False)

    # check: the shared rows must reproduce experiment A
    main = pd.read_csv("results/table_main.csv")
    chk = table.merge(main, on=["missing", "method"], suffixes=("", "_A"))
    print(f"check: max difference with table_main.csv = {np.abs(chk.utility_with_measuring - chk.utility_with_measuring_A).max():.1e}"
          f" over {len(chk)} rows")

    t30 = table[np.isclose(table.missing, LEVEL)]
    cols = ["method", "accuracy", "determinacy", "test_determinacy", "width_H", "gain_vs_probability", "gain_se",
            "brier_H", "agree_oracle", "differs_from_prob", "right_where_differs", "differs_inside_ambiguity",
            "prob_wrong_where_abstained", "prob_wrong_overall"]
    print(f"\nD1-D3 at {int(LEVEL * 100)}% missing\n", t30[cols].to_string(index=False))

    be = breakeven(table)
    be.to_csv("results/table_breakeven.csv", index=False)
    print("\nD4 break-even cost of measuring HighChol\n", be.to_string(index=False))

    refit_raw["gain"] = refit_raw.utility_with_measuring - refit_raw.utility_prob
    refits = refit_raw.groupby(["B", "method"], sort=False).mean(numeric_only=True).drop(columns="repeat").reset_index()
    refits["time_s"] = refits.B * timing["refit"]
    refits.to_csv("results/table_refits.csv", index=False)
    print("\nD5 number of bootstrap refits\n", refits.to_string(index=False))

    cost = pd.DataFrame([
        {"theory": "Probability", "construction": "2 logistic fits", "seconds": timing["models"], "numbers_per_record": "1 (q)"},
        {"theory": "Belief functions", "construction": f"+ {E.N_BOOTSTRAP} bootstrap refits", "seconds": timing["refit"] * E.N_BOOTSTRAP,
         "numbers_per_record": "2 free masses"},
        {"theory": "Credal set", "construction": "same refits as BF (no extra)", "seconds": 0.0,
         "numbers_per_record": "2 bounds (200 vertices)"},
        {"theory": "Fuzzy clustering", "construction": f"+ FCM, K={E.FCM_CLUSTERS}", "seconds": timing["fcm"],
         "numbers_per_record": f"{E.FCM_CLUSTERS} memberships -> 1 value"},
        {"theory": "all", "construction": "intervals + decisions, all 10 methods, all masked records", "seconds": timing["decide_all_methods"],
         "numbers_per_record": ""}])
    cost.to_csv("results/table_cost.csv", index=False)
    print("\nD6 cost per split\n", cost.to_string(index=False))

    example.to_csv("results/worked_example.csv", index=False)
    print("\nD7 worked example (30% missing)\n", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in example.attrs.items()})
    print(example.to_string(index=False))

    plot(table, be, refits, timing, "results/fig_tradeoff.png")
    print("\nSaved results/table_compare.csv, table_breakeven.csv, table_refits.csv, table_cost.csv, "
          "worked_example.csv, fig_tradeoff.png")
