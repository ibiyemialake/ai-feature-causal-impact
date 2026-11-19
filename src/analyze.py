"""Estimate the effect of the AI assistant on monthly retention four ways and
compare each estimate with the known true effect.

1. Naive comparison (biased on purpose)
2. Difference-in-differences with region and month fixed effects
3. Propensity score matching on pre-launch account characteristics
4. Synthetic control for the treated regions as one combined unit
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from simulate import LAUNCH_MONTH, TRUE_EFFECT, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 7


def naive(panel: pd.DataFrame) -> dict:
    post = panel[panel.post == 1]
    t = post[post.treated == 1].active.mean()
    c = post[post.treated == 0].active.mean()
    return {"method": "Naive (post-launch, treated vs not)", "estimate": t - c, "ci_low": np.nan, "ci_high": np.nan}


def did(panel: pd.DataFrame, accounts: pd.DataFrame) -> dict:
    rm = panel.groupby(["region", "month"], as_index=False).agg(active=("active", "mean"), n=("active", "size"))
    rm = rm.merge(accounts.groupby("region").treated_region.first().reset_index(), on="region")
    rm["post"] = (rm.month >= LAUNCH_MONTH).astype(int)
    rm["d"] = rm.treated_region * rm.post
    model = smf.wls("active ~ d + C(region) + C(month)", data=rm, weights=rm.n).fit(
        cov_type="cluster", cov_kwds={"groups": rm.region}
    )
    est, (lo, hi) = model.params["d"], model.conf_int().loc["d"]

    # Parallel trends plot
    trend = rm.groupby(["treated_region", "month"]).apply(
        lambda g: np.average(g.active, weights=g.n), include_groups=False
    ).unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trend.index, trend[1], marker="o", label="Regions with AI assistant")
    ax.plot(trend.index, trend[0], marker="o", label="Regions without")
    ax.axvline(LAUNCH_MONTH - 0.5, color="grey", ls="--", lw=1)
    ax.text(LAUNCH_MONTH - 0.4, ax.get_ylim()[1] * 0.995, "launch", va="top", fontsize=9, color="grey")
    ax.set(xlabel="Month", ylabel="Monthly retention", title="Difference-in-differences: trends before and after launch")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "did_parallel_trends.png", dpi=150)
    plt.close(fig)

    # Pre-trend test: does the treated-control gap change over the pre-launch months?
    pre = rm[rm.post == 0].copy()
    pre["trend_x_treated"] = pre.month * pre.treated_region
    pt = smf.wls("active ~ trend_x_treated + C(region) + C(month)", data=pre, weights=pre.n).fit(
        cov_type="cluster", cov_kwds={"groups": pre.region}
    )
    return {"method": "Difference-in-differences", "estimate": est, "ci_low": lo, "ci_high": hi,
            "note": f"pre-trend slope p = {pt.pvalues['trend_x_treated']:.2f}"}


def smd(x_t: pd.Series, x_c: pd.Series) -> float:
    return (x_t.mean() - x_c.mean()) / np.sqrt((x_t.var() + x_c.var()) / 2)


def psm(panel: pd.DataFrame, accounts: pd.DataFrame, n_boot: int = 200) -> dict:
    pre_rate = panel[panel.post == 0].groupby("account_id").active.mean().rename("pre_active_rate")
    post_rate = panel[panel.post == 1].groupby("account_id").active.mean().rename("post_active_rate")
    df = accounts.join(pre_rate, on="account_id").join(post_rate, on="account_id")
    df = pd.get_dummies(df, columns=["plan"], drop_first=True, dtype=int)
    covs = ["tenure_months", "seats", "engagement", "pre_active_rate", "plan_premium", "plan_ultimate"]

    X = StandardScaler().fit_transform(df[covs])
    ps = LogisticRegression(max_iter=1000).fit(X, df.treated_region).predict_proba(X)[:, 1]
    df["logit_ps"] = np.log(ps / (1 - ps))

    t, c = df[df.treated_region == 1], df[df.treated_region == 0]
    caliper = 0.2 * df.logit_ps.std()
    nn = NearestNeighbors(n_neighbors=1).fit(c[["logit_ps"]])
    dist, idx = nn.kneighbors(t[["logit_ps"]])
    keep = dist[:, 0] <= caliper
    t_m, c_m = t[keep], c.iloc[idx[keep, 0]]

    est = t_m.post_active_rate.mean() - c_m.post_active_rate.mean()

    rng = np.random.default_rng(SEED)
    diffs = t_m.post_active_rate.to_numpy() - c_m.post_active_rate.to_numpy()
    boots = [rng.choice(diffs, len(diffs)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # Covariate balance before and after matching
    bal = pd.DataFrame({
        "covariate": covs,
        "smd_before": [smd(t[v], c[v]) for v in covs],
        "smd_after": [smd(t_m[v], c_m[v]) for v in covs],
    })
    bal.to_csv(OUT / "psm_balance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(covs))
    ax.scatter(bal.smd_before.abs(), y, label="Before matching")
    ax.scatter(bal.smd_after.abs(), y, label="After matching")
    ax.axvline(0.1, color="grey", ls="--", lw=1)
    ax.set_yticks(y, covs)
    ax.set(xlabel="Absolute standardized mean difference", title="Propensity score matching: covariate balance")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "psm_balance.png", dpi=150)
    plt.close(fig)

    return {"method": "Propensity score matching", "estimate": est, "ci_low": lo, "ci_high": hi,
            "note": f"{keep.sum():,} of {len(t):,} treated accounts matched, max |SMD| after = {bal.smd_after.abs().max():.2f}"}


def synth_weights(y_treated_pre: np.ndarray, Y_donors_pre: np.ndarray) -> np.ndarray:
    k = Y_donors_pre.shape[1]
    loss = lambda w: np.sum((y_treated_pre - Y_donors_pre @ w) ** 2)
    res = minimize(loss, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x


def synthetic_control(panel: pd.DataFrame, accounts: pd.DataFrame) -> dict:
    rm = panel.groupby(["region", "month"]).active.mean().unstack(0)  # months x regions
    treated = accounts.groupby("region").treated_region.first()
    t_regions, donors = treated[treated == 1].index, treated[treated == 0].index

    sizes = accounts.region.value_counts()
    y = (rm[t_regions] * sizes[t_regions]).sum(axis=1) / sizes[t_regions].sum()
    pre = rm.index < LAUNCH_MONTH

    def fit_gap(y_series, donor_cols):
        w = synth_weights(y_series[pre].to_numpy(), rm.loc[pre, donor_cols].to_numpy())
        synth = rm[donor_cols].to_numpy() @ w
        return y_series.to_numpy() - synth, w, synth

    gap, w, synth = fit_gap(y, donors)
    est = gap[~pre].mean()
    pre_rmse = np.sqrt(np.mean(gap[pre] ** 2))

    # Placebo-in-space: pretend each donor region was treated, compare post/pre RMSE ratios
    def ratio(g):
        return np.sqrt(np.mean(g[~pre] ** 2)) / np.sqrt(np.mean(g[pre] ** 2))
    ratios = [ratio(fit_gap(rm[d], donors.drop(d))[0]) for d in donors]
    p_value = (1 + sum(r >= ratio(gap) for r in ratios)) / (1 + len(ratios))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rm.index, y, marker="o", label="Regions with AI assistant (actual)")
    ax.plot(rm.index, synth, marker="o", ls="--", label="Synthetic control")
    ax.axvline(LAUNCH_MONTH - 0.5, color="grey", ls="--", lw=1)
    ax.set(xlabel="Month", ylabel="Monthly retention", title="Synthetic control: actual vs counterfactual")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_control.png", dpi=150)
    plt.close(fig)

    top = pd.Series(w, index=donors).sort_values(ascending=False).head(5)
    pd.Series(w, index=donors, name="weight").to_csv(OUT / "synthetic_control_weights.csv")
    return {"method": "Synthetic control", "estimate": est, "ci_low": np.nan, "ci_high": np.nan,
            "note": f"pre-launch RMSE = {pre_rmse:.4f}, placebo p = {p_value:.2f}, {int((w > 0.01).sum())} donor regions used"}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    accounts, panel = simulate()
    results = pd.DataFrame([
        naive(panel),
        did(panel, accounts),
        psm(panel, accounts),
        synthetic_control(panel, accounts),
    ])
    results["true_effect"] = TRUE_EFFECT
    results["error_pp"] = (results.estimate - TRUE_EFFECT) * 100
    for col in ["estimate", "ci_low", "ci_high"]:
        results[col + "_pp"] = results[col] * 100
    results = results[["method", "estimate_pp", "ci_low_pp", "ci_high_pp", "error_pp", "note"]].round(2)
    results.to_csv(OUT / "results.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"True effect: +{TRUE_EFFECT * 100:.1f} percentage points\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
