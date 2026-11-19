"""Generate a synthetic SaaS dataset where an AI assistant feature is rolled out
to some regions, and the rollout is NOT random.

The true effect is known (TRUE_EFFECT), so every method can be scored against it.

Setup
- 40 regions, 5,000 accounts, 12 months of activity.
- In month 7 the AI assistant launches in 10 regions.
- Regions with more engaged accounts are more likely to get the feature first.
  That is the selection problem: a naive comparison will overstate the effect.
- Outcome: whether the account was active (retained) in a given month.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_REGIONS = 40
N_ACCOUNTS = 5_000
N_MONTHS = 12
LAUNCH_MONTH = 7
N_TREATED_REGIONS = 10
TRUE_EFFECT = 0.04  # +4 percentage points monthly retention for treated accounts

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def simulate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # Regions differ in baseline engagement. Higher-engagement regions get the feature first.
    region_eng = rng.normal(0, 1, N_REGIONS)
    score = region_eng + rng.normal(0, 0.6, N_REGIONS)
    treated_regions = set(np.argsort(-score)[:N_TREATED_REGIONS])

    # Accounts
    region = rng.integers(0, N_REGIONS, N_ACCOUNTS)
    tenure_months = rng.gamma(shape=2.0, scale=10.0, size=N_ACCOUNTS).round(1)
    seats = rng.poisson(lam=8, size=N_ACCOUNTS) + 1
    plan = rng.choice(["free", "premium", "ultimate"], size=N_ACCOUNTS, p=[0.5, 0.35, 0.15])
    plan_lift = pd.Series(plan).map({"free": 0.0, "premium": 0.06, "ultimate": 0.10}).to_numpy()

    # Account engagement depends partly on the region (drives selection bias)
    engagement = 0.6 * region_eng[region] + rng.normal(0, 1, N_ACCOUNTS)

    accounts = pd.DataFrame({
        "account_id": np.arange(N_ACCOUNTS),
        "region": region,
        "tenure_months": tenure_months,
        "seats": seats,
        "plan": plan,
        "engagement": engagement.round(3),
        "treated_region": np.isin(region, list(treated_regions)).astype(int),
    })

    # Monthly panel
    season = 0.03 * np.sin(np.arange(1, N_MONTHS + 1) / 12 * 2 * np.pi)
    region_noise = rng.normal(0, 0.01, (N_REGIONS, N_MONTHS))
    rows = []
    for m in range(1, N_MONTHS + 1):
        post = int(m >= LAUNCH_MONTH)
        d = accounts["treated_region"].to_numpy() * post
        p = (
            0.55
            + 0.07 * engagement
            + 0.002 * np.minimum(tenure_months, 40)
            + 0.004 * np.minimum(seats, 20)
            + plan_lift
            + season[m - 1]
            + region_noise[region, m - 1]
            + TRUE_EFFECT * d
        )
        p = np.clip(p, 0.02, 0.98)
        active = rng.binomial(1, p)
        rows.append(pd.DataFrame({
            "account_id": accounts["account_id"],
            "region": region,
            "month": m,
            "post": post,
            "treated": d,
            "active": active,
        }))
    panel = pd.concat(rows, ignore_index=True)
    return accounts, panel


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    accounts, panel = simulate()
    accounts.to_csv(DATA_DIR / "accounts.csv", index=False)
    panel.to_csv(DATA_DIR / "monthly_activity.csv", index=False)
    print(f"Wrote {len(accounts):,} accounts and {len(panel):,} account-months to {DATA_DIR}")
    print(f"True effect: +{TRUE_EFFECT * 100:.1f} percentage points")


if __name__ == "__main__":
    main()
