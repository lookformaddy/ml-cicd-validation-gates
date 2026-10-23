"""Synthetic churn dataset — clean-room, seeded, fully reproducible.

Domain is deliberately generic (customer churn) so nothing here can leak or
resemble any employer's data. The label is a noisy function of the features,
so a linear model can learn real signal (~0.75-0.85 accuracy).
"""
from __future__ import annotations

import random

Row = dict[str, object]

SCHEMA: dict[str, str] = {
    "tenure_months": "int",
    "monthly_spend": "float",
    "support_tickets": "int",
    "contract_type": "str",
    "region": "str",
    "age_group": "str",
    "is_churned": "int",
}

FEATURE_NAMES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "contract_month_to_month",
    "contract_one_year",
    "region_west",
    "region_central",
    "region_east",
    "age_18_34",
    "age_35_54",
]


def generate_churn_dataset(n_rows: int, seed: int) -> list[Row]:
    """Generate a clean synthetic churn dataset. Deterministic in `seed`."""
    rng = random.Random(seed)
    rows: list[Row] = []
    for _ in range(n_rows):
        tenure = rng.randint(1, 72)
        spend = round(rng.uniform(20.0, 150.0), 2)
        tickets = rng.choices([0, 1, 2, 3, 4, 5], weights=[40, 25, 15, 10, 6, 4])[0]
        contract = rng.choices(
            ["month-to-month", "one-year", "two-year"], weights=[50, 30, 20]
        )[0]
        region = rng.choices(["west", "central", "east", "north"], weights=[30, 25, 25, 20])[0]
        age = rng.choices(["18-34", "35-54", "55+"], weights=[40, 35, 25])[0]
        logit = (
            -1.2
            + 0.55 * tickets
            - 0.03 * tenure
            + 0.004 * spend
            + (0.9 if contract == "month-to-month" else 0.0)
            + rng.gauss(0.0, 0.7)
        )
        churned = 1 if logit > 0 else 0
        rows.append(
            {
                "tenure_months": tenure,
                "monthly_spend": spend,
                "support_tickets": tickets,
                "contract_type": contract,
                "region": region,
                "age_group": age,
                "is_churned": churned,
            }
        )
    return rows


def featurize(row: Row) -> list[float]:
    """One-hot + numeric encoding. Feature order matches FEATURE_NAMES."""
    contract = row["contract_type"]
    region = row["region"]
    age = row["age_group"]
    return [
        float(row["tenure_months"]),
        float(row["monthly_spend"]),
        float(row["support_tickets"]),
        1.0 if contract == "month-to-month" else 0.0,
        1.0 if contract == "one-year" else 0.0,
        1.0 if region == "west" else 0.0,
        1.0 if region == "central" else 0.0,
        1.0 if region == "east" else 0.0,
        1.0 if age == "18-34" else 0.0,
        1.0 if age == "35-54" else 0.0,
    ]


def split_rows(
    rows: list[Row], holdout_fraction: float, seed: int
) -> tuple[list[Row], list[Row]]:
    """Deterministic train/holdout split."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_holdout = int(len(rows) * holdout_fraction)
    holdout_idx = set(idx[:n_holdout])
    train = [r for i, r in enumerate(rows) if i not in holdout_idx]
    holdout = [r for i, r in enumerate(rows) if i in holdout_idx]
    return train, holdout


def to_xy(rows: list[Row], label: str = "is_churned") -> tuple[list[list[float]], list[int]]:
    X = [featurize(r) for r in rows]
    y = [int(r[label]) for r in rows]
    return X, y
