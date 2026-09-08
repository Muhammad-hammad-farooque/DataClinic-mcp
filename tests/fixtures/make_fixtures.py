"""Generate deliberately broken datasets.

Every defect here is one a real export produces and one the server is expected
to notice. The sorted file matters most: it is the case where sampling the
first N rows -- as competing servers do -- reports statistics that are simply
wrong.

Run: uv run python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROWS = 5000


def messy(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = ROWS

    df = pd.DataFrame(
        {
            # unique per row: an identifier, not a feature
            "customer_id": [f"CUST{i:06d}" for i in range(n)],
            # right-skewed with genuine extremes
            "price": np.round(rng.lognormal(3.2, 0.9, n), 2),
            "age": rng.integers(18, 80, n).astype(float),
            # inconsistent categorical encoding
            "country": rng.choice(["USA", "usa", "U.S.A.", "UK", "uk", "France"], n),
            # numbers stored as text, with thousands separators
            "revenue": [f"{v:,.2f}" for v in rng.uniform(100, 90000, n)],
            # dates in one format, as text
            "signup_date": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
            # a single value throughout: no signal
            "region_code": ["EMEA"] * n,
            # entirely empty
            "notes": [None] * n,
            # imbalanced target
            "churned": rng.choice([0, 1], n, p=[0.94, 0.06]),
            # leaks the target
            "churn_score": np.zeros(n),
        }
    )

    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")
    df["churn_score"] = df["churned"] * 0.98 + rng.normal(0, 0.01, n)

    # missing not at random: later signups lack a country
    late = df.index > n * 0.6
    drop = late & (rng.random(n) < 0.7)
    df.loc[drop, "country"] = None

    # moderate missingness elsewhere
    df.loc[rng.random(n) < 0.25, "age"] = np.nan

    # disguised nulls that pandas would otherwise read as text
    df.loc[rng.random(n) < 0.05, "revenue"] = "N/A"
    df.loc[rng.random(n) < 0.03, "revenue"] = "-"

    # impossible values
    df.loc[rng.random(n) < 0.01, "age"] = -5

    # exact duplicate rows
    df = pd.concat([df, df.head(150)], ignore_index=True)
    return df


def write_all() -> None:
    df = messy()

    df.to_csv(HERE / "messy.csv", index=False)

    # Sorted by price: any tool reading only the first rows sees the cheapest
    # end of the distribution and reports a badly wrong mean.
    df.sort_values("price").to_csv(HERE / "messy_sorted.csv", index=False)

    # Semicolon-delimited, cp1252, with non-ASCII text -- a European
    # spreadsheet export. The accented characters are the point: they are what
    # makes encoding detection do real work.
    euro = df.head(500).copy()
    euro["country"] = ["España", "Français", "München", "Danmark"] * 125
    euro.to_csv(HERE / "messy_semicolon.csv", index=False, sep=";", encoding="cp1252")

    pd.DataFrame({"a": [], "b": []}).to_csv(HERE / "empty.csv", index=False)

    for path in sorted(HERE.glob("*.csv")):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    write_all()
