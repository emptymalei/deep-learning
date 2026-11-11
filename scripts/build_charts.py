# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=6.3,<7",
#     "pandas>=2.2",
# ]
# ///
"""Export interactive Altair charts as Vega-Lite JSON specs for the book.

Run from the repository root:

    uv run scripts/build_charts.py

Each spec is embedded in a page with the `vega` widget:

    <div class="dl-widget" data-widget="vega"
         data-spec="time-series/assets/timeseries-transformations/box-cox-lambda.vl.json" markdown>
    ![Static fallback](assets/timeseries-transformations/....png)
    </div>

The widget applies the site's light/dark colours at render time, so the specs
here only set layout and encodings.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "dl"

AIR_PASSENGERS_URL = (
    "https://raw.githubusercontent.com/unit8co/darts/master/datasets/AirPassengers.csv"
)


def box_cox_lambda() -> alt.TopLevelMixin:
    """Air passengers with a λ slider: transformed series and rolling spread."""
    df = pd.read_csv(AIR_PASSENGERS_URL)
    df.columns = ["month", "passengers"]

    lam = alt.param(
        name="lam",
        value=1.0,
        bind=alt.binding_range(min=-1, max=1.5, step=0.05, name="Box-Cox λ  "),
    )

    base = alt.Chart(df).transform_calculate(
        bc="abs(lam) < 1e-9 ? log(datum.passengers) "
        ": (pow(datum.passengers, lam) - 1) / lam"
    )

    series = (
        base.mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y(
                "bc:Q", title="transformed passengers", scale=alt.Scale(zero=False)
            ),
            tooltip=[
                alt.Tooltip("month:T", format="%Y-%m"),
                alt.Tooltip("passengers:Q"),
                alt.Tooltip("bc:Q", title="transformed", format=".3f"),
            ],
        )
        .properties(width="container", height=170, title="Transformed series")
        .add_params(lam)
    )

    spread = (
        base.transform_window(row="row_number()")
        .transform_window(rolling_std="stdev(bc)", frame=[-11, 0])
        .transform_filter("datum.row >= 12")
        .transform_joinaggregate(mean_std="mean(rolling_std)")
        .transform_calculate(relative="datum.rolling_std / datum.mean_std")
    )
    spread_chart = alt.layer(
        spread.mark_line(strokeWidth=1.8).encode(
            x=alt.X("month:T", title=None),
            y=alt.Y(
                "relative:Q",
                title="12-month std ÷ its mean",
                scale=alt.Scale(domain=[0, 2.5], clamp=True),
            ),
        ),
        alt.Chart().mark_rule(strokeDash=[4, 4]).encode(y=alt.datum(1)),
    ).properties(
        width="container",
        height=120,
        title="Rolling spread: flat at 1 means constant variance",
    )

    return (
        alt.vconcat(series, spread_chart, spacing=18)
        .resolve_scale(x="shared")
        .properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    )


CHARTS = {
    DOCS
    / "time-series/assets/timeseries-transformations/box-cox-lambda.vl.json": box_cox_lambda,
}


def main() -> None:
    for path, build in CHARTS.items():
        spec = build().to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
