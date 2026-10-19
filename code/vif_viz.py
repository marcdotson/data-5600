
import numpy as np
import pandas as pd

from plotnine import (
    ggplot, aes, annotate, geom_point,
    geom_linerange, geom_segment,
    scale_y_log10, scale_color_manual,
    labs, theme_minimal, theme,
    element_text, coord_cartesian
)


def plot_collinearity(vif_data):
    """
    Visualize VIF scores and optional confidence intervals.

    Required columns:
        Term, VIF

    Optional columns:
        VIF_CI_low, VIF_CI_high
    """
    df = vif_data.copy()

    if "VIF_CI_low" not in df.columns:
        df["VIF_CI_low"] = np.nan
    if "VIF_CI_high" not in df.columns:
        df["VIF_CI_high"] = np.nan

    if df["VIF"].isna().any() or (df["VIF"] < 1).any():
        raise ValueError("VIF values must be >= 1.")

    df["group"] = pd.cut(
        df["VIF"],
        bins=[0, 5, 10, np.inf],
        labels=[
            "Low (< 5)",
            "Moderate (< 10)",
            "High (>= 10)"
        ],
        right=False
    )

    colors = {
        "Low (< 5)": "#3aaf85",
        "Moderate (< 10)": "#1b6ca8",
        "High (>= 10)": "#cd201f"
    }

    y_top = max(10, np.ceil(df["VIF"].max())) * 1.15

    has_ci = df[
        ["VIF_CI_low", "VIF_CI_high"]
    ].notna().all(axis=1)

    df["ci_low_plot"] = df["VIF_CI_low"].clip(lower=1)
    df["ci_high_plot"] = df["VIF_CI_high"].clip(upper=y_top)

    p = (
        ggplot(df, aes(x="Term", y="VIF", color="group"))
        + annotate(
            "rect", xmin=-np.inf, xmax=np.inf,
            ymin=1, ymax=5,
            fill="#3aaf85", alpha=0.15
        )
        + annotate(
            "rect", xmin=-np.inf, xmax=np.inf,
            ymin=5, ymax=10,
            fill="#1b6ca8", alpha=0.15
        )
        + annotate(
            "rect", xmin=-np.inf, xmax=np.inf,
            ymin=10, ymax=np.inf,
            fill="#cd201f", alpha=0.15
        )
        + geom_point(size=3.5)
        + scale_color_manual(values=colors)
        + scale_y_log10()
        + coord_cartesian(ylim=(1, y_top))
        + labs(
            title="Collinearity",
            subtitle=(
                "High collinearity (VIF) may inflate "
                "parameter uncertainty"
            ),
            x=None,
            y="Variance Inflation Factor (VIF, log-scaled)",
            color=None
        )
        + theme_minimal()
        + theme(
            legend_position="bottom",
            plot_title=element_text(size=12),
            axis_title_y=element_text(size=10)
        )
    )

    if has_ci.any():
        p += geom_linerange(
            data=df.loc[has_ci],
            mapping=aes(
                ymin="ci_low_plot",
                ymax="ci_high_plot"
            ),
            size=0.8
        )

    return p

vif_results = pd.DataFrame({
    "Term": ["cyl", "disp", "gear", "wt"],
    "VIF": [5.6, 10.0, 1.5, 5.0],
    "VIF_CI_low": [2.8, 4.9, 1.05, 2.6],
    "VIF_CI_high": [7.8, 15.0, 2.1, 6.9]
})

plot_collinearity(vif_results)
