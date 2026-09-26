"""Print-safe chapter figures from results/metrics.csv and results/global_importance.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
CAT = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#9B4F96", "#56B4E9"]
HATCH = ["", "///", "...", "\\\\\\", "xxx", "ooo"]
MARK = ["o", "s", "^", "D", "v", "P"]
plt.rcParams.update({
    "savefig.dpi": 400, "savefig.bbox": "tight", "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "axes.axisbelow": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.8, "hatch.linewidth": 0.55,
})
LABEL = {"tmax": "Tmax", "tmin": "Tmin", "rh": "RH", "u2": "Wind", "rs": "Solar rad.", "ps": "Pressure",
         "lat": "Latitude", "doy": "Day of year", "tmean_obs": "Tmean*", "qv": "Spec. hum.*",
         "precip": "Rain*", "cloud": "Cloud*"}


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png")
    fig.savefig(FIG / f"{stem}.eps")
    plt.close(fig)


def fig1_accuracy_vs_fidelity(m):
    groups = [("E1_causal_only", "E1 causal features"), ("E2_with_proxies", "E2 + proxy features"),
              ("E3_no_rs", "E3 solar radiation removed"), ("E4_holdout_kupang", "E4 held-out site/period"),
              ("E4_holdout_2021_24", None), ("E5_noise", "E5 noisy target"), ("E6_bootstrap", "E6 bootstrap refits")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharex=True)
    k = 0
    for case, label in groups:
        d = m[m.case == case]
        if label is None:
            idx = k - 1
        else:
            idx, k = k, k + 1
        for ax, col in zip(axes, ["spurious_share", "local_top1_match"]):
            ax.scatter(d.r2, d[col], color=CAT[idx % 6], marker=MARK[idx % 6], s=26,
                       edgecolor="black", linewidth=0.4, label=label if ax is axes[0] else None)
    # Label only the outliers; the high-accuracy cluster is described in the caption.
    names = {"ridge": "Ridge", "hgb": "Gradient boosting", "mlp": "Neural net"}
    for _, r in m[((m.case == "E2_with_proxies") & (m.model == "ridge")) | (m.case == "E3_no_rs")].iterrows():
        for ax, col in zip(axes, ["spurious_share", "local_top1_match"]):
            ax.annotate(names[r.model], (r.r2, r[col]), fontsize=6.5, xytext=(5, -3), textcoords="offset points")
    axes[0].set_ylabel("Attribution on non-causal proxies")
    axes[1].set_ylabel("Share of days with correct top driver")
    for ax in axes:
        ax.set_xlabel("Test R² of ET0 prediction")
    axes[0].set_ylim(-0.03, 0.55)
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle("High predictive accuracy does not guarantee faithful explanations",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig1_accuracy_vs_fidelity")


def fig2_global_bars(gi):
    pick = {(g["case"], g["model"]): g for g in gi if g["seed"] == 0 and g["noise"] == 0}
    truth = pick[("E2_with_proxies", "hgb")]["true_global"]
    series = [("FAO-56 truth", truth), ("Gradient boosting", pick[("E2_with_proxies", "hgb")]["model_global"]),
              ("Ridge regression", pick[("E2_with_proxies", "ridge")]["model_global"])]
    feats = sorted(truth, key=lambda f: -truth[f])
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    w = 0.27
    for i, (name, s) in enumerate(series):
        ax.bar([x + (i - 1) * w for x in range(len(feats))], [s.get(f, 0) for f in feats], w,
               color=CAT[i], hatch=HATCH[i], edgecolor="black", linewidth=0.4, label=name)
    ax.set_xticks(range(len(feats)), [LABEL[f] for f in feats], rotation=30, ha="right")
    ax.set_ylabel("Share of mean |SHAP|")
    ax.set_title("A linear model with R² 0.98 puts much of its attribution on non-causal proxies (*)")
    ax.legend()
    save(fig, "fig2_global_importance_e2")


def fig3_absorption(m):
    d = m[(m.case == "E3_no_rs")].set_index("model")
    cols = [c for c in d.columns if c.startswith("corr_rs_to_")]
    feats = [c.replace("corr_rs_to_", "") for c in cols]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    w = 0.4
    for i, model in enumerate(d.index):
        ax.bar([x + (i - 0.5) * w for x in range(len(cols))], d.loc[model, cols].astype(float), w,
               color=CAT[i], hatch=HATCH[i], edgecolor="black", linewidth=0.4,
               label={"hgb": "Gradient boosting", "mlp": "Neural network"}.get(model, model))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(cols)), [LABEL[f] for f in feats], rotation=30, ha="right")
    ax.set_ylabel("Corr(true solar-rad. SHAP,\nmodel SHAP of feature)")
    ax.set_title("With solar radiation missing, its effect is credited to cloud, temperature and humidity")
    ax.legend()
    save(fig, "fig3_missing_driver_absorption")


def main():
    FIG.mkdir(exist_ok=True)
    m = pd.read_csv(RES / "metrics.csv")
    gi = json.loads((RES / "global_importance.json").read_text())
    fig1_accuracy_vs_fidelity(m)
    fig2_global_bars(gi)
    fig3_absorption(m)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
