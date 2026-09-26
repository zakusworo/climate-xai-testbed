"""Run the six XAI-faithfulness experiments and write results/*.csv.

E1 model class      : does a more accurate model give truer explanations?
E2 proxy features   : how much attribution goes to features the physics never reads?
E3 missing driver   : Rs removed (common with sparse station networks); where does its effect go?
E4 site / time shift: do explanations survive a held-out climate (Kupang) or later years?
E5 target noise     : measured-ET-like noise on the training target
E6 seed stability   : spread of global rankings across bootstrap refits

Usage: python scripts/run_experiments.py [--n-explain 300] [--n-background 100]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from climxai.data import build_dataset
from climxai.xai import CAUSAL, PROXIES, compare, global_importance, permutation_global, physics_fn, shapley

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def make_model(name: str, seed: int):
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=seed)
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=seed)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True, random_state=seed))
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV())
    raise ValueError(name)


def run_case(case: str, model_name: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str],
             args, seed: int = 0, noise: float = 0.0, extra: dict | None = None):
    rng = np.random.default_rng(seed)
    y = train["et0"].to_numpy()
    if noise:
        y = y * (1 + noise * rng.standard_normal(len(y)))
    model = make_model(model_name, seed).fit(train[features], y)
    pred = model.predict(test[features])

    expl = test.sample(args.n_explain, random_state=seed)
    bg = train.sample(args.n_background, random_state=seed + 1)
    phi_m = shapley(lambda X: model.predict(pd.DataFrame(X, columns=features)), expl[features], bg[features], seed)

    full = CAUSAL + [c for c in PROXIES if c in features]
    phi_t_full = shapley(physics_fn(full), expl[full], bg[full], seed)
    # Truth restricted to the model's features (a dropped driver's share is reported separately).
    phi_t = phi_t_full[:, [full.index(c) for c in features]]

    row = {
        "case": case, "model": model_name, "seed": seed, "noise": noise,
        "n_train": len(train), "n_test": len(test),
        "r2": r2_score(test["et0"], pred),
        "rmse_mm": mean_squared_error(test["et0"], pred) ** 0.5,
        **compare(phi_m, phi_t, features),
        "perm_spearman": permutation_global(model, test[features].sample(2000, random_state=seed),
                                            test["et0"].sample(2000, random_state=seed), seed)
        .corr(global_importance(phi_t, features), method="spearman"),
    }
    dropped = [c for c in CAUSAL if c not in features]
    if dropped:
        # Share of the true total attribution carried by drivers the model cannot see.
        g_full = global_importance(phi_t_full, full)
        row["dropped_true_share"] = float(g_full[dropped].sum())
        # Which visible features absorb the dropped driver's local effect?
        for d in dropped:
            true_d = phi_t_full[:, full.index(d)]
            for j, c in enumerate(features):
                row[f"corr_{d}_to_{c}"] = float(np.corrcoef(true_d, phi_m[:, j])[0, 1])
    if extra:
        row.update(extra)
    gi = {
        "case": case, "model": model_name, "seed": seed, "noise": noise,
        "model_global": global_importance(phi_m, features).to_dict(),
        "true_global": global_importance(phi_t_full, full).to_dict(),
    }
    return row, gi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-explain", type=int, default=300)
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = build_dataset()
    rng = np.random.default_rng(42)
    is_test = rng.random(len(df)) < 0.25
    tr, te = df[~is_test], df[is_test]
    with_proxies = CAUSAL + PROXIES

    rows, globals_ = [], []

    def log(r, g):
        rows.append(r)
        globals_.append(g)
        print(f"{r['case']:<22}{r['model']:<6} seed={r['seed']} noise={r['noise']:.2f} "
              f"R2={r['r2']:.3f} rho={r['global_spearman']:.2f} top1={r['local_top1_match']:.2f} "
              f"spur={r['spurious_share']:.3f}", flush=True)

    t0 = time.time()
    for m in ["hgb", "rf", "mlp", "ridge"]:  # E1 + E2
        log(*run_case("E1_causal_only", m, tr, te, CAUSAL, args))
        log(*run_case("E2_with_proxies", m, tr, te, with_proxies, args))
    no_rs = [c for c in with_proxies if c != "rs"]  # E3
    for m in ["hgb", "mlp"]:
        log(*run_case("E3_no_rs", m, tr, te, no_rs, args))
    # E4: hold out a climate and a period.
    log(*run_case("E4_holdout_kupang", "hgb", df[df.site != "kupang"], df[df.site == "kupang"], with_proxies, args))
    log(*run_case("E4_holdout_2021_24", "hgb", df[df.index.year <= 2020], df[df.index.year > 2020], with_proxies, args))
    for nz in [0.1, 0.2]:  # E5
        log(*run_case("E5_noise", "hgb", tr, te, with_proxies, args, noise=nz))
    for s in range(args.seeds):  # E6
        boot = tr.sample(frac=1.0, replace=True, random_state=100 + s)
        log(*run_case("E6_bootstrap", "hgb", boot, te, with_proxies, args, seed=s))

    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    (OUT / "global_importance.json").write_text(json.dumps(globals_, indent=1))
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = ""
    (OUT / "run_meta.json").write_text(json.dumps({
        "n_rows": len(df), "sites": sorted(df.site.unique()), "years": [int(df.index.year.min()), int(df.index.year.max())],
        "n_explain": args.n_explain, "n_background": args.n_background, "seeds": args.seeds,
        "git_rev": rev or None, "runtime_s": round(time.time() - t0, 1),
    }, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
