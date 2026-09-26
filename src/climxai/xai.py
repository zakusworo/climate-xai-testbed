"""Explain a model and the FAO-56 equation with the same explainer, then compare.

Both the fitted model and the physics are explained with SHAP's permutation
explainer against the same background sample (interventional Shapley values).
Because the explanation definition is identical, any disagreement comes from
what the model learned, not from differences between explanation methods.
Features the physics never reads receive exactly zero Shapley value, so any
attribution a model gives them is spurious by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from .fao56 import et0

CAUSAL = ["tmax", "tmin", "rh", "u2", "rs", "ps", "lat", "doy"]
PROXIES = ["tmean_obs", "qv", "precip", "cloud"]


def physics_fn(columns: list[str]):
    """Return f(X) evaluating FAO-56 ET0 from the causal columns of X."""
    idx = {c: columns.index(c) for c in CAUSAL if c in columns}
    missing = set(CAUSAL) - set(idx)
    if missing:
        raise ValueError(f"physics needs {sorted(missing)}")

    def f(X):
        X = np.asarray(X, dtype=float)
        return et0(*(X[:, idx[c]] for c in CAUSAL))

    return f


def shapley(fn, X_explain: pd.DataFrame, X_background: pd.DataFrame, seed: int = 0) -> np.ndarray:
    masker = shap.maskers.Independent(X_background.to_numpy(), max_samples=len(X_background))
    explainer = shap.PermutationExplainer(fn, masker, seed=seed)
    n_feat = X_explain.shape[1]
    out = explainer(X_explain.to_numpy(), max_evals=max(2 * n_feat + 1, 10 * n_feat), silent=True)
    return out.values


def global_importance(phi: np.ndarray, columns: list[str]) -> pd.Series:
    imp = pd.Series(np.abs(phi).mean(axis=0), index=columns)
    return imp / imp.sum()


def compare(phi_model: np.ndarray, phi_true: np.ndarray, columns: list[str]) -> dict:
    """Agreement metrics between model and true attributions (same columns)."""
    g_model = global_importance(phi_model, columns)
    g_true = global_importance(phi_true, columns)
    num = (phi_model * phi_true).sum(axis=1)
    den = np.linalg.norm(phi_model, axis=1) * np.linalg.norm(phi_true, axis=1)
    cos = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    proxies = [c for c in columns if c in PROXIES]
    return {
        "global_spearman": spearmanr(g_model, g_true).statistic,
        "top1_global_match": g_model.idxmax() == g_true.idxmax(),
        "local_top1_match": float(
            (np.abs(phi_model).argmax(axis=1) == np.abs(phi_true).argmax(axis=1)).mean()
        ),
        "local_cosine_median": float(np.nanmedian(cos)),
        "local_phi_mae_mm": float(np.abs(phi_model - phi_true).mean()),
        "spurious_share": float(g_model[proxies].sum()) if proxies else 0.0,
    }


def permutation_global(model, X: pd.DataFrame, y, seed: int = 0) -> pd.Series:
    r = permutation_importance(model, X, y, n_repeats=5, random_state=seed, n_jobs=1)
    imp = pd.Series(np.clip(r.importances_mean, 0, None), index=X.columns)
    return imp / imp.sum()
