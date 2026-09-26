# Climate XAI Testbed — Progress Log

_Last updated: 2026-09-26_

Handoff notes for the next working session. README describes *what the testbed is*;
this file tracks *where the work stands* and *what to do next*.

Target: *AI for Climate Learning and Sustainable Development* (ISTIC–UNESCO / UM,
Springer Nature Singapore, 2027), Part II "Explainable AI (XAI) for Climate
Decision-Making". The EOI deadline (21 Sep 2026) had already passed when this project
was created. The proposal in `docs/chapter-proposal.md` is for a late-consideration
request, a later call or a journal paper. **Nothing has been submitted.**

## Resume here

The repository is `git init`'d with **no commits**. The first commit should include
`data/cache/` (1 MB NASA POWER snapshot), because the reported numbers depend on it.
Keep `git_rev` in `results/run_meta.json` in mind: it is `null` for the current results
because they were produced before any commit. Rerun after committing to record a revision.

Immediate next steps:

1. Commit (author: Zulfikar Aji Kusworo, no Claude trailer), then rerun
   `scripts/run_experiments.py` so `run_meta.json` records a git revision.
2. Robustness: extend seeds/bootstrap refits to **all four model classes** (E6 currently
   covers HGB only) and raise `--n-explain` above 200. Report spread, not single values.
3. Add a conditional/observational SHAP comparison (e.g. TreeExplainer
   `feature_perturbation="tree_path_dependent"`) to show how the Shapley definition
   changes the spurious-share result.
4. Build the reference list with the verified-citations workflow (Crossref): FAO-56,
   SHAP, XAI-faithfulness literature, NASA POWER. No citations exist yet.
5. Decide whether to email the editor (hannah@istic-unesco.org) about late consideration.
   This is the author's call; no email has been drafted or sent.

## 2026-09-26 — project created

- Implemented FAO-56 daily Penman-Monteith (`src/climxai/fao56.py`). It reproduces FAO-56
  Example 18 (ET0 3.9 mm/day, tolerance 0.05) and Example 8 (Ra 32.2 MJ m⁻² day⁻¹).
- NASA POWER daily data for six sites (Sleman, Malang, Pontianak, Medan, Makassar,
  Kupang), 2011–2024, 30,684 site-days, cached as parquet with request URLs in `.meta.json`.
- Identical-explainer design: the model and FAO-56 are both explained with the SHAP
  PermutationExplainer against the same background. Tests confirm exact-zero Shapley for
  non-causal features (≤1e-6) and the efficiency property.
- Full run: 19 cases, `--n-explain 200 --n-background 80 --seeds 5`, 608 s on laptop CPU.
- Key results (see `RESULTS.md`):
  - E1 clean features: R² ≥ 0.978, correct top driver on 87–96% of days.
  - E2 + proxies: Ridge R² 0.982, 41% spurious attribution, top-driver match 0.43;
    HGB 6%.
  - E3 without solar radiation (44% of true attribution): R² ≈ 0.90, top-driver match
    0.05 (MLP) / 0.33 (HGB); effect credited to cloud (r 0.73), Tmax (0.65), RH (~0.6),
    rain (~0.5).
  - E4 Kupang holdout R² 0.978 and top-driver match 0.875; E5 20% noise top-driver match 0.905.
  - E6 HGB bootstrap: top-driver match 0.88–0.95.
- **Metric correction:** all-feature Spearman ρ is depressed by proxies tied at a true
  importance of 0. `scripts/make_results.py` adds `causal_spearman` (8 causal inputs only).
  Ridge E2 has causal ρ 0.83, not the 0.28 all-feature value. Figure 1 therefore plots
  spurious share and top-driver match, not all-feature ρ.
- Figures 1–3 (PNG 400 dpi + EPS) rendered and checked visually; captions in
  `figures/CAPTIONS.md`.

## Known limits (do not overclaim)

- ET0 is a smooth known function, so the results are an **optimistic bound** for real
  climate targets.
- NASA POWER is reanalysis (~0.5° grid), and coastal cells have damped temperature
  ranges. It is valid as testbed input, but it does not describe measured site conditions.
- Rso uses z = 0 and G = 0 (daily). Interventional SHAP evaluates off-manifold combinations.
- One explanation set per case, and E6 covers only HGB.
