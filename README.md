# Climate XAI Testbed

**Do explainable-AI methods tell the truth about climate drivers?** On a real
climate model we cannot check, because the true drivers are unknown. This testbed
uses a target whose mechanism *is* known: FAO-56 Penman-Monteith reference
evapotranspiration (ET0), the quantity behind irrigation scheduling and drought
water-demand estimates. The inputs are real daily NASA POWER data for six
Indonesian sites (2011–2024, 30,684 site-days).

Machine-learning models are trained to predict ET0. Each model and the FAO-56
equation are then explained with the **same** SHAP permutation explainer and the
**same** background sample, so any disagreement comes from what the model learned,
not from the explanation method. Features the equation never reads (rain, cloud,
observed mean temperature, specific humidity) have an exact Shapley value of zero.
Any attribution a model gives them is spurious by construction.

Target chapter: *AI for Climate Learning and Sustainable Development* (ISTIC–UNESCO /
UM, Springer 2027), Part II "Explainable AI (XAI) for Climate Decision-Making". See
[docs/chapter-proposal.md](docs/chapter-proposal.md).

## Experiments

| ID | Question | Setup |
|---|---|---|
| E1 | Does model class matter when features are clean? | HGB, RF, MLP, Ridge on the 8 causal inputs |
| E2 | Do models credit correlated but non-causal proxies? | E1 + 4 proxy features |
| E3 | What happens when a driver is unmeasured? | Solar radiation removed (common in sparse station networks) |
| E4 | Do explanations survive a shift? | Train without Kupang → test on Kupang; train 2011–20 → test 2021–24 |
| E5 | Measurement noise on the target | 10% and 20% multiplicative noise |
| E6 | Stability | 5 bootstrap refits |

Metrics: test R², RMSE; Spearman ρ between model and true global importance; share
of days where the model's top local driver equals the true one; median cosine
similarity of per-day SHAP vectors; spurious share (attribution on proxies);
permutation-importance ρ against truth. For E3, the correlation between the true
solar-radiation SHAP and each visible feature's model SHAP shows where the missing
driver's effect is credited.

## Results

See [RESULTS.md](RESULTS.md), generated from `results/metrics.csv`.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                        # FAO-56 Examples 8 and 18, zero-Shapley and efficiency checks
.venv/bin/python scripts/run_experiments.py --n-explain 200 --n-background 80 --seeds 5   # ~10 min on a laptop CPU
.venv/bin/python scripts/make_figures.py   # figures/*.png + *.eps (print-safe)
```

`data/cache/` holds the NASA POWER snapshot used for the reported results. Its
request URLs are in the `.meta.json` files. Delete it to re-download.

## Honest limits

- ET0 is a **smooth, known** function with 8 inputs. Real climate targets (yield,
  flood, dengue) are noisier and confounded, so faithfulness there is likely *worse*,
  not better. The testbed gives an optimistic bound.
- NASA POWER is reanalysis on a ~0.5° grid. Coastal cells (Kupang, Makassar) have
  sea-damped temperature ranges. This does not affect the testbed, whose ground truth
  is the equation on these same inputs, but these are not measured site conditions.
- Rso uses sea-level form (z = 0) and G = 0 (daily step). Both are standard simplifications.
- Interventional SHAP with an independent masker evaluates the equation at
  off-manifold input combinations. That is the definition used for both model and
  truth, so the comparison is fair, but it is one of several Shapley definitions.
- One explanation set per case (200 days, 80-sample background). E6 gives the
  refit spread only for HGB.
