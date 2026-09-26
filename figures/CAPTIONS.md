# Figure captions

Numbers are from `results/metrics.csv` (run settings in `results/run_meta.json`).

**Figure 1.** Predictive accuracy against explanation faithfulness for 19 model runs
predicting FAO-56 reference evapotranspiration from NASA POWER data at six Indonesian
sites (2011–2024). Left: share of SHAP attribution given to features the FAO-56
equation does not use. Right: share of 200 explained days on which the model's
largest SHAP feature matches the largest true Shapley feature. Unlabelled points
cluster at R² ≥ 0.978 with ≤ 7% spurious attribution and ≥ 0.87 top-driver agreement.
Ridge regression with proxy features (R² 0.982) and both models trained without solar
radiation (R² 0.90) are the exceptions.

**Figure 2.** Normalised mean |SHAP| per feature for the FAO-56 equation (truth),
gradient boosting and ridge regression, all trained with four proxy features (*)
that the equation does not read. Ridge assigns 41% of its attribution to the proxies,
mainly specific humidity and observed mean temperature. Gradient boosting assigns 6%.

**Figure 3.** Experiment E3, solar radiation removed from the inputs. Bars show the
correlation, across 200 days, between the true Shapley value of solar radiation and
each visible feature's model SHAP value. The missing driver's effect appears in cloud
amount, maximum temperature, relative humidity and rainfall. Cloud amount and rainfall
play no role in the FAO-56 equation.
