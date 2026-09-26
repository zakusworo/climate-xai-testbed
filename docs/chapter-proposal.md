# Chapter proposal

**Book:** *Artificial Intelligence for Climate Learning and Sustainable Development*
(ISTIC under the auspices of UNESCO, with Universitas Negeri Malang; Springer Nature
Singapore, expected 2027)

**Preferred topic:** Part II, Explainable AI (XAI) for Climate Decision-Making
(secondary: Part I, Data Ecosystems for Climate Learning)

**Timing note:** The EOI deadline was 21 September 2026. This proposal is prepared
for a late-consideration request to the editor (hannah@istic-unesco.org), a later
call, or a standalone journal paper. It does not assume acceptance.

**Tentative title:** When Explanations Can Be Checked: A Physics-Anchored Audit of
Explainable AI for Evapotranspiration in Indonesia

**Author:** Zulfikar Aji Kusworo, Ministry of Energy and Mineral Resources of the
Republic of Indonesia (zakusworo@esdm.go.id)

## Abstract (≈270 words)

Explainable AI is increasingly used to justify climate-related decisions. Its
explanations are rarely checked, because the true drivers of real climate outcomes
are unknown. This chapter uses a target whose mechanism is known: FAO-56
Penman-Monteith reference evapotranspiration, which underlies irrigation and drought
water-demand planning. The inputs are 30,684 days of NASA POWER data from six
Indonesian sites, 2011–2024. Four model classes are trained to predict it. Each model
and the physical equation are explained with the same SHAP permutation explainer and
background sample, so disagreement reflects what the model learned rather than the
explanation method.

With the eight causal inputs, all models reach R² ≥ 0.978 and identify the true
dominant driver on 87–96% of days. Adding four correlated variables that the equation
never uses changes accuracy little but splits the models. Gradient boosting places
6% of its attribution on these proxies, while ridge regression (R² 0.982) places 41%
there and matches the true top driver on only 43% of days. Removing solar radiation,
which is often unmeasured in sparse station networks and carries 44% of the true
attribution, still leaves R² ≈ 0.90. Yet correct top-driver identification falls to
5–33%, and the missing effect is credited to cloud cover, temperature and rainfall.
A held-out semi-arid site, later years and 20% target noise degrade explanations only
modestly.

Accuracy is therefore a poor guide to explanation faithfulness. The largest failures
come from gaps in the data ecosystem, not from model choice. The chapter proposes
a practical checklist for agencies and educators: benchmark explainers on a
known-mechanism case, audit for missing drivers, and report attribution on
non-causal features before explanations inform decisions.

## Outline

1. Why climate decisions increasingly rest on XAI outputs, and why these are rarely verified
2. Known-mechanism benchmarking: rationale and FAO-56 as a test case
3. Data: NASA POWER for six Indonesian climates; limits of reanalysis
4. Method: identical-explainer comparison; faithfulness metrics
5. Results: model class, proxies, missing drivers, shift, noise, stability
6. Implications for Global South data ecosystems (radiation and station coverage)
7. Checklist for decision-makers and a classroom exercise on questioning explanations
8. Limits: smooth known function as an optimistic bound; Shapley definition choices

## Evidence status

| Claim | Support now | Needed before submission |
|---|---|---|
| FAO-56 implementation | Matches FAO-56 Examples 8 and 18 (unit tests) | — |
| All numbers in abstract | `results/metrics.csv`, `RESULTS.md` | Rerun with more seeds for all model classes, not only HGB |
| Robustness | 1 explanation set/case; E6 only for HGB | Seeds × models; larger explain set; conditional SHAP comparison |
| Literature framing | None yet | Verified citations (Crossref) for XAI-faithfulness and FAO-56 |
