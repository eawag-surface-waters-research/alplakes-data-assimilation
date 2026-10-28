# Experiments

*Last updated: October 2026. Unless stated otherwise: 2025, 20 members, native EnKF, operational setup apart from the one thing each experiment changes.*

!!! abstract "In short"
    Every experiment we ran, what it asked, and what came of it. Daily assimilation, seiche filtering and profile data pay off. Making σ more refined makes the filter better calibrated, not more accurate. Surface-only assimilation damages the layers below. Extra spread from model parameters helps a little at daily steps, but only at amplitudes we cannot defend; at longer steps it pays off. Assimilation leaves the model's natural variability intact.

    The more recent work is promising: oxygen can be assimilated together with temperature, with large gains where the profiler covers the column; centring the forcing ensemble removes a systematic windiness of the members; and process noise gives an honest forecast uncertainty band. Deep mixing remains open.

✅ Operational · (✅) Positive, potentially to be adopted · ⚙️ Optional · ❌ Tested, not adopted · 🔬 Open

| Experiment | Question | Outcome | Status |
|---|---|---|---|
| [Assimilation frequency](#frequency) | Daily, or less often? | Daily is best: −26 % vs −23 % (3 days) and −20 % (7 days) | ✅ daily |
| [Observation filtering](#filtering) | Does seiche filtering help? | Yes: without it, nothing is gained at the thermocline | ✅ |
| [Surface-only](#surface-only) | Is a surface station enough? | Helps the surface, damages 10–35 m against CTD | ❌ |
| [Depth-resolved σ](#sigma-depth) | Should σ vary with depth? | Better shape and distribution, same accuracy; fitted on the free run, so its generalization to assimilation runs and other years is untested | ❌ |
| [Rescaled σ](#sigma-scaled) | Can σ be fitted to the filter's own innovations iteratively? | NIS ≈ 1 everywhere, accuracy unchanged, more complex and less generalizable | ❌ |
| [Adaptive σ](#adaptive-sigma) | Can σ follow the innovations on the fly? | Fewer assumptions: σ per depth and from recent innovations, NIS ≈ 1. Unbounded it can feed back on itself (six of seven lakes less accurate), so it needs limits | (✅) |
| [Fitting year](#fitting-year) | Does fitting σ on 2025 bias the results? | Barely: gain within 1.5 points, the method is not very sensitive and can be used also on unseen years | ✅ passes |
| [Vertical localization](#localization) | Are spurious correlations the problem at depth? | Small gains only, no clear advantage in using operationally for now, but potentially useful in the future on a different setup or application | (✅) |
| [Parameter sensitivity](#param-sensitivity) | Could parameters add spread at depth? | Yes: ±1 °C at 11 m in summer | 🔬 |
| [Parameter perturbation](#param-perturbation) | Does perturbing them help the filter? | Daily: only slightly, and at amplitudes we cannot defend. With a longer assimilation step, though, it helps: every 3 days, `p_lw` ±10 % brings the accuracy back to the daily result | (✅) |
| [Joint estimation](#joint) | Can the filter estimate the parameters? | They don't converge; but they absorb the seasonal bias | 🔬 |
| Deep mixing: [Lugano](#deep-mixing) & [Geneva](#geneva-ctd) | Can CTD assimilation fix deep mixing? | Temperature yes (Lugano 30–100 m 0.29 → 0.16 °C; Geneva 1.02 → 0.62 °C, O₂ 105 → 38). Mixing no: more frequent on Lugano (uncentred forcing), unchanged and still too frequent on Geneva (centred) | 🔬 |
| [3D model](#3d) | Does the layout transfer to Delft3D? | Yes, with promising first results | (✅) |
| [Natural variability](#variability) | Does DA disturb the model's own oscillations? | No: same spectral shape, one clean step per update | ✅ passes |
| Oxygen: [assimilation](#to2-profiler), [depth thinning](#depth-thinning) & [CTD interpolation](#ctd-interpolation) | Can O₂ be assimilated with T, from how few depths, and from casts alone? | Profilers: Ägeri O₂ −45 %, Murten −18 %; the bottom boundary is a model problem. 5 of 23 depths keep ~35–40 % of the T gain, ~70 % of the O₂ gain. Monthly casts: good between casts in the deep water only | (✅) |
| [Centred forcing](#centred-forcing) | Does the forcing noise bias the ensemble mean? | Yes: mean wind speed +31 to +63 %; centring removes it | (✅) |
| [Uncertainty band](#uncertainty-band) | Can process noise give an honest forecast band? | Yes: 83 % (2025, in sample) and 80 % (2026, out of sample) inside the 90 % band, against ~25 %; RMSE −22 % / −17 % | (✅) |

---

## Observations

### Assimilation frequency { #frequency }

**Status:** ✅ Operational (daily) · **Lakes:** Upper Lugano

**Question:** How much accuracy is lost by assimilating less often than daily?

**Setup:** Same run at daily, 3-day and 7-day frequency, scored against hourly buoy data.

**Result:** The free run's 0.83 °C drops to 0.61 °C (−26 %) with daily assimilation (365 profiles), 0.64 °C (−23 %) every 3 days (122 profiles) and 0.67 °C (−20 %) every 7 days (53 profiles).

**Verdict:** Daily outperforms the lower frequencies we tested ([Method](method.md#timing)).

**Next step:** With more model spread, lower frequencies may catch up: with `p_lw` ±10 %, 3-day assimilation almost matches the daily result ([Parameter perturbation](#param-perturbation)).

**To reproduce:** `feat/da-combi`: `notebooks/generate_assimilation_series.py --every-days N`.

### Observation filtering { #filtering }

**Status:** ✅ Operational · **Lakes:** Upper Lugano

**Question:** What does the seiche filtering ([Method](method.md#observations)) actually gain?

**Setup:** Assimilation in 2025 on unfiltered vs filtered observations, both scored (pooled RMSE °C over the year) against the same raw hourly data.

![Upper Lugano: assimilation on unfiltered vs filtered observations](../images/12_filtering_effect.png){ width="800" .center }
/// figure-caption
Upper Lugano 2025: RMSE by depth with assimilation of unfiltered (left) and filtered (right) observations, scored against the same raw hourly observations.
///

**Result:** Without filtering (Figure 1, left) almost nothing is achieved at the thermocline, and the red fill around 13 and 17 m says it was made slightly worse: the filter is chasing a signal it cannot represent and fails to inject it into the simulation. With filtering (right) we get an improvement everywhere.

**Verdict:** The data becomes more meaningful, and more digestible, for the filter and we get the hoped-for improvement at intermediate depths. All other results use filtered observations.

**Reproduce:** `feat/da-combi`: `notebooks/calibrate_filter.py`, `notebooks/filter_observations.py`.

### Surface-only assimilation { #surface-only }

**Status:** ❌ Tested, not adopted · **Lakes:** Lower Lugano, Lower Zürich (plus Upper Lugano with surface data only)

**Question:** Several lakes have a surface station and nothing else. Is assimilating the surface enough?

**Setup:** Two lake sections where only surface data is available, with the same seasonal σ recipe applied to the single surface depth: Lower Lugano 0.41 °C mixed / 0.61 °C stratified, Lower Zürich 0.65 / 0.88 °C.

![Lower Zürich and Lower Lugano, surface assimilation through the year](../images/16_surface_only_timeseries.png){ width="800" .center }
/// figure-caption
Surface-only assimilation through 2025 at 1 m: Lower Zürich and Lower Lugano, free run vs assimilation.
///

**Result at the surface:** Both lakes improve at the assimilated depth of 1 m, but very unequally. **Lower Lugano 0.90 → 0.58 °C (−35 %); Lower Zürich 0.80 → 0.68 °C (−15 %).** The difference most likely comes from the error each free run makes, not from the filter. Each correction is the innovation (observation minus forecast) times a gain, and that gain stays small when the ensemble is narrow compared with σ. Our setup has a fairly narrow ensemble and a large σ, so a correction can only be large when the error is large. When the free-run error stays low, the corrections are small and hard to see. Zürich's error is low (around 0.4 °C) for most of the year, with only short large excursions, so the filter has less to correct over the year. In January–February, and also in November–December for Zürich, the assimilated and free-run curves in Figure 2 are indistinguishable. The difference appears in March for Lugano and April for Zürich and lasts through the stratified season.

![Surface-only vs profile assimilation, scored against CTD casts](../images/17_surface_vs_profile_ctd.png){ width="800" .center }
/// figure-caption
RMSE against CTD casts: surface-only assimilation, Lower and Upper Lugano (left), and full-profile assimilation, Upper Lugano (right).
///

**Result in the column:** Figure 3, left: Lower and Upper Lugano with surface-only assimilation; right: Upper Lugano assimilating the full profile. Both scored against CTD casts, a sub-optimal reference: casts are instantaneous and very exposed to the variability problem described in [Method](method.md#observations), while our assimilation works on filtered daily values, and the two panels are different lake sections. Surface-only gains at the top and, weakly, in the deep in lower Lugano. Similar results are obtained when assimilating the upper Lake Lugano section with surface measurements only (note: deep results change). **Between them it loses badly: the RMSE degrades by up to 0.65 °C between roughly 10 and 35 m, more than the 0.4 °C it gains at the surface.**

**Profile assimilation gains almost everywhere** down to 60 m, up to 0.7 °C at the surface, and below that does essentially nothing, within 0.02 °C of the free run all the way to 280 m.

**Alternatives tried:** The model is 1D, so no matter how many stations report, the filter must be handed one value per depth. The freedom is in how that value and its uncertainty are constructed. We tried defining σ from **the observations alone** rather than from the model gap: when the spread *between stations* is large (from climatology), trust the value less; when a value is derived from *more* stations at run time, trust it more. We tried localizing the corrections ([Localization](#localization)), and arbitrarily prescribing a low σ of 0.3 °C. **The results change only at the surface and none of them improved the comparison against the CTD profiles.**

**Verdict:** Profile assimilation shows clearly better gains against CTD, and it appears to do little harm where it has no information; **having profile information over the whole upper 30–40 m seems to be essential for correcting a 1D model.**

**Reproduce:** `feat/multi-station-obs`: Lower Lugano and Lower Zürich setups, `notebooks/sigma_rep_from_obs.py`.

---

## Observation σ { #sigma }

A single fixed σ by season ([Method](method.md#sigma)) is simple, well suited to operational use, and already gives significant improvement across the year and the water column. But the simple recipe has clear drawbacks, and each one suggests a fix:

- Seasonal aggregation assumes a clean mixed/stratified split, at the same calendar dates, across lakes and years. A finer temporal resolution (say monthly) would need more history than we have, and monthly conditions vary a lot from year to year.
- Thermocline layers in summer remain challenging to assimilate.
- Deep observations are trusted far too little, because their true representativeness error is much smaller than the depth-median.

In the tables below, "calibration" is the pooled NIS: 1 is perfectly consistent, below 1 over-cautious, above 1 over-confident.

### Depth-resolved σ { #sigma-depth }

**Status:** ❌ Tested, not adopted · **Lakes:** all seven

**Question:** Does σ resolved by depth, instead of collapsed to a median, do better?

**Setup:** Same fitting concept, resolved by depth.

| lake | gain, simple | gain, σ(z) | NIS simple (all / mixed / strat.) | NIS σ(z) (all / mixed / strat.) |
|---|---|---|---|---|
| upperlugano | −26.3 % | −25.5 % | 0.55 / 0.73 / 0.51 | 0.56 / 0.68 / 0.52 |
| maggiore | −28.5 % | −28.4 % | 0.87 / 0.38 / 1.04 | 0.91 / 0.38 / 1.13 |
| greifensee | −27.1 % | −28.5 % | 0.63 / 0.55 / 0.64 | 0.65 / 0.41 / 0.73 |
| geneva | −42.5 % | −41.8 % | 1.07 / 1.54 / 0.98 | 0.62 / 1.18 / 0.55 |
| aegeri | −35.2 % | −28.5 % | 0.94 / 0.66 / 1.14 | 0.75 / 0.63 / 0.79 |
| hallwil | −49.2 % | −50.6 % | 1.00 / 0.54 / 1.46 | 0.62 / 0.41 / 0.79 |
| murten | −34.2 % | −30.9 % | 0.60 / 0.96 / 0.50 | 0.62 / 0.66 / 0.59 |

**Result:** The structure of the budget looks visibly better: σ now follows the shape of the innovation profile instead of cutting across it. But the overall calibration stays about the same, because fixing the shape redistributes the misfit rather than removing it (the budget is more evenly distributed, not better in the pool). **And the RMSE gain barely moves**, while on Ägeri it gets notably worse (−35.2 → −28.5 %). The level stays too high for the same reason as in the seasonal σ of `simplified_final`: σ is still fitted on the free run's errors, and the purpose of assimilation is to make exactly those errors smaller, so the filter only ever sees smaller innovations than the ones σ was fitted on ([Rescaled σ](#sigma-scaled)).

**Verdict:** The depth fit changes the *shape* of the uncertainty profile but not its *level*, and it brings no gain over the seasonal σ: for the moment we see no reason to use it. Fixing the level is the next experiment.

**Reproduce:** `feat/da-combi`: `src/assimilator/sigma_rep.py`, `notebooks/fit_sigma_gap.py`.

### Rescaled σ { #sigma-scaled }

**Status:** ❌ Tested, not adopted · **Lakes:** all seven

**Question:** The depth fit had the right shape and the wrong amplitude. Does fixing the level help?

**Setup:** Since the free-run fit sets σ too high ([above](#sigma-depth)), we take a previous run's uncertainty profile and rescale it against **that run's own innovations**, with four numbers per lake to keep the risk of overfitting as low as possible: above and below 4 m (where the seiche filtering starts), for each of the two seasons.

![Uncertainty budget: depth-resolved σ (left) vs rescaled (right), Upper Lugano](../images/18_sigma_depth_vs_scaled.png){ width="800" .center }
/// figure-caption
Uncertainty budget with depth-resolved σ (left) and rescaled σ (right), Upper Lugano 2025.
///

| lake | gain, σ(z) | gain, scaled | NIS σ(z) (all / mixed / strat.) | NIS scaled (all / mixed / strat.) |
|---|---|---|---|---|
| upperlugano | −25.5 % | −26.9 % | 0.56 / 0.68 / 0.52 | 0.91 / 0.96 / 0.89 |
| maggiore | −28.4 % | −25.7 % | 0.91 / 0.38 / 1.13 | 1.13 / 0.87 / 1.16 |
| greifensee | −28.5 % | −30.5 % | 0.65 / 0.41 / 0.73 | 0.93 / 0.93 / 0.93 |
| geneva | −41.8 % | −41.9 % | 0.62 / 1.18 / 0.55 | 0.97 / 1.02 / 0.96 |
| aegeri | −28.5 % | −28.6 % | 0.75 / 0.63 / 0.79 | 1.00 / 0.94 / 1.01 |
| hallwil | −50.6 % | −49.9 % | 0.62 / 0.41 / 0.79 | 1.05 / 1.01 / 1.07 |
| murten | −30.9 % | −32.8 % | 0.62 / 0.66 / 0.59 | 0.94 / 0.99 / 0.91 |

**Result:** Figure 4 compares the two budgets on Upper Lugano. **As a calibration test this works very well**, as expected: every lake lands close to 1. Spring and winter remain hard in opposite directions, which is the price of the remaining two-season approximation. **The RMSE gains move only a little, some up, some down, and the simple setup still beats this calibrated version in several cases:** Some general tendencies are nonetheless visible. Calibration tends to score worse in spring, and that loss sits mainly at the surface. The depth-fitted experiments, on the other hand, tend to score better in deeper water and, for the scaled fit, in the winter months. At the thermocline we cannot generalise: of the two lakes with the largest losses there, Greifensee improves, closing part of its intermediate-depth degradation, while Ägeri gets worse.

**Verdict:** Filter self-consistency and filter accuracy are separate objectives: calibrating does not guarantee any improvement in performance. Calibration can offer a targeted repair at difficult depths and times, but not a reliable one. How well more complex or iterative fits would generalise to other years with the limited data at our disposal also remains questionable.

**Reproduce:** `feat/da-combi`: `src/assimilator/sigma_rep.py`, `notebooks/fit_scale_by_depth.py`.

### Adaptive σ { #adaptive-sigma }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** all seven; CTD check on Upper Lugano

**Question:** Both fits above are made offline, on a reference year, and assume that year stands for the next ones; the scaled fit also needs several steps. Can σ instead follow the innovations as the run goes?

**Setup:** At each depth, σ² is the mean squared innovation minus the mean ensemble spread² over the 14 days before each analysis, `σ(z,t)² = gap² − spread²`. It needs no free run, no seasonal split, and no multiplier to fit iteratively. It is causal, and has one safeguard: a 0.05 °C floor. The 14-day window is needed to collect enough samples given the seiche filtering. The seasonal σ is only used for the first 14 days, until the window fills.

![Adaptive σ actually used, native EnKF vs OpenDA chain, against the seasonal σ, Upper Lugano](../images/19_adaptive_sigma_time.png){ width="800" .center }
/// figure-caption
Adaptive σ actually used, native EnKF vs OpenDA chain, against the seasonal σ, Upper Lugano 2025.
///

**Result on Upper Lugano:** σ (Figure 5) climbs through the spring warming at the surface and peaks at the thermocline in October. Below 25 m it falls much lower than the seasonal value. Native and OpenDA pick the same σ (correlation ≥ 0.997). **It is calibrated by construction, and it could be run operationally:** Its NIS of 1 is largely what the estimator is built to produce. **Pooled RMSE again barely moves** (−26 → −27 %), but the **gain has a location:** Where the seasonal σ distrusted the observations, the correction improves (e.g. at 30 m). The surface gives back one or two percentage points.

Checked against data neither run assimilated, the 23 CTD casts of 2025:

| depth (m) | free run | seasonal σ | adaptive σ |
|---|---|---|---|
| 0–4 | 1.03 | 0.33 (−68 %) | 0.35 (−66 %) |
| 4.5–12 | 1.00 | 0.63 (−38 %) | 0.62 (−38 %) |
| 12.5–18.5 | 0.55 | 0.50 (−10 %) | 0.47 (−15 %) |
| 19–35 | 0.33 | 0.23 (−31 %) | 0.20 (−37 %) |
| 35.5–60 | 0.14 | 0.13 (−9 %) | 0.12 (−11 %) |
| below 60 | 0.19 | 0.20 (+9 %) | 0.21 (+10 %) |

*RMSE (°C) against CTD casts, OpenDA chain, same 13,043 (time, depth) pairs for all three. The native engine gives the same picture to within two points:*

The casts confirm the buoy result: adaptive σ wins at 98 of the 121 CTD levels in the top 60 m, mostly between 7 and 60 m. It is slightly worse in the top 6.5 m and in a thin band around 18–22 m, and makes no difference below the deepest sensor. The adaptive σ still hides the under-dispersion problem, only more effectively. It also has lag: after a quiet fortnight a sudden error could get over-trusted until the window catches up.

**Result across seven lakes: the method does not hold up:**

| lake | gain, seasonal σ | gain, adaptive σ | NIS seasonal (all / mixed / strat.) | NIS adaptive (all / mixed / strat.) |
|---|---|---|---|---|
| upperlugano | −26.3 % | −26.9 % | 0.55 / 0.74 / 0.51 | 0.99 / 1.00 / 0.98 |
| maggiore | −28.5 % | −25.7 % | 0.87 / 0.38 / 1.04 | 0.99 / 0.81 / 1.02 |
| greifensee | −27.1 % | −26.5 % | 0.63 / 0.55 / 0.64 | 0.92 / 0.73 / 0.97 |
| geneva | −42.5 % | −40.3 % | 1.07 / 1.54 / 0.98 | 1.00 / 1.05 / 0.98 |
| aegeri | −35.2 % | −13.6 % | 0.94 / 0.66 / 1.14 | 1.00 / 0.78 / 1.08 |
| hallwil | −49.2 % | −45.9 % | 1.00 / 0.54 / 1.46 | 0.98 / 0.87 / 1.03 |
| murten | −34.2 % | −30.8 % | 0.60 / 0.96 / 0.50 | 0.98 / 0.99 / 0.98 |

Every lake ends up with NIS near 1, as expected by construction. Upper Lugano is the only lake whose accuracy improves. The other six lose, most of them by a few points, and Ägeri loses more than half its gain.

![Ägeri: RMSE by depth with adaptive σ](../images/21_aegeri_adaptive_profile.png){ width="380" .center }
/// figure-caption
Ägeri 2025: RMSE by depth with adaptive σ.
///

**Ägeri shows the failure mode well:** Judged on calibration alone the adaptive run looks perfect. On accuracy, the seasonal σ brings the error from 0.46 to 0.30 °C, the adaptive σ only to 0.40 °C, and the loss sits in a red band (Figure 6) between about 10 and 20 m, from July to October, at the thermocline. The reason is probably a feedback loop. The estimator cannot tell whether a large innovation comes from the measurement or from the model, so it always assigns it to the observations. In late summer the model starts misplacing the thermocline, the innovations grow, and the method concludes the measurements have become unreliable: σ at the thermocline climbs from 0.2 to 0.9 °C, the filter largely ignores the observation at that depth and corrects it only through what the covariance propagates from other depths, and the error keeps growing.

**Verdict:** A self-tuning error model can look perfectly consistent while degrading accuracy. With the right safeguards it could still be a valid alternative. For example, σ could be bounded between the 0.05 °C floor and the seasonal value, which would stop the runaway seen on Ägeri while keeping the gain where the seasonal σ is too cautious. This capped version has not been tested yet. For now we see no need to make it operational. Why the seasonal σ stays operational is argued in [Method](method.md#sigma).

**Reproduce:** `adaptive_sigma`: the `adaptive_sigma` config block, native engine and OpenDA restart chain; `operational_openda-restart`: `src/assimilator/openda/restart.py`, the `openda_restart` config block.

### Fitting year { #fitting-year }

**Status:** ✅ Passes · **Lakes:** Ägeri, Hallwil, Greifensee

**Question:** The seasonal σ is independent of the filter, but not of the evaluation data: it was fitted on 2025, and every result here is scored on 2025. Does that matter? (2025 is the only year with profile observations on all seven lakes.)

**Setup:** Three lakes also have full records for 2023 and 2024 on an unchanged sensor chain. For those we refitted σ on 2023–24 and scored the result on 2025, changing nothing else.

| lake | σ fitted on 2025 | σ fitted on 2023–24 | RMSE gain, 2025 fit | RMSE gain, 2023–24 fit |
|---|---|---|---|---|
| aegeri | 0.24 | 0.27 | −35.2 % | −34.3 % |
| hallwil (mixed / strat.) | 0.41 / 0.39 | 0.33 / 0.53 | −49.2 % | −47.7 % |
| greifensee (mixed / strat.) | 0.52 / 0.94 | 0.52 / 0.91 | −27.1 % | −27.6 % |

**Result:** The fitted values do move between years, by up to a third (Hallwil's stratified season, odd case mentioned earlier). The result barely does. The gain changes by at most 1.5 points, and no depth differs by more than 0.03 °C. Hallwil's calibration even improves.

**Verdict:** Where it can be tested, **the choice of fitting year matters much less than one might think and the recipe itself**. Two caveats: the test covers three lakes and one scored year, and the forcing perturbation is still the one fitted on 2025, because the ICON reanalysis it needs does not reach back to 2024.

**Reproduce:** `master`: `notebooks/generate_sigma.py`, fitted on 2023–24.

---

## Model spread and structure { #model-spread }

The problems above all point to an under-dispersed ensemble ([Open problems](open-problems.md)). These experiments look at the model side.

### Vertical localization { #localization }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** all seven (radii); example Greifensee

**Question:** Do the failures in the deeper layers come from spurious correlations? With 20 members, a small correlation is potentially indistinguishable from sampling noise, and without a limiting mechanism the filter acts on it anyway. Besides, observations often stop well before the lake floor, and stratification complicates the profiles. Localization forces small correlations to zero instead of trusting them.

![Localization taper: correlation weight by state depth and observation depth](../images/22_localization_taper.png){ width="380" .center }
/// figure-caption
Localization taper: correlation weight by state depth and observation depth.
///

**Setup:** Radii are derived offline, per lake, from a free ensemble (perturbed forcing, no analysis) in the stratified season: the ensemble anomalies (each member minus the ensemble mean) are correlated between depths, pooled over all members and times, and the radius at a given depth is the smallest separation at which that correlation first drops below a threshold of 0.45. That is 2/√20, the level below which a correlation estimated from 20 members cannot be told apart from sampling noise (two-sided 5 %): links weaker than this are ones our 20-member filter would mostly act on by chance. The radius is then used as the support of a smooth taper (Figure 7) (weight 1 at zero separation, falling to 0 at the radius). Two matrices built from the taper modify the Kalman gain. The radii come out narrow near the surface, where the gradient is sharp, and wider with depth, where the column is more coherent. Conceptually, corrections from far-away cells shrink, less spread is removed in general, and the extra spread gives a larger gain on the links that remain. Implementation details are in [Localization implementation](#localization-implementation).

![Greifensee August profile: without localization (left) and with (right)](../images/23_localization_greifensee.png){ width="800" .center }
/// figure-caption
Greifensee, August 2025: error profile without localization (left) and with it (right).
///

**Result:** On Greifensee in August for example, localization brings the pooled error from 1.32 to 0.93 °C (Figure 8, left) instead of 0.97 °C (right), and trims the worst analysis error at 7.1 m from 1.76 to 1.57 °C. But the band between about 6 and 14 m is still worse than the free run in both cases. **The improvements, when present, remain small.**

**Verdict:** **Long-range spurious correlations don't seem to be a central problem in our lakes in the current set up.** We provide localization as an optional element rather than part of the operational configuration.

**Next step:** The scheme is measured per lake and ready to be tested and integrated where that stops being true: with more spread at depth, or in 3D, where spurious horizontal correlations are a bigger concern ([3D](#3d)).

**Reproduce:** `feat/da-combi`: `src/assimilator/localization.py`, `notebooks/localization.py`.

#### Localization implementation { #localization-implementation }

Radii are derived offline per lake from a free ensemble run in the stratified season (`notebooks/localization.py`): member anomalies are correlated between model depths, pooled over (member, time), and the radius at a depth is the smallest separation at which that correlation first drops below the threshold (0.45 = 2/√20, the sampling floor of a 20-member ensemble). The stratified radius is used all year: in a mixed column it localizes harder than needed, which discards some real information but never lets a spurious correlation through. That radius is the **support** of a Gaspari–Cohn taper: weight 1 at zero separation, falling smoothly to exactly 0 at the radius. Two matrices are built from it: a *(state, obs)* taper for `PHᵀ` and an *(obs, obs)* taper for `HPHᵀ`.

**Both sides must be tapered, not just `PHᵀ`:** The tempting shortcut is to taper only the cross-covariance, since that is the term that spreads the correction. But the gain also contains `(HPHᵀ + R)⁻¹`, and if that inverse is still built for the *full* observation set, the two halves of the gain disagree about which observations exist.

**The taper must not be a boxcar:** A binary on/off mask can produce a correlation structure that is no longer positive semi-definite: negative eigenvalues, and a covariance matrix that is mathematically invalid.

The taper accepts two arbitrary depth axes; per-depth radii are interpolated from the table and held **flat past the table's ends** rather than extrapolated; and the support of a pair is the **mean** of the two radii, which is what keeps ρ symmetric. Deleting `localization/<lake>.json` is the kill switch; the scheme falls back to a flat 5 m radius.

The intended consequence is that nothing far below the deepest observation gets updated. The deep water free-runs, and that is correct behaviour rather than a limitation: there is no information down there, so the model should be trusted rather than nudged by noise.

### Parameter sensitivity { #param-sensitivity }

**Status:** 🔬 Open · **Lakes:** Upper Lugano

**Question:** An obvious source of additional spread is the model itself. Three clear candidates, all already used in Simstrat's own calibration: `f_wind` (wind factor: wind speed is multiplied by it), `p_lw` (longwave radiation factor: longwave irradiance is multiplied by it) and `a_seiche` (seiche factor: how much wind energy is absorbed by internal waves). How much spread could they give, and where?

**Setup:** Seven continuous free runs over 2025, identical unperturbed forcing and identical warm start: control plus a low/high pair per parameter (`f_wind` ±10 %, `p_lw` ±3 %, `a_seiche` ×3 / ÷3, all inside the lake-calibrator bounds). Shown with the control subtracted, at 1, 11 and 30 m, against raw hourly observations. Light grey is the free ensemble from forcing perturbation; darker grey the assimilated one.

![Parameter sensitivity envelopes at 1, 11 and 30 m, 2025](../images/24_parameter_sensitivity.png){ width="800" .center }
/// figure-caption
Parameter sensitivity envelopes at 1, 11 and 30 m, Upper Lugano 2025, control subtracted, against raw hourly observations; light grey: free forcing ensemble, darker grey: assimilated ensemble.
///

**Result:** In Figure 9, at **1 m**, the error to explain reaches −2 °C in midsummer. The AR(1) forcing ensemble is about ±0.3 °C. The parameter envelopes reach 0.5–1.2 °C. At **11 m**, observations minus free run peaks near +3 °C in September. The forcing spread is still around ±0.3 °C, but `a_seiche` and `f_wind` open to about ±1 °C, and they open *in the stratified season*, which is exactly where more spread would be particularly helpful. At **30 m** the error is smaller, but more spread would still help.

**Verdict:** Parameter variability could be a viable way to insert uncertainty into the deep column. Two ways to use it follow: perturbing the parameters, or estimating them.

**Reproduce:** Not in the repository; easily reproducible, scripts available from the author on request.

### Parameter perturbation { #param-perturbation }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** Upper Lugano

**Question:** Parameter differences are structural, so the analysis should have a harder time collapsing them, and it moves uncertainty away from the observation error and toward the model, where it partly belongs: more model spread means the observations are trusted more. Does it help the filter?

**Setup:** Seven assimilation experiments, native EnKF (seasonal σ). The only thing that changes is which model parameter is spread across the members, and by how much. Members are spread deterministically around the calibrated value, and the control member keeps it.

| perturbation | mean spread (°C) | spread below 19 m | pooled NIS | RMSE, hourly |
|---|---|---|---|---|
| none (operational) | 0.072 | 0.033 | 0.55 | −26.3 % |
| `a_seiche_w` ×/÷ 5 | 0.072 | 0.033 | 0.55 | −26.3 % |
| `f_wind` ±10 % | 0.074 | 0.034 | 0.54 | −26.5 % |
| `a_seiche` ×/÷ 5 | 0.082 | 0.039 | 0.55 | −26.1 % |
| `p_lw` ±5 % + `a_seiche` ±80 % | 0.084 | 0.036 | 0.50 | −27.3 % |
| `p_lw` ±10 % | 0.090 | 0.035 | 0.49 | −27.4 % |
| `f_wind` ±50 % | 0.108 | 0.052 | 0.48 | −27.3 % |
| `p_lw` ±25 % | 0.116 | 0.037 | 0.44 | −28.3 % |

**Result:** **It works in the direction predicted, but it is small.** More parameter spread gives a slightly better analysis. The best case, `p_lw` ±25 %, moves RMSE from −26.3 % to −28.3 %, and it cuts the total depth-months made worse by assimilation from 41 to 25 out of 192. The corrections become bigger (median 0.013 → 0.017 °C at the surface), which is what a filter with a wider ensemble is supposed to do.

**The amplitudes in parameter perturbations that achieve this are not defensible.** `p_lw` ±25 % is an eightfold exaggeration of the ±3 % used in the sensitivity test, and the calibrated values across our seven lakes span less. The perturbations we *can* eventually justify (`f_wind` ±10 %, `a_seiche` ×/÷ 5) change essentially nothing: 0.072 → 0.074 and 0.082 °C of spread. **σ still dominates** and the imbalance grows with depth.

![Upper Lugano: ensemble spread vs actual error, daily and 3-day, with and without p_lw ±10 %](../images/25_plw_daily_vs_3day.png){ width="800" .center }
/// figure-caption
Upper Lugano 2025: ensemble spread (solid) vs actual error (dashed) and σ (grey), daily and 3-day assimilation, with and without p_lw ±10 %.
///

**Where it becomes interesting is at lower frequency:** In Figure 10, solid lines are ensemble spread, dashed the actual error it should cover, grey σ. With `p_lw` ±10 %, the spread grows by about a quarter at daily frequency, and by more than a third when assimilating every 3 days, because the parameter differences have longer to act between analyses. At 3 days this is enough to bring the accuracy back to almost exactly the daily result, while assimilating a third of the profiles.

**Verdict:** Not adopted at defensible amplitudes; only worth it at lower assimilation frequencies.

**Next step:** The 3-day result is a quick test only: the deepest measurement actually got worse, so the way the additional spread is generated needs more care, but it is a promising direction.

**Reproduce:** Not in the repository; scripts available from the author on request.

### Joint state–parameter estimation { #joint }

**Status:** 🔬 Open · **Lakes:** Upper Lugano

**Question:** Instead of fixing the parameter spread, can the filter estimate the parameters together with the temperature? There is precedent in the literature (e.g. the FLARE framework of [Thomas et al. (2020)](../literature-review.md#thomas-2020)).

**Setup:** The filter updates `f_wind`, `p_lw` and `a_seiche` at every analysis. At the start each of the 20 members draws its own set of values around the calibrated ones: `p_lw` with a 5 % spread in logit space, bounded to 0.8–1.2; `a_seiche` and `f_wind` in log space, with a spread of ×1.65 and about 10 %. The transforms keep the parameters positive and within bounds, and the draws are stratified and shuffled so the three parameters start uncorrelated. The transformed parameters are appended to each member's temperature profile as an extended state. Since they are not observed, they change only through their ensemble covariance with the observed temperatures, and the temperature update itself is unchanged. Between analyses they stay fixed. Each update narrows their spread; a floor holds the spread from collapsing.

![Joint state–parameter estimation: the three parameters through 2025, coloured by month](../images/26_joint_parameter_cloud.png){ width="800" .center }
/// figure-caption
Joint state–parameter estimation, Upper Lugano 2025: the members' values of the three parameters, coloured by month; the star marks the calibrated values.
///

**Result:** The cloud in Figure 11 shows the members' values through 2025, coloured by month; the star marks the calibrated values. **They do not converge.** They keep moving with the seasons, with much less wind and much more seiche mixing in summer, and the longwave factor pinned to its lower bound. The filter is using the parameters to absorb the seasonal bias. The accuracy ends up the same as without updating them, but reached in a different way.

**Verdict:** That opens real questions. Can a parameter be updated daily? Is 20 members enough for a joint state? Do the parameters stay physically defensible (broad bounds help, but probably something tighter is needed)? And there is an underlying attribution problem: we change the model's physics and its state at the same time on limited evidence, so the parameters can absorb errors that aren't theirs, and they trade off against each other.

**Next step:** A single run both corrects the temperatures and yields a distribution of the parameters, which could be a cheap first estimate to compare against a traditional calibration or to analyse potential systematic model failures. Coming back to the spread question, the next step would be adding justified random variability to the model state itself: structural uncertainty through state perturbation (see [Uncertainty band](#uncertainty-band)).

**Reproduce:** Not in the repository; scripts available from the author on request.

### Deep mixing { #deep-mixing }

**Status:** 🔬 Open · **Lakes:** Lake Lugano, 1987–2025

**Question:** Can assimilation improve the representation of deep mixing in Simstrat?

**Setup:** The buoy stops at 40 m, but for Lake Lugano there are almost 40 years of CTD casts, one every two weeks, down to 280 m, and we started by assimilating those. The profiles were thinned to a coarser grid as described in [Method](method.md#observations); the seiche filtering is not possible on fortnightly casts.

![Lake Lugano, 1987–2025: model − observation, free run vs CTD assimilation](../images/27_deep_mixing_ctd.png){ width="880" .center }
/// figure-caption
Lake Lugano, 1987–2025: model − observation, free run vs CTD assimilation; arrows mark the winters the lake mixed below 100 m.
///

**Result:** The free run starts too cold in the deep water and slowly builds up a warm bias. It mixes all the way below 100 m in 10 winters, while the lake did so in only two (the arrows in Figure 12). With assimilation the deep temperatures are corrected: **RMSE between 30 and 100 m almost halves, 0.29 → 0.16 °C**, and at 1 m it drops from 0.88 to 0.79 °C.

![Independent check against the 2025 Upper Lugano buoy](../images/28_deep_mixing_buoy_check.png){ width="380" .center }
/// figure-caption
Independent check of the CTD-assimilated run against the 2025 Upper Lugano buoy.
///

Against the 2025 buoy, which this run never saw, the error still drops from 0.83 to 0.72 °C. **But look at the dotted lines in Figure 13: deep mixing becomes more frequent, 15 winters instead of 10.**

**Verdict:** For now, assimilating temperature fixes the temperature but tends to make the mixing worse. Correcting the state may change when and how deep the model overturns.

**A later finding may explain part of this:** the forcing perturbation makes every member windier than the control ([Centred forcing perturbations](#centred-forcing)), and more wind brings more mixing. The Lugano run above was made before this was found, with uncentred forcing. On Geneva, with centred forcing, the free run and the assimilated run mix to the bottom in the same winters (both still too often), but Geneva has not been run without centring, so the comparison is indirect. The Lugano run has not yet been repeated with centred forcing.

**Reproduce:** Not in the repository; scripts available from the author on request.

### The same layout in 3D { #3d }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** Lake Lugano (Delft3D-FLOW), one-month test window

**Question:** Does the ensemble DA layout transfer to a 3D model, so that it can sit beside Simstrat in the module as an optional engine?

**Setup:** Delft3D-FLOW on Lake Lugano, driven the same way: OpenDA treats the model as a black box, writes a restart file, launches the model in a container, reads the output, adjusts the restart, and repeats, so the simulation advances window by window. The model is expensive, so these experiments use 7 members instead of 20. A big new thing in 3D is that the covariance also spreads a single station's correction *horizontally* to nearby cells, so one buoy can inform a whole region. That is attractive, because we have few stations, and risky at the same time: with seven members, spurious horizontal correlations become a real concern, more so than in 1D.

![Satellite surface temperature, Lake Lugano, DA improvement](../images/29_lugano_satellite.png){ width="800" .center }
/// figure-caption
Delft3D-FLOW, Lake Lugano: satellite surface temperature (left) and cells where the assimilated run is closer to it than the operational run (right, green).
///

**Result:** Surface-only assimilation of the five stations around Lake Lugano gives promising first results: against a satellite surface-temperature snapshot (Figure 14, left), the assimilated run is closer than the operational Alplakes run in 94 % of the cells (right, green).

**Next step:** The first results point clearly at the need for a vertical localization mechanism, or for deep measurements where they exist. Beyond that, it opens up possibilities to explore also assimilating the satellite surface temperature itself, providing the whole lake surface, but at lower resolution.

**Reproduce:** Not in the repository; scripts available from the author on request.

---

## Ongoing { #ongoing }

Work started after the results above. Numbers are preliminary, and the runs are not part of the operational setup. Scripts, configurations and notes are in `local/` on `feat/da-combi`; each block names its folder.

### Does assimilation disturb the model's natural variability? { #variability }

**Status:** ✅ Passes · **Lakes:** Upper Lugano, 2025

**Question:** A daily correction is a jump in the model state. Does it leave the model's own oscillations intact, or does it add artificial ones?

**Setup:** The operational setup (`simplified_final`: 20 members, one analysis a day at 06 UTC) against the free run, over 2025. Power spectra of the hourly temperature at every depth (one FFT over the year) for the free run, the ensemble mean with assimilation, and the buoy observations (the raw 10-minute record averaged to hourly, not filtered, so that the variability being compared is still in it); plus the mean absolute hourly temperature change by hour of day.

![Upper Lugano: spectral energy by depth and period, without and with DA, and at 0.5, 11 and 40 m](../images/30_da_spectra.png){ width="880" .center }
/// figure-caption
Upper Lugano 2025, `simplified_final`: spectral energy of hourly temperature by depth and period, without and with DA and their ratio (top); spectra at 0.5, 11 and 40 m with the observations (middle); spectra of the model − observation error (bottom).
///

**Result so far:** The assimilated run keeps generally the spectral shape of the free run at every depth (Figure 15), so the model's natural behaviour is preserved. At short periods (days to hours) and at 11 and 40 m, the energy with assimilation lies between the free run and the observations: it moves towards the real lake, which has far more short-period energy than the 1D model. 

![Upper Lugano: mean absolute hourly temperature change by hour of day, without and with DA](../images/31_da_hourly_change.png){ width="880" .center }
/// figure-caption
Upper Lugano 2025, `simplified_final`: mean absolute hourly temperature change by hour of day at 0.5, 11 and 40 m, without and with DA; the update falls in the 06 → 07 UTC step.
///

Each update is a single step (Figure 16, the 06 → 07 UTC hour), with no oscillation after it; in every other hour the assimilated run changes like the free run.

**Next step:** Results support the claim in [Overview](overview.md) that the shocks are not generating artificial oscillations and that the scheme is safe to run unattended.

**Reproduce:** Not in the repository; scripts available from the author on request.

### Oxygen assimilation { #to2-profiler }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** Ägeri (Nov 2022 – Jun 2026), Murten (Jun 2022 – Jun 2026)

**Question:** Can the same filter correct dissolved oxygen as well as temperature, and what does the vertical coverage of the oxygen data change?

**Setup:** Temperature and oxygen are assimilated together, with their updates decoupled (temperature observations correct only temperature, oxygen only oxygen), a seasonal σ for oxygen, and a small white noise (standard deviation 2 mmol/m³) on every member's oxygen after each analysis, to widen an oxygen spread that was far below its innovations. Here the profilers are assimilated and the CTD casts are only used for scoring; the next two experiments change which data go in:

| Data | Ägeri | Murten |
|---|---|---|
| Profiler, temperature | 23 depths, 2–79 m | 35 depths, 0.5–43 m |
| Profiler, oxygen | 23 depths, 2–79 m | 3 depths: 25, 40 and 41 m |
| Profiler period, as assimilated | Nov 2022 – Oct 2026, seiche-filtered, one profile a day at 23 UTC | Jun 2022 – Jul 2026 (oxygen from Dec 2022), seiche-filtered, one profile a day at 04 UTC |
| CTD casts (T and O₂) | 14 casts, about quarterly, Dec 2022 – Mar 2026, 0.5–78.5 m | 42 casts, about monthly, Jul 2022 – Dec 2025, 0.5–42.5 m |
| [Oxygen assimilation](#to2-profiler) | profiler in, casts for scoring | profiler in, casts for scoring |
| [Depth thinning](#depth-thinning) | 5 of the 23 profiler depths in, all profiler depths and casts for scoring | – |
| [CTD interpolation](#ctd-interpolation) | casts in, profiler for scoring | casts in, profiler for scoring |

Ägeri's profiler is high-resolution both in time and in depth; Murten's is high-resolution in time, but measures oxygen only in the deep water.

![Ägeri: RMSE profiles of T and O₂, free run vs EnKF, against the profiler and the CTD casts](../images/32_aegeri_to2_profiler.png){ width="700" .center }
/// figure-caption
Ägeri, Nov 2022 – Jun 2026, profiler assimilated: RMSE profiles of T and O₂, free run vs EnKF, against the profiler (top) and the CTD casts (bottom).
///

**Result so far.**

- **Ägeri: large improvement (Figure 17):** O₂ error against the profiler drops by about 45 % (46 → 25 mmol/m³), and is spread more evenly over the column, with large gains at 10–20 m and again in the deep water, but not directly at the lake bed. Temperature 0.52 → 0.32 °C.
- **Murten: smaller improvement (Figure 18):** O₂ drops by about 18 % against the CTD casts (69 → 57 mmol/m³), mostly at the deeper levels near the three oxygen sensors. Temperature 0.75 → 0.36 °C against the profiler.

![Murten: RMSE profiles of T and O₂, free run vs EnKF](../images/34_murten_to2_profiler.png){ width="700" .center }
/// figure-caption
Murten, Jun 2022 – Jun 2026, profiler assimilated: RMSE profiles of T and O₂, free run vs EnKF, against the profiler (top) and the CTD casts (bottom).
///

![Ägeri, 31 October 2024: O₂ profile, observations (red), free run (dashed) and EnKF (blue)](../images/33_aegeri_o2_bottom_profile.png){ width="320" .center }
/// figure-caption
Ägeri, 31 October 2024: O₂ profile, observations (red), free run (dashed) and EnKF (blue).
///

The filter struggles right above the lake bed. In the example of Figure 19 the assimilated profile (blue) fits the observations (red) as well as it can, but keeps the wrong shape at the bottom. Discussed with Fabian: this is a structural model error, the model cannot simulate the oxygen depletion at the bottom boundary.

**Verdict so far:** Vertical coverage matters, and the bottom boundary has to be fixed in the model, not by assimilation.

**Reproduce:** Not in the repository; scripts available from the author on request.

### Observation depth thinning { #depth-thinning }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** Ägeri

**Question:** How many depths do we need for a useful improvement? If the filter can interpolate vertically, fewer sensors could be enough in the field.

**Setup:** Same Ägeri profiler data as above, temperature and oxygen, but keep only 5 of the 23 depths (2, 10, 20, 40, 70 m; a second layout 2, 7, 15, 45, 70 m) and score against all profiler depths and hours, with the 23-depth run as reference.

![Ägeri: 5 assimilated depths (blue) vs 23 (dashed) vs free run](../images/35_aegeri_to2_thinned.png){ width="700" .center }
/// figure-caption
Ägeri: RMSE profiles with 5 assimilated depths (blue) vs 23 (dashed) vs free run; red fill where 5 depths are worse than the free run.
///

**Result so far:** With 5 depths, about 35–40 % of the temperature gain (0.52 → 0.45 °C, against 0.32 with 23 depths) and about 70 % of the oxygen gain (46 → 31 mmol/m³, against 25) remain. These are scored on the profiler, of which the 5-depth run assimilates a few depths; against the CTD casts, which neither run assimilates, the shares are almost the same (about 40 % for temperature, 0.43 → 0.35 °C against 0.24; about 70 % for oxygen, 47 → 34 mmol/m³ against 28). The deep levels above the bottom keep good gains with just a few measurements; the rest to a lower extent. Each layout opens possible new gaps between its sensors (red areas in Figure 20, where the filter is worse than the free run) and the oxygen and temperature structure of each lake needs to carefully be taken into account and requires more study.

**Reproduce:** Not in the repository; scripts available from the author on request.

### CTD casts as a temporal interpolator { #ctd-interpolation }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** Ägeri, Murten

**Question:** Suggested by Fabian, for practical purposes: with only the monthly or quarterly CTD casts assimilated, does the run give realistic profiles *between* the casts?

**Setup:** Assimilate the CTD casts (T and O₂) instead of the profiler: 14 casts on Ägeri, 42 on Murten (see the [data table](#to2-profiler)); the high-resolution profiler, never seen by the filter, becomes an independent check of the interpolation in time. Note the rows in Figures 21 and 22 are swapped with respect to the profiler runs: top row = independent profiler, bottom row = the assimilated casts, one hour after the analysis.

![Ägeri: CTD assimilation, scored against the independent profiler (top) and the casts (bottom)](../images/36_aegeri_to2_ctd.png){ width="700" .center }
/// figure-caption
Ägeri, CTD casts assimilated: RMSE profiles against the independent profiler (top) and the casts, one hour after the analysis (bottom).
///

**Result so far:**

- **Ägeri: it works essentially only in the deep water:** Between casts, O₂ at 50–80 m stays closer to the profiler than the free run; above about 40 m it is no better than the free run. Overall about −10 % (O₂ 46 → 41 mmol/m³, T 0.52 → 0.48 °C), not evenly distributed.
- **Murten: similar (Figure 22).** Gains where observations are available (O₂ 62 → 52 mmol/m³, T 0.75 → 0.60 °C). Here we would have expected oxygen gains up to about 15 m.

![Murten: CTD assimilation, scored against the independent profiler (top) and the casts (bottom)](../images/37_murten_to2_ctd.png){ width="700" .center }
/// figure-caption
Murten, CTD casts assimilated: RMSE profiles against the independent profiler (top) and the casts, one hour after the analysis (bottom).
///

These runs need the forcing ensemble centred on the original forcing, because the observations are weeks apart ([Centred forcing perturbations](#centred-forcing)).

**Verdict so far:** Assimilation is a promising temporal interpolator for the deep layers, but not for the faster-varying upper layers with monthly or sub-monthly sampling.

**Reproduce:** Not in the repository; scripts available from the author on request.

### Centred forcing perturbations { #centred-forcing }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** Ägeri, Murten, Geneva

**Question:** Does the forcing perturbation shift the ensemble mean away from the original forcing, and does that matter?

**Setup:** The AR(1) noise is added to the wind components U and V with zero mean. But the wind *speed* is the length of the vector, so noise in either direction makes it larger on average. With the option `center_perturbations`, each forcing row is rescaled so that the ensemble-mean wind speed, solar radiation and air temperature equal the unperturbed forcing.

**Result so far:** Without centring, every member is windier than the control: the ensemble-mean wind speed is 63 % above it on Ägeri (2.52 against 1.55 m/s), 57 % on Geneva and 31 % on Murten. With centring, the two agree to within about 2 %. With daily assimilation the filter corrects the resulting bias every day, but when the observations are weeks apart (CTD casts) it accumulates between casts: more wind, more mixing, a deeper overturn. That may explain part of the more frequent deep mixing seen on Lake Lugano ([Deep mixing](#deep-mixing)).

**Next step:** The operational setup (`simplified_final`) runs without centring, so check its effect there too, and repeat the Lugano deep-mixing run with centred forcing.

**Reproduce:** Branch `feat/center-perturbations`, option `center_perturbations`.

### Long-term CTD assimilation in Geneva { #geneva-ctd }

**Status:** 🔬 Open · **Lakes:** Geneva (LéXPLORE casts, Nov 2019 – Jun 2026)

**Question.** Can six and a half years of CTD casts correct temperature and oxygen together, and does the result hold against the high-resolution profiler?

**Result so far.** Annual σ: temperature RMSE against the profiler 1.02 → 0.62 °C; oxygen at the casts 105 → 38 mmol/m³. The free run's deep oxygen is about 120 mmol/m³ too low below 40 m. The ensemble spread is tiny compared with σ (0.026 against 0.297 °C).

**Air-temperature perturbation:** Perturbing air temperature as well (AR(1), standard deviation 1.67 °C) changes essentially nothing: in winter the spread stays around 0.03 °C, because the deep mixed layer damps the noise.

**Winter-specific σ:** Same-day double casts scatter by only about 0.10 °C, so we tested that value as σ from November to April (0.297 °C the rest of the year). It is too small: the winter NIS/n (pooled, December–March) goes from 0.73 to 4.95, so the filter now expects far less misfit than it sees. The double-cast scatter measures the casts, not what the model cannot represent; the innovations point to a winter σ of about 0.25 °C, close to the annual value. Accuracy is essentially unchanged: temperature against the profiler stays at 0.62 °C, oxygen at the casts moves from 38 to 40 mmol/m³, and the winter mixed-layer depth error stays the same (about 22 m, about 15 m too deep). The stronger updates halve the already tiny spread (0.028 → 0.014 °C), and the model drifts back between casts, so the misfit before each update is unchanged.

Most of that 0.25 °C is model uncertainty that σ has to carry because the ensemble does not. A process noise on the temperature profile, as in the [forecast uncertainty band](#uncertainty-band), would move it into the ensemble: the spread would grow, σ could shrink towards the cast scatter, and each cast would correct more. In a near-isothermal winter column the noise must not create inversions, though, so it would need the constraint to physical profiles discussed there.

**Year by year:** The casts mix to the station bottom (about 100 m) only in 2020, 2021 and 2026. The model mixes to the bottom in almost every winter, with or without assimilation: in 2022 and 2024 the lake stopped at about 65 m and 45 m, but both runs reach the bottom. Unlike on Lake Lugano, assimilation does not make the deep mixing more frequent; over the whole 309 m column it even mixes less (the free run overturns completely in 2021 and 2022, the EnKF only in 2022). For oxygen, the filter needs about a year to lift the deep water out of the free run's low bias (NIS/n about 20 at the start, about 1 from 2022 to 2025).

**Verdict so far:** Neither more forcing noise nor a smaller winter σ helps. Together they point to a systematic winter model error, mixing too deep with a surface that is too cold, the same in every member, rather than a filter-tuning problem ([Open problems](open-problems.md)).

**Reproduce:** Not in the repository; scripts available from the author on request.

### Forecast uncertainty band { #uncertainty-band }

**Status:** (✅) Positive, potentially to be adopted · **Lakes:** Upper Lugano

**Question:** The ensemble spread is far smaller than the error, so a band built from it would be far too narrow. Can a simple process noise on the temperature profile give an honest band for the Alplakes forecasts?

**Setup:** Following the idea of FLARE ([Thomas et al., 2020](../literature-review.md#thomas-2020)), where process uncertainty dominates the forecast variance at depth, but simplified: forcing taken as given, parameters fixed. Every day, a noise drawn from the model's typical daily error is added to each member's profile at the analysis hour. Its covariance comes from the free run alone: an AR(1) fit to the free-run residuals at 06 UTC gives the daily "new error" per depth, with one yearly depth correlation and a monthly amplitude. Since the model error now has its own term, σ is lowered accordingly. The noise is fitted on 2025 and the run is scored on 2026, out of sample.

![Upper Lugano, May–July 2026, 1 m: observations, free run, current DA and DA + process noise with their 90 % bands](../images/38_uncertainty_band.png){ width="880" .center }
/// figure-caption
Upper Lugano, 1 m, May–July 2026 (out of sample): observations, free run, current DA and DA + process noise with their 90 % bands.
///

**Result so far:** The mechanism works, with good results and a more honest spread, both in sample and out of sample.

| | In sample: 2025 | Out of sample: March–July 2026 |
|---|---|---|
| noise | fitted on 2025, seasonal | fitted on 2025, monthly |
| RMSE, free run | 0.67 °C | 0.91 °C |
| RMSE, current setup → with noise | 0.36 → 0.28 °C | 0.43 → 0.36 °C |
| bias, current setup → with noise | 0.10 → 0.01 °C | 0.06 → 0.03 °C |
| CRPS, current setup → with noise | 0.20 → 0.13 °C | 0.25 → 0.18 °C |
| spread / RMSE | 0.26 → 1.00 | 0.25 → 0.97 |
| observations inside the 90 % band (ensemble) | 27 → 83 % | 24 → 80 % |
| same, band including σ | 95 → 97 % | 92 → 97 % |
| NIS/n | 0.64 / 0.47 → 0.52 / 0.26 (mixed / stratified) | 0.77 → 0.50 |

*Upper Lugano, scored against the hourly filtered observations. The two columns use slightly different noise: the 2025 run was the first, seasonal version; the monthly amplitude was introduced for the 2026 test, after the seasonal one was found too small in spring (April spread 0.21 against an error of 0.39 °C).*

In sample, the noise improves every depth from 0.5 to 30 m. Out of sample, the gain holds at 0.5–15 m and is level at 17–40 m, and spread ≈ RMSE in every month. At 1 m from May to July, 78 % of the observations fall inside the band against 35 % (Figure 23).
- **As a forecast, 1–7 days:** a reforecast with an analysis only every 7th day, the ensemble running free with the daily noise in between (about 30 forecasts per lead, January–July 2026). Pooled over depths, the band stays roughly honest at every lead: 80–86 % coverage of the 90 % band, spread/RMSE 0.87–1.21, and the forecast beats the free run at every lead (0.54 against 0.81 °C on day 7). By depth it is too narrow at 0.5–3 m at every lead (61–84 %), and too wide at 7–21 m from about day 4, where the noise accumulates faster than the real error grows. In May–July at 15–25 m the forecast is also worse than the free run at most leads (e.g. 0.65 against 0.45 °C at 15 m on day 7), while the daily run beats the free run there. The error is mostly random, not a bias that grows with lead, so it most likely comes from the weaker weekly start states and the large summer noise at the thermocline rather than from the uncentred forcing, though a centred run has not been tried.
- **Stability:** the noise creates new unstable cells only in February–March below about 15 m (5–7 % of the cells right after the update in March), with a burst of mixing for 6–12 h; they are mixed out by the next morning. From April to July the stratification absorbs it, and near the surface the daily convection dominates anyway.

**Open:**

- Parameter uncertainty is not included.
- σ is still too large once the noise is added (NIS ≈ 0.4–0.5): it needs a more consistent definition.
- The noise is zero-mean, but the analysis also corrects a systematic tendency of up to about 0.15 °C per day near the surface (warming in spring, cooling in summer). A forecast needs a correction for it, or a stated bias.
- Avoid the spurious mixing: either taper the noise below 15 m in February–March, or, more generally, constrain each perturbation to physical profiles, so that it can change the vertical gradient but never invert it.
- The forecast test is only an approximation of the real use. It is driven by the observed weather instead of a weather forecast, which makes it optimistic near the surface. And since the same ensemble cannot both assimilate every day and run free for 7 days, the test assimilated only every 7th day, so each forecast starts from a less accurate state than the operational daily analysis would give, which makes it pessimistic. A faithful test would start each 7-day forecast from a copy of the daily-assimilated ensemble, while the original keeps assimilating. The module cannot do this yet: it does not save the analysis states.

**Reproduce:** Will be added as a branch.
