# Calibration of the EnKF Observation-Error Model

This page documents how the native EnKF is calibrated for a single lake, why each component exists,
how its values were determined, and what a controlled experiment on **Upper Lake Lugano (2025)**
measured for each one. The end of the page gives the final configuration.

The worked example is `upperlugano`, a 25 km basin of mean depth 171 m with a 16-depth thermistor
chain reaching 40 m, assimilated over the full calendar year 2025 with 20 members.

---

## 1. What is being calibrated

The EnKF analysis is
$$
\mathbf{x}^a = \mathbf{x}^f + \mathbf{K}\bigl(\mathbf{y} - \mathbf{H}\mathbf{x}^f\bigr),
\qquad
\mathbf{K} = \mathbf{P}\mathbf{H}^{\!\top}\bigl(\mathbf{H}\mathbf{P}\mathbf{H}^{\!\top} + \mathbf{R}\bigr)^{-1}.
$$

Everything on this page moves one of three things: the observations $\mathbf{y}$ themselves, the
observation-error covariance $\mathbf{R}$, or which entries of $\mathbf{P}\mathbf{H}^{\!\top}$ are
allowed to be non-zero. The ensemble spread $\mathbf{P}$ is set by the forcing perturbations and is
*not* tuned here.

The diagonal of $\mathbf{R}$ is built per observation by `src/assimilator/sigma_rep.py`:

$$
\sigma_i^2 \;=\; \sigma_\text{common}^2 \;+\; \frac{\bigl(k(z_i)\,\sigma_\text{rep}(z_i, m_i)\bigr)^2}{N_i}
$$

with $z_i$ the depth, $m_i$ the month, $N_i$ the number of stations contributing (1 on every lake
today), $k$ the tunable `sigma_rep_scale`, and $\sigma_\text{rep}$ a table fitted from data. The
three terms are deliberately separated: $\sigma_\text{common}$ is instrument-and-slow error with no
depth or season structure, $\sigma_\text{rep}$ carries the measured depth and season shape, and $k$
scales that shape without redefining it.

The single scalar that says whether $\mathbf{R}$ is right is the **normalized innovation squared**,

$$
\text{NIS} \;=\; \frac{\langle d^2 \rangle}{\langle \sigma_\text{spread}^2 \rangle + \langle \sigma^2 \rangle},
\qquad d = y - H\mathbf{x}^f .
$$

NIS is reported **pooled**: the ratio of mean squared innovation to mean total variance, not the
mean of per-cell ratios. The two differ substantially, because the per-cell mean gives a depth
with 0.09 °C of error the same weight as the thermocline with 1.2 °C. Both are printed by
`local/plotting/plot_multilake_nis.py`; the pooled one is the number that inverts to a scale.

NIS = 1 means the filter's stated uncertainty matches its actual errors. NIS > 1 is
**over-confident** (the filter will over-correct); NIS < 1 is **over-damped**.

!!! danger "NIS constrains only the product"
    $\sigma_\text{rep}$ and $k$ enter only as $k\sigma_\text{rep}$, so NIS alone cannot separate
    them. This is why the table is *fitted from data* and only the scale is *tuned from a run* —
    if both were tuned the problem would be unidentifiable.

---

## 2. The components

### 2.1 Internal-wave observation filter

**How it works:** Simstrat is one-dimensional and cannot represent basin-scale seiches, so a
thermistor at a fixed depth records isotherm displacements the model has no way to reproduce.
Assimilating them injects an increment for a signal the model will immediately lose. A causal
trailing box filter of width one fundamental internal-seiche period is applied to each series below
`THERMO_DEPTH_MIN`:

$$
W(z) = \begin{cases} W_\text{seiche} & z \ge \texttt{THERMO\_DEPTH\_MIN} \\ 0 & z < \texttt{THERMO\_DEPTH\_MIN}\end{cases}
$$

Depths above the cut are passed through **untouched** — the surface layer's sub-daily signal is
diurnal heating, which the model does resolve.

**How the values were determined.** $W_\text{seiche}$ comes from Merian's two-layer formula
$T = 2L/\sqrt{g' h_\text{eff}}$, $h_\text{eff} = h_1h_2/(h_1+h_2)$, evaluated at peak stratification
by `notebooks/calibrate_filter.py`. `THERMO_DEPTH_MIN = 4.0` m on every lake.

**Values in use** (`filter/<lake>.json`, derived 2026-08-06):

| lake | $L$ (km) | mean depth (m) | $h_1$ (m) | $h_2$ (m) | $T_1/T_2$ (°C) | $g'$ (m s⁻²) | $h_\text{eff}$ (m) | **$W_\text{seiche}$ (h)** | removed (°C) | lag (°C) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| geneva | 72.3 | 152.7 | 9.5 | 143.2 | 21.5 / 7.5 | 0.01908 | 8.91 | **97.4** | 1.213 | 0.800 |
| maggiore | 54.0 | 177.0 | 8.0 | 169.0 | 21.9 / 8.5 | 0.01960 | 7.64 | **77.5** | 0.983 | 0.621 |
| upperlugano | 25.0 | 171.0 | 10.0 | 161.0 | 24.8 / 7.0 | 0.02758 | 9.42 | **27.3** | 0.908 | 0.228 |
| greifensee | 6.4 | 17.7 | 5.7 | 12.0 | 21.0 / 7.1 | 0.01825 | 3.86 | **13.4** | 0.712 | 0.124 |
| murten | 8.2 | 23.2 | 8.0 | 15.2 | 22.9 / 7.1 | 0.02271 | 5.24 | **13.2** | 0.454 | 0.071 |
| hallwil | 8.4 | 28.6 | 8.25 | 20.35 | 23.4 / 6.3 | 0.02408 | 5.87 | **12.4** | 0.479 | 0.066 |
| aegeri | 6.9 | 49.0 | 7.3 | 41.7 | 21.2 / 5.2 | 0.01912 | 6.21 | **11.1** | 0.577 | 0.083 |

$g'$ spans a factor of 1.5 across the set and $L$ a factor of 11, so the window is set by basin
length. The last two columns are the filter's measured trade at the depth where it is worst:
`lag/removed` is 0.14–0.25 on the five short-window lakes and **0.63–0.66 on maggiore and geneva**,
where a 77–97 h trailing box also smooths signal the model can follow.

!!! note "Why the shortest period"
    The period swings by two orders of magnitude over the year (geneva 92.7 h in Aug, 4691 h in
    Feb) because $g' \to 0$ as the lake destratifies. The winter value is the formula degenerating,
    not a seiche; taking the peak-stratification value gives the narrowest box that still covers a
    full oscillation when one exists.

**Config.** `filter/<lake>.json`. **Result.** Worth ~15 points of skill — see §5.

### 2.2 Assimilation hour and thinning

**How it works.** The chain samples hourly; the filter assimilates once per day. Which hour is
chosen does not change so much how much information enters — it changes the **bias** injected, because the
free run's diurnal error is not flat over the day.

**How determined.** `local/hour_decision/best_hour.py` scores every sampled hour by the free-run
bias in the surface band (0 m to `THERMO_DEPTH_MIN`) and picks the balanced low-bias hour, over
candidates restricted to the night window **00–06 UTC** so the analysis lands before the day it has
to predict and every lake updates at a comparable point in the diurnal cycle.

The hour is a function of the band, so this must be re-run whenever `THERMO_DEPTH_MIN` changes.

**Values in use** (`local/best_hour.json`, 2026-08-11). "bias" is the free-run deterministic
surface bias at the chosen hour over May–Oct; "unconstrained" is what the data alone would pick
without the night window.

| lake | candidate hours | low-bias hours | **hour used** | bias (°C) | worst hour | bias there | unconstrained | assimilated file |
|---|---|---|:--:|--:|:--:|--:|:--:|---|
| upperlugano | 0–6, 23 | 6 | **06** | +0.040 | 13 | −0.210 | 17 | `temperature_filtered_h06_1d.csv` |
| maggiore | 0–6, 23 | 1–6 | **01** | +0.074 | 12 | −0.301 | 1 | `temperature_filtered_h01_1d.csv` |
| geneva | 0–6, 23 | 2–6 | **02** | +0.073 | 22 | +0.176 | 2 | `temperature_filtered_h02_1d.csv` |
| murten | 0–6, 23 | all | **04** | +0.019 | 11 | −0.108 | 17 | `temperature_filtered_h04_1d.csv` |
| aegeri | 2, 5, 23 | 2, 5, 23 | **23** | +0.025 | 8 | −0.030 | 20 | `temperature_filtered_h23_1d.csv` |
| hallwil | 0–6, 23 | 0–3, 23 | **23** | −0.002 | 17 | +0.107 | 21 | `temperature_filtered_h23_1d.csv` |
| greifensee | 1, 4 | 1 | **00–06 window** | −0.009 | 19 | +0.257 | 22 | `temperature_filtered_w0006_1d.csv` |

Two lakes do not sample every hour — **aegeri reports 9 hours a day and greifensee 17, and neither
includes 12:00**, so an assumed noon would have been interpolated to an instant never measured.
Greifensee's sampling phase drifts, so it is thinned with `--hour-window 0-6` instead of a single
hour, which is why its file is named `w0006`.

The night window costs something: on four lakes the unconstrained optimum is an afternoon hour
(17, 20, 21, 22). It is rejected because that is the second zero crossing of the diurnal bias —
equally good on bias, but with far less correctable error available.

### 2.3 The $\sigma_\text{rep}$ table

**How it works.** Representativeness error is the part of the observation the model *cannot*
represent even when it is behaving correctly. It is estimated as the sub-block variability the
observations have and the free run does not:

$$
\sigma_\text{rep}(z, s) \;=\; \sqrt{\;\overline{\operatorname{var}_\text{block}(\text{obs})} \;-\; \overline{\operatorname{var}_\text{block}(\text{free run})}\;}
$$

per depth and season. Taking the variance *within* a block means the free run's slow drift cannot
leak into the estimate. It is a **lower bound** by construction: nothing slower than the block
length is captured, and $\sigma_\text{common}$ plus the scale $k$ carry that remainder.

**Block length.** `max(24 h, W_\text{seiche})`, from `sigma_rep_from_obs.window_for_lake`. A 24 h
block on a lake whose seiche is longer sees only part of one oscillation and systematically
under-estimates the term — measured at 18 % of the seiche variance on geneva (97.4 h) and 28 % on
maggiore (77.5 h), versus 98–100 % everywhere else. For upperlugano the block is **27.3 h**.

**Fit against the series actually assimilated.** A table fitted on raw observations and applied to
filtered ones double-counts variability the filter already removed.

**Depth smoothing.** The written table is smoothed over depth (centred neighbour blend 0.5, 2
passes, endpoints fixed) so a peak keeps its measured depth but isolated sensor artefacts do not
survive. Justified in §5.

**Command.**
```bash
python notebooks/sigma_rep_from_obs.py args/run_enkf.json --lake upperlugano \
    --obs-file observations/upperlugano/temperature_filtered.csv \
    --smooth 0.5 --smooth-passes 2 --suffix _smooth
```
**Creates.** `observations/<lake>/sigma_rep_filtered_smooth.json`, and without `--smooth`,
`sigma_rep_filtered_nosmooth.json`.

!!! warning "The final runs use the **unsmoothed** table"
    `A_final.json` and `C_final.json` both point at `sigma_rep_filtered_nosmooth.json`. The tables
    below and the scales of §2.5 are therefore the unsmoothed ones. The smoothing argument of §5
    (C vs Cns) was made on the upperlugano-only matrix, where the thinned grid is 2 m and no
    finer-resolution fit exists to check it against.

    **Aegeri and greifensee settle it.** Their records are dense enough in depth to fit σ_rep at
    native resolution — 0.2 m over 239 depths and 0.1 m over 160 — which gives a reference profile
    the thinned tables can be scored against (`sigma_rep_filtered_fullres.json`, fitted
    2026-08-12). Deviation from that reference at the assimilated depths:

    | | native rms | **unsmoothed** rms dev | max | **smoothed** rms dev | max |
    |---|--:|--:|--:|--:|--:|
    | aegeri, stratified | 0.133 | **0.011** | 0.052 | 0.015 | 0.049 |
    | aegeri, mixed | 0.084 | 0.008 | 0.037 | **0.005** | 0.017 |
    | greifensee, stratified | 0.265 | **0.010** | 0.040 | 0.028 | 0.071 |
    | greifensee, mixed | 0.152 | **0.005** | 0.019 | 0.009 | 0.021 |

    The unsmoothed table is closer in three cases of four, decisively on greifensee stratified —
    where the smoother's worst cell is **3 m: native 0.279, unsmoothed 0.282, smoothed 0.208**, a
    26 % under-estimate of R at a depth the raw fit had right to three decimals. The reason is the
    one §3 already warns about: these grids are 1 m and 2 m, so a centred neighbour blend at 0.5
    mixes each depth halfway into water 1–2 m away and flattens structure that is real. On a lake
    with no native-resolution fit the same reasoning applies, so all seven use the unsmoothed table.

    What smoothing was for does not disappear: aegeri stratified at 4 m reads 0.175 natively and
    0.122 / 0.126 on the two thinned fits, so both are ~30 % low there. That is a **thinning** loss,
    not a smoothing one, and no depth blend recovers it.

#### The unscaled tables in use

Fitted 2026-08-11 against `temperature_filtered.csv`, block = `max(24, W_seiche)`, before
`sigma_rep_scale`. Summary; the full per-depth tables follow.

| lake | block (h) | depths | range (m) | mixed at top | strat at top | **strat peak below 4 m** | mixed there | ratio | at deepest sensor |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|
| upperlugano | 27.3 | 16 | 0.5–40 | 0.283 | 0.342 | **0.268** @ 9 m | 0.094 | 2.8 | 0.029 / 0.008 |
| maggiore | 77.5 | 16 | 0.5–40 | 0.320 | 0.671 | **0.385** @ 11 m | 0.111 | 3.5 | 0.118 / 0.072 |
| geneva | 97.4 | 22 | 0.25–90 | 0.443 | 0.700 | **0.376** @ 24 m | 0.100 | 3.8 | 0.025 / 0.017 |
| greifensee | 24.0 | 17 | 1–16.9 | 0.198 | 0.000 | **0.412** @ 10 m | 0.145 | 2.8 | 0.121 / 0.089 |
| aegeri | 24.0 | 22 | 2–75 | 0.165 | 0.053 | **0.299** @ 8 m | 0.155 | 1.9 | 0.012 / 0.003 |
| hallwil | 24.0 | 19 | 0.5–44 | 0.247 | 0.000 | **0.222** @ 6 m | 0.115 | 1.9 | 0.018 / 0.012 |
| murten | 24.0 | 18 | 0.5–43 | 0.498 | 0.456 | **0.198** @ 9 m | 0.102 | 1.9 | 0.024 / 0.000 |

Below the filter cut the stratified table peaks at the thermocline (6–24 m) on every lake and
decays to a few hundredths by the deepest sensor — the same band, and the same physics, as the
localization radius minimum of §2.6. Above the cut the ordering flips on the small lakes: mixed
exceeds stratified on aegeri, greifensee and hallwil, where the winter surface carries more
sub-block variability than the summer one.

!!! danger "Exact zeros are real, and they are over-confident"
    The estimator floors at 0 when the free run's within-block variance exceeds the observations'.
    That happens at **greifensee 1 m, hallwil 0.5 and 1 m (stratified) and murten 43 m**. At those
    cells $\sigma_i$ collapses to $\sigma_\text{common} = 0.075$ °C whatever the scale, so the
    filter trusts them almost absolutely. Not addressed by any knob on this page.

#### Full tables, `sigma_rep_filtered_nosmooth.json`

**upperlugano** — block 27.3 h

| z (m) | 0.5 | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 25 | 30 | 35 | 40 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 0.283 | 0.215 | 0.158 | 0.042 | 0.085 | 0.094 | 0.095 | 0.084 | 0.087 | 0.097 | 0.107 | 0.106 | 0.069 | 0.023 | 0.023 | 0.029 |
| stratified | 0.342 | 0.245 | 0.252 | 0.143 | 0.258 | 0.268 | 0.250 | 0.217 | 0.195 | 0.145 | 0.109 | 0.084 | 0.050 | 0.024 | 0.012 | 0.008 |

**maggiore** — block 77.5 h

| z (m) | 0.5 | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 25 | 30 | 35 | 40 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 0.320 | 0.283 | 0.257 | 0.115 | 0.114 | 0.114 | 0.111 | 0.108 | 0.100 | 0.099 | 0.099 | 0.105 | 0.124 | 0.129 | 0.128 | 0.118 |
| stratified | 0.671 | 0.634 | 0.778 | 0.338 | 0.379 | 0.381 | 0.385 | 0.382 | 0.360 | 0.335 | 0.298 | 0.274 | 0.226 | 0.155 | 0.103 | 0.072 |

**geneva** — block 97.4 h

| z (m) | 0.25 | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 21 | 24 | 27 | 30 | 35 | 40 | 45 | 50 | 70 | 75 | 90 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 0.443 | 0.340 | 0.346 | 0.115 | 0.108 | 0.095 | 0.087 | 0.088 | 0.077 | 0.085 | 0.078 | 0.093 | 0.100 | 0.115 | 0.102 | 0.094 | 0.090 | 0.069 | 0.063 | 0.043 | 0.047 | 0.025 |
| stratified | 0.700 | 0.517 | 0.518 | 0.258 | 0.297 | 0.294 | 0.312 | 0.350 | 0.352 | 0.305 | 0.315 | 0.358 | 0.376 | 0.369 | 0.333 | 0.240 | 0.180 | 0.125 | 0.090 | 0.030 | 0.027 | 0.017 |

**greifensee** — block 24 h

| z (m) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 16.9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 0.198 | 0.166 | 0.173 | 0.100 | 0.113 | 0.121 | 0.135 | 0.139 | 0.142 | 0.145 | 0.139 | 0.183 | 0.190 | 0.178 | 0.153 | 0.122 | 0.121 |
| stratified | 0.000 | 0.195 | 0.282 | 0.144 | 0.287 | 0.344 | 0.353 | 0.347 | 0.366 | 0.412 | 0.343 | 0.269 | 0.211 | 0.157 | 0.122 | 0.099 | 0.089 |

**aegeri** — block 24 h

| z (m) | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 21 | 24 | 27 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 0.165 | 0.127 | 0.141 | 0.155 | 0.133 | 0.100 | 0.083 | 0.066 | 0.058 | 0.058 | 0.054 | 0.046 | 0.042 | 0.037 | 0.024 | 0.022 | 0.019 | 0.018 | 0.016 | 0.015 | 0.013 | 0.012 |
| stratified | 0.053 | 0.122 | 0.194 | 0.299 | 0.289 | 0.274 | 0.190 | 0.134 | 0.099 | 0.057 | 0.034 | 0.022 | 0.016 | 0.011 | 0.007 | 0.007 | 0.006 | 0.006 | 0.007 | 0.006 | 0.005 | 0.003 |

**hallwil** — block 24 h

| z (m) | 0.5 | 1 | 2.5 | 5 | 6 | 7.5 | 9 | 10 | 11 | 12.5 | 15 | 17.5 | 20 | 25 | 30 | 35 | 40 | 43 | 44 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 0.247 | 0.186 | 0.234 | 0.117 | 0.115 | 0.085 | 0.063 | 0.057 | 0.058 | 0.084 | 0.070 | 0.051 | 0.040 | 0.031 | 0.025 | 0.020 | 0.019 | 0.018 | 0.014 |
| stratified | 0.000 | 0.000 | 0.161 | 0.170 | 0.222 | 0.215 | 0.177 | 0.150 | 0.149 | 0.129 | 0.072 | 0.045 | 0.042 | 0.025 | 0.016 | 0.015 | 0.016 | 0.012 | — |

**murten** — block 24 h

| z (m) | 0.5 | 1 | 2.5 | 5 | 6 | 7.5 | 9 | 10 | 11 | 12.5 | 15 | 17.5 | 20 | 25 | 30 | 35 | 40 | 43 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 0.498 | 0.353 | 0.303 | 0.120 | 0.127 | 0.122 | 0.102 | 0.090 | 0.096 | 0.062 | 0.051 | 0.052 | 0.059 | 0.052 | 0.032 | 0.023 | 0.014 | 0.024 |
| stratified | 0.456 | 0.253 | 0.226 | 0.146 | 0.156 | 0.183 | 0.198 | 0.182 | 0.158 | 0.144 | 0.124 | 0.129 | 0.141 | 0.030 | 0.013 | 0.012 | 0.005 | 0.000 |

### 2.4 $\sigma_\text{common}$

Assumed instrument error. It has
**no month argument**, which is exactly why it is *not* the term used to close the innovation
budget, Held at the default **0.075 °C** yearround.

### 2.5 `sigma_rep_scale` — the two-band multiplier

**How it works.** A multiplier on the fitted table. It is a legitimate knob precisely because $\sigma_\text{rep}$ is a lower bound: the multiplier states what fraction of the
total representativeness error the measured sub-block band represents. It inherits the table's depth
*and* month shape instead of imposing a new one.

**Why two bands.** The scale is a property of the **observation series**, not of the lake. The
filter of §2.1 leaves depths above `THERMO_DEPTH_MIN` untouched, so a single file carries a *raw*
series above 4 m and a *filtered* series below it. One scalar cannot serve both. The split is
therefore placed at 4 m, and `_depth_stepped` accepts `{"0": a, "4": b}` directly.

**How determined.** Fitted from a completed run by `local/sigma/fit_sigma_common.py` (`k_steps`),
inverting the pooled-NIS identity per band:
$$
k^2 = \frac{\langle d^2\rangle - \langle \sigma_\text{spread}^2\rangle - \sigma_\text{common}^2}{\langle \sigma_\text{rep}^2\rangle}
$$
Fitted on Run A: **1.25 above 4 m, 2.23 at and below**. The ratio held when fitted per season
(1.95 stratified, 1.74 mixed), confirming it is the series change and not a seasonal artefact.

!!! Do not *lower* the shallow scale below its current value. Lowering R above 4 m makes the filter
    trust surface observations more, and the upperlugano buoy reads **+0.45 °C at 0.5 m** and
    **+0.28 °C at 1 m** against the Gandria CTD (21 of 23 casts). NIS cannot distinguish bias from
    scatter; do not buy calibration by trusting a biased sensor harder.

**Evidence the split is the filter, not the lake:** refitted on the *unfiltered* series the two
bands come out 1.12 and 1.15 — a ratio of **1.03**, against 1.79 on the filtered series.

#### The annual scales — first pass

Fitted from `A_final` (unsmoothed table, `sigma_rep_scale = 1.0`, localization off) and written into
`args/experiments/C_final.json`. Superseded by the season-keyed values below, which the final run
uses; the annual pair is kept because everything in §5 and §6 was measured under it.

| lake | $k$ (0–4 m) | $k$ (≥ 4 m) | ratio |
|---|--:|--:|--:|
| upperlugano | 1.2745 | 2.2347 | 1.75 |
| maggiore | 1.5163 | 2.5036 | 1.65 |
| murten | 1.5042 | 2.3793 | 1.58 |
| geneva | 1.6485 | 1.8178 | 1.10 |
| hallwil | 2.4635 | 2.6333 | 1.07 |
| aegeri | 2.2123 | 1.9208 | 0.87 |
| greifensee | 2.4546 | 2.0898 | 0.85 |

Every value is above 1: the measured sub-block band carries **38–79 %** of the total
representativeness error, and the rest is what the scale restores. The deep band needs more damping
than the shallow one on five of seven lakes; aegeri and greifensee invert it, so the two-band split
is fitted per lake and not assumed.

Multiplying through gives the σ_rep actually used. At the stratified thermocline peak and at the
shallowest sensor:

| lake | strat peak (m) | $k\sigma_\text{rep}$ | $\sigma_i$ | top depth, mixed | $k\sigma_\text{rep}$ | $\sigma_i$ |
|---|--:|--:|--:|--:|--:|--:|
| maggiore | 11 | 0.964 | 0.967 | 0.5 m | 0.485 | 0.491 |
| greifensee | 10 | 0.860 | 0.864 | 1 m | 0.485 | 0.491 |
| geneva | 24 | 0.684 | 0.688 | 0.25 m | 0.730 | 0.733 |
| upperlugano | 9 | 0.598 | 0.603 | 0.5 m | 0.361 | 0.368 |
| hallwil | 6 | 0.586 | 0.590 | 0.5 m | 0.609 | 0.614 |
| aegeri | 8 | 0.575 | 0.580 | 2 m | 0.365 | 0.373 |
| murten | 9 | 0.471 | 0.477 | 0.5 m | 0.749 | 0.753 |

$\sigma_i = \sqrt{\sigma_\text{common}^2 + (k\sigma_\text{rep})^2}$ with $N_i = 1$ — the scaled term
dominates everywhere, and $\sigma_\text{common}$ moves $\sigma_i$ by at most 0.008 °C.

#### Season-keyed scales — the final values

**Why a second axis.** $\sigma_\text{rep}$ already carries a season shape, so a season-keyed $k$ is
not re-stating it: $k$ covers the **synoptic band the estimator cannot see** — everything slower
than one block — and that band is *not* a fixed fraction of the sub-block band that was measured.
Where the two disagree, one annual $k$ is a count-weighted compromise that can sit wrong in both
seasons at once. Geneva is the case that forced it: below 4 m the annual scale left pooled NIS 1.51
mixed against 0.94 stratified.

`sigma_rep_scale` therefore accepts a season key on top of the depth steps, and
`local/sigma/fit_sigma_common.py` emits it automatically — the same closed-form band solve, run on
each season's rows, restricted to the depths the annual fit kept so both step functions cover the
same water. Same run, same command; read the block printed under *"per-season fit of the same two
bands"*.

```json
"sigma_rep_scale": { "mixed":      { "0": 1.5173, "4": 2.6102 },
                     "stratified": { "0": 1.0991, "4": 2.1410 } }
```

Fitted from `A_final` and written into `args/experiments/C_seasonal_noloc.json`, the final run:

| lake | mixed 0–4 m | mixed ≥ 4 m | strat 0–4 m | strat ≥ 4 m | mixed/strat, 0–4 m | mixed/strat, ≥ 4 m |
|---|--:|--:|--:|--:|--:|--:|
| upperlugano | 1.5173 | 2.6102 | 1.0991 | 2.1410 | 1.38 | 1.22 |
| maggiore | 1.8265 | 2.5512 | 1.4581 | 2.4971 | 1.25 | 1.02 |
| geneva | 1.8592 | 2.3362 | 1.5641 | 1.7643 | 1.19 | 1.32 |
| murten | 1.5789 | 2.5892 | 1.3931 | 2.3050 | 1.13 | 1.12 |
| greifensee | 2.3528 | 2.0942 | 2.5128 | 2.0890 | 0.94 | 1.00 |
| hallwil | 2.4214 | 2.0505 | 2.6896 | 2.7703 | 0.90 | 0.74 |
| aegeri | 2.0176 | 1.9073 | 3.4782 | 1.9240 | 0.58 | 0.99 |

The split is real but not large, and it does not point one way. Five lakes want **more** damping in
winter (the measured sub-block band under-states the total worst when the lake is mixed), hallwil
and aegeri want more in summer. **Aegeri is the largest single seasonal ratio of the seven and it is
confined to the surface** — 3.48 stratified against 2.02 mixed above 4 m, while below 4 m the two
seasons agree to 1 % (1.907 vs 1.924). Greifensee has effectively no split at all, and is carried
season-keyed only so all seven read the same way; its record starts 2025-02-21, so its mixed season
is thin.

!!! note "This is a one-shot solve, not a fixed point"
    $k$ is fitted by inverting the NIS identity on a *completed* run, but raising $\mathbf{R}$
    changes the analysis — hence the spread and the innovations the next run produces. One pass
    therefore lands near 1, not on it. §10 measures how near.

### 2.6 Vertical localization

**How it works.** With $N = 20$ members, a correlation below $2/\sqrt{N} = 0.45$ is indistinguishable
from sampling noise. The observations stop at 40 m while the column runs to ~288 m, so without
localization most of the state is updated purely through noise. Each state cell carries a radius
$L(z)$ — the separation at which its ensemble correlation drops below the threshold — and

$$
\rho_{ij} = \text{GC}\!\left(\frac{2\,|z_i - z_j|}{L_{ij}}\right),
\qquad L_{ij} = \tfrac{1}{2}\left(L(z_i) + L(z_j)\right)
$$

with GC the Gaspari–Cohn (1999) correlation function: 1 at zero separation, falling smoothly to
exactly 0 at $L_{ij}$. The radius profile is *measured* per depth rather than fitted to a functional
form, and one threshold is the only knob.

!!! note "Applied to both PHᵀ and HPHᵀ"
    The gain is $(\rho\circ\mathbf{PH}^{\!\top})(\rho\circ\mathbf{HPH}^{\!\top}+\mathbf{R})^{-1}$,
    and both factors must describe the same observation set. Tapering only $\mathbf{PH}^{\!\top}$
    leaves an inverse built for the full set, and the damped observations re-enter through it with
    the wrong sign — that is what diverged murten in April 2025. Gaspari–Cohn is
    positive semi-definite, so by the Schur product theorem $\rho\circ\mathbf{P}$ stays a valid
    covariance; a binary mask is not, and cannot be applied to either side safely.
    The intended consequence is unchanged: **water far below the deepest observation free-runs.**

**How determined.** `notebooks/localization.py` against a dedicated **free ensemble run** (20
members, no assimilation), over the stratified season, threshold 0.45 on all seven lakes.

**Values in use** (`localization/<lake>.json`, derived 2026-08-06/10). A full per-depth profile is
too long to print, and four numbers reproduce its shape: the radius at the surface, the minimum and
where it sits, the radius at the deepest observed depth, and the maximum at the bottom. The right
three columns are what `localization.summarize()` logs at the first analysis of the run — the taper
applied to the real Simstrat state grid and the real assimilated depths, not a derivation.

!!! warning "Right three columns are stale"
    They were logged under the binary mask. `summarize()` now counts a cell as reached at taper
    weight ≥ 0.01 and reports summed weight rather than a tally of non-zeros, so both the counts and
    the observations-per-cell figures will move. Re-log them from the next run.

| lake | column (m) | observed (m, n) | $L$ surface | **min $L$** | at deepest obs | max $L$ | state cells updated | deepest updated | obs per updated cell |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| geneva | 309 | 0.25–90 (27) | 11 | 7.0 @ 5 m | 67 | 202 | 209/316 (**66 %**) | 202 m | 11.4/27 |
| maggiore | 369 | 0.5–40 (16) | 15 | 9.0 @ 8 m | 23 | 174 | 118/371 (**32 %**) | 116 m | 6.3/16 |
| upperlugano | 287 | 0.5–40 (16) | 7 | 2.0 @ 6 m | 17 | 155 | 58/289 (**20 %**) | 56 m | 4.7/16 |
| aegeri | 81 | 2–75 (22) | 5.4 | 1.8 @ 4.2 m | 52.6 | 58.6 | 243/243 (100 %) | 81 m (bottom) | 7.6/22 |
| hallwil | 47 | 0.5–44 (19) | 6 | 2.5 @ 5 m | 29 | 32 | 53/53 (100 %) | 47 m (bottom) | 7.1/19 |
| murten | 44 | 0.5–43 (35) | 6 | 3.0 @ 4 m | 28 | 29 | 51/51 (100 %) | 44 m (bottom) | 13.7/35 |
| greifensee | 31 | 1–16.9 (17) | 4.8 | 1.7 @ 4.1 m | 8.2 | 19.1 | 176/176 (100 %) | 31 m (bottom) | 6.5/17 |

The shape is the same on every lake: a few metres at the surface, a **minimum of 1.7–9 m at
4–8 m** — the thermocline, the same band where σ_rep peaks — then monotone growth to tens or
hundreds of metres in the near-isothermal hypolimnion.

Localization only cuts anything off on the three deep lakes, where the chain stops far short of the
bottom: everything below 56 m (upperlugano), 116 m (maggiore) or 202 m (geneva) free-runs, which is
68–80 % of the state on the two Ticino lakes. On the four pre-alpine lakes the chain reaches near
the bottom and the growing radius covers the rest, so **every state cell keeps an observation in
reach** — the mask still removes long-range pairs (4.7–13.7 of the available observations reach a
given cell) but leaves no free-running water.

**Config.** `localization/<lake>.json`; `"localization": true`. Deleting the json disables it.

!!! warning "The final run has localization OFF on all seven lakes"
    `C_seasonal_noloc.json` sets `"localization": false`. That is deliberate, not an oversight: the
    season-keyed scales were fitted on `A_final`, which ran unlocalized, so this is the one
    configuration in which the fit's own assumptions hold — see §10. The localized counterpart
    (`C_seasonal.json`, same scales, taper on) exists and the pair isolates localization at fixed
    $\mathbf{R}$, exactly as `A_final`/`C_final` did for the annual scale.

### 2.7 Season definition

`STRATIFIED_MONTHS = (5, 6, 7, 8, 9, 10)` in `src/assimilator/functions.py`, the single definition
used by the σ_rep fit, the localization derivation, and every diagnostic.

!!! info "The calendar was tested and won"
    A measured stratification rule (Δρ threshold, with hysteresis and a minimum run length) was
    implemented and scored against the calendar as a labelling for σ_rep. A whole-lake index was
    **9 % worse** on within-label homogeneity; a per-depth index was 0.3 % better at the cost of one
    free parameter per depth. The calendar is kept. April's elevated NIS is **ensemble
    under-dispersion**, not a mislabelled regime: its measured within-day variability (0.12–0.33 °C)
    sits at the mixed level (0.13–0.21), not the stratified one (0.49–0.93), while its innovations
    are summer-sized and its spread is 30–43 % smaller. See `local/analyses&plans/regime_selection.md`.

**The decision, per lake.** The same rule is used everywhere, and this is the evidence it was
checked against. NIS is pooled in the *label-sensitive band* — the depths where raw
σ_rep(stratified)/σ_rep(mixed) ≥ 2, i.e. the only depths a regime change would move. `|mean d|/rms d`
separates representativeness error (zero-mean scatter) from model bias, which a larger R cannot fix.

| lake | label-sensitive band | Apr NIS | Apr / May–Oct | Apr bias frac | Dec NIS | Dec bias frac | season used |
|---|---|--:|--:|--:|--:|--:|---|
| murten | 7.5–20 m | 6.86 | 2.39 | 0.01 | 0.32 | 0.82 | May–Oct |
| upperlugano | 5–17 m | 5.73 | 3.64 | 0.01 | 0.34 | 0.92 | May–Oct |
| geneva | 2–55 m | 5.05 | 2.10 | 0.23 | 0.48 | 0.13 | May–Oct |
| maggiore | 1–25 m | 4.80 | 2.56 | 0.05 | 0.28 | 0.70 | May–Oct |
| hallwil | 5–12.5 m | 2.14 | 1.16 | 0.29 | 0.22 | 0.86 | May–Oct |
| aegeri | 6.4–19.2 m | 1.89 | 1.25 | 0.14 | 0.43 | 0.68 | May–Oct |
| greifensee | 4.6–11.6 m | 1.83 | 1.20 | 0.17 | 0.05 | 0.92 | May–Oct |

Two facts decided it. **April is the least biased month of the year on almost every lake** (bias
fraction 0.01–0.29), so its large innovations really are representativeness scatter — but its
*measured* within-day variability sits at the mixed level, so relabelling it stratified would raise
R for the wrong reason. **December is the most biased month** (0.68–0.92 on six of seven) and
already over-damped (0.05–0.48); a Δρ index would have relabelled it stratified on the deep lakes
and made it worse. That asymmetry is what disqualified the measured rule.

A per-lake fixed calendar was also considered and rejected: measured onset/turnover at Δρ ≥ 0.3
moves by four weeks (onset) and seven weeks (turnover) between years on the same lake — geneva
onset 01–30 Apr over 2020–2025, turnover 16 Oct–04 Dec — so a per-lake constant removes the
systematic April/November error but not the year-to-year spread.

---

## 3. Calibration order

The order is forced by the dependencies, and each step invalidates everything downstream:

1. **Free run** → the reference the σ_rep estimator subtracts (`local/make_ref.py`).
2. **Filter the observations** (§2.1) — needs `W_SEICHE`. Always reads the raw sub-daily record.
3. **Choose the hour, thin** (§2.2) — one instant per day, and for the fine chains a depth grid too.
4. **Fit σ_rep** (§2.3) against the **filtered sub-daily** series, at the depths of step 3.
5. **Free ensemble run → localization table** (§2.6).
6. **Run once at `sigma_rep_scale = 1.0`**, localization on — the scale can only be fitted *from a run*.
7. **Fit the two-band scale** (§2.5) from that run's budget.
8. **Final run** with the fitted scale.

Steps 6–7 are the only unavoidable two-pass loop.

Three ordering rules that are easy to get backwards:

**Filter before thinning, never the reverse.** The filter removes sub-daily variability, so it has to
see the sub-daily record; thinning first destroys what it is meant to act on. It also has to be
*filter then thin* rather than both at once — a filtered series joined onto a thinned schedule
matches on exact timestamps, and the filter's hourly binning shifts them.

**Fit σ_rep on the sub-daily file, at the thinned depths.** The estimator needs within-block
variance, which a one-per-day file does not have — so pass `temperature_filtered.csv`, not the
thinned output. But pass `--depths` matching the thinned grid, because depth smoothing blends each
depth toward its *neighbours*, and neighbours 0.1 m apart are not neighbours 2 m apart.

**Localization must be OFF during the scale fit.** The mask is applied to $\mathbf{P}\mathbf{H}^{\!\top}$
but not to $\mathbf{H}\mathbf{P}\mathbf{H}^{\!\top}$ (§2.6), so the two sides of the gain describe
different covariances and the analysis is no longer guaranteed to reduce spread. $\mathbf{R}$ is what
keeps that bounded — and step 6 sets $k = 1$, which makes $\mathbf{R}$ as small as it ever gets.
Measured: with localization on at $k = 1$, upperlugano and murten diverged in April (spread growing
14–38× per analysis); with it off, identical settings, both complete the year cleanly. Localization
is safe once the scale is damped, so it belongs in step 8. A second fit *from* the localized run is
then legitimate, because by that point $k \neq 1$.

---

## 4. The experiment matrix

Six runs, each differing from its neighbour in exactly one respect, so every difference is
attributable. Configs are in `args/experiments/upperlugano_*.json`.

| run | engine | observations | σ_rep table | `sigma_rep_scale` | localization | isolates |
|---|---|---|---|---|---|---|
| **A** | native | filtered | smoothed | 1.0 | off | baseline; the run the scale is fitted from |
| **B** | native | filtered | smoothed | 1.25 / 2.23 | off | the two-band scale |
| **Araw** | native | **raw** | raw-fit | 1.0 | off | the observation filter |
| **C** | native | filtered | smoothed | 1.25 / 2.23 | **on** | localization |
| **Cns** | native | filtered | **unsmoothed** | 1.25 / 2.23 | on | depth smoothing |
| **OpenDA** | **OpenDA** | filtered | smoothed | 1.25 / 2.23 | off¹ | the filter implementation |

¹ OpenDA's Hamill scheme is a distance taper with one scalar radius, not the measured binary
per-depth mask, so it is not the same instrument and enabling it would confound the comparison.
See §6.

All six are scored against the **raw hourly** series (139,728 pairs), never the filtered one: a run
that assimilated filtered observations and is scored against them reports a flattering RMSE against
its own target.

---

## 5. Results

### Summary

| | A<br>*scale 1.0* | B<br>*two-band R* | Araw<br>*unfiltered obs* | **C**<br>***+ localization*** | Cns<br>*unsmoothed table* | OpenDA<br>*C's R, no loc* |
|---|--:|--:|--:|--:|--:|--:|
| observations scored | 5824 | 5824 | 5824 | 5824 | 5824 | 5840 |
| **pooled RMSE** (free 0.831) | 0.615 | 0.631 | 0.738 | 0.611 | 0.611 | **0.598** |
| gain vs free run | −26.0 % | −24.1 % | −11.2 % | −26.5 % | −26.4 % | **−28.0 %** |
| depths improved | 15/16 | **16/16** | 11/16 | **16/16** | **16/16** | **16/16** |
| pooled NIS | 2.639 | 1.081 | 1.302 | **1.028** | 1.086 | 1.060 |
| mean NIS over depths | 2.648 | 0.997 | 1.247 | 0.938 | 1.038 | **0.994** |
| NIS mixed | 2.725 | 1.270 | 1.943 | 1.335 | 1.510 | **1.300** |
| NIS stratified | 2.601 | 1.019 | 1.209 | 0.926 | 0.950 | **0.980** |
| RMSE (assimilated series) | 0.3512 | 0.3791 | 0.6156 | 0.3489 | 0.3505 | **0.3287** |
| bias | 0.0828 | 0.1185 | 0.1509 | 0.0681 | 0.0672 | **0.0516** |
| MAE | 0.2169 | 0.2351 | 0.3458 | 0.2238 | 0.2240 | **0.2056** |
| mean spread | 0.0504 | 0.0621 | 0.0707 | **0.0741** | 0.0735 | 0.0633 |
| coverage 1σ | 0.1485 | 0.1743 | 0.1545 | **0.1878** | 0.1851 | 0.1848 |
| coverage 2σ | 0.2922 | 0.3340 | 0.3008 | **0.3856** | 0.3695 | 0.3611 |

!!! note "Why OpenDA scores 5840 and the native runs 5824"
    OpenDA's window ends at `end_date + 1` so the last day's observation falls inside it, giving one
    extra analysis — 16 obs, one day × 16 depths (`openda/config.py`). Too small to move these
    numbers, but the two engines are not scored over an identical instant set.

### RMSE by depth, vs raw hourly observations (°C)

| depth | free run | A<br>*scale 1.0* | B<br>*two-band R* | Araw<br>*unfiltered* | C<br>*+ localization* | Cns<br>*unsmoothed* | OpenDA<br>*C's R* |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.5 | 0.948 | 0.481 | 0.472 | **0.456** | 0.480 | 0.478 | 0.474 |
| 1 | 0.951 | 0.390 | 0.384 | **0.371** | 0.388 | 0.389 | 0.384 |
| 3 | 1.014 | 0.387 | 0.393 | 0.397 | **0.386** | 0.391 | 0.388 |
| 5 | 1.066 | **0.529** | 0.557 | 0.583 | 0.535 | 0.530 | 0.540 |
| 7 | 1.033 | 0.774 | 0.795 | 0.858 | 0.775 | **0.772** | 0.775 |
| 9 | 1.137 | 0.979 | 0.995 | 1.137 | 0.968 | 0.968 | **0.958** |
| 11 | 1.221 | 1.076 | 1.108 | 1.351 | 1.056 | 1.053 | **1.030** |
| 13 | 1.008 | 0.963 | 1.005 | 1.270 | 0.951 | 0.951 | **0.911** |
| 15 | 0.865 | 0.795 | 0.815 | 1.012 | 0.792 | 0.793 | **0.775** |
| 17 | 0.702 | 0.656 | 0.669 | 0.801 | 0.661 | 0.664 | **0.637** |
| 19 | 0.590 | 0.509 | 0.528 | 0.615 | 0.517 | 0.522 | **0.488** |
| 21 | 0.529 | 0.413 | 0.427 | 0.484 | 0.418 | 0.424 | **0.405** |
| 25 | 0.452 | 0.253 | 0.260 | 0.296 | 0.256 | 0.263 | **0.250** |
| 30 | 0.326 | **0.117** | 0.132 | 0.148 | 0.125 | 0.125 | 0.134 |
| 35 | 0.139 | 0.085 | 0.071 | **0.070** | 0.073 | 0.072 | 0.072 |
| 40 | 0.086 | 0.101 | 0.084 | **0.079** | 0.080 | 0.082 | 0.085 |
| *mean all* | 0.754 | 0.532 | 0.543 | 0.620 | 0.529 | 0.530 | **0.519** |
| *mean 0–4 m* | 0.971 | 0.419 | 0.416 | 0.408 | 0.418 | 0.419 | **0.415** |
| *mean 4 m+* | 0.704 | 0.558 | 0.573 | 0.670 | **0.554** | 0.555 | **0.543** |

### NIS by depth

Closest to 1.00 is best; **bold** marks the winner at each depth.

| depth | A<br>*scale 1.0* | B<br>*two-band R* | Araw<br>*unfiltered* | C<br>*+ localization* | Cns<br>*unsmoothed* | OpenDA<br>*C's R* |
|--:|--:|--:|--:|--:|--:|--:|
| 0.5 | 1.138 | 0.736 | **1.039** | 0.753 | 0.763 | 0.734 |
| 1 | 1.334 | 0.888 | 1.211 | 0.881 | 1.203 | 0.877 |
| 3 | 2.353 | 1.690 | **1.398** | 1.611 | 1.505 | 1.654 |
| 5 | 3.237 | **1.030** | 1.468 | 0.908 | 1.829 | 1.125 |
| 7 | 4.692 | 1.322 | 1.218 | 1.304 | **1.137** | 1.310 |
| 9 | 3.755 | **1.011** | 1.056 | 1.192 | 1.121 | 1.017 |
| 11 | 3.802 | 1.192 | 1.282 | **1.009** | 0.971 | 1.122 |
| 13 | 4.111 | 1.333 | 1.430 | **1.034** | 1.074 | 1.208 |
| 15 | 3.742 | 1.067 | 1.238 | **1.020** | 0.983 | 1.071 |
| 17 | 2.984 | 0.903 | 1.411 | 0.884 | **0.951** | 0.844 |
| 19 | 2.895 | 1.021 | 1.544 | 0.938 | **1.002** | 0.880 |
| 21 | 2.962 | 1.055 | 1.826 | 0.988 | **1.015** | 0.982 |
| 25 | 2.275 | 0.981 | 1.370 | 0.948 | 1.167 | **1.066** |
| 30 | 0.784 | 0.694 | **1.042** | 0.594 | 0.839 | 0.927 |
| 35 | 0.860 | 0.323 | **0.555** | 0.340 | 0.383 | 0.348 |
| 40 | 1.445 | 0.700 | **0.860** | 0.605 | 0.664 | 0.735 |

### What each comparison settles

**Observation filtering (A vs Araw) — worth ~15 points.** −26.0 % against −11.2 %, and 15/16 depths
against 11/16. Raw observations actively *damage* the 9–21 m band (13 m: 0.963 → 1.270, +32 %).
This is the single largest effect measured. The one place raw wins is 0.5–1 m (0.456 vs 0.480),
which is consistent — that band is passed through untouched, so the difference there is the
σ_rep table, not the filter.

**The two-band scale (A vs B) — buys calibration, costs accuracy.** NIS 2.64 → 1.08 and 15/16 →
16/16 depths, at the price of 2 points of RMSE (0.615 → 0.631). This is expected and not a defect:
raising $\mathbf{R}$ shrinks the gain, so increments get smaller everywhere, which helps only where
they were doing harm.

**Localization (B vs C) — recovers the cost for free.** 0.631 → 0.611, back to A's accuracy while
keeping B's 16/16 coverage; NIS 1.081 → 1.028, coverage 1σ 0.174 → 0.188, bias 0.119 → 0.068.
Spread *rises* (0.062 → 0.074) with RMSE flat — the honest signature of an ensemble that has stopped
collapsing through spurious long-range updates.

The reason this works is that B and C address the same problem in different places. Extra
$\mathbf{R}$ says *"trust this observation less everywhere"*; localization says *"this observation
may not reach here at all."* The second is better targeted, so it removes the bad updates without
damping the good ones.

**Depth smoothing (C vs Cns) — accuracy-neutral, calibration-positive.** Identical pooled RMSE
(0.611 both) and mean per-depth RMSE within 0.001 °C. The difference is entirely calibration, and it
is concentrated at one depth: **5 m, NIS 0.908 smoothed against 1.829 unsmoothed.** The raw fit has
a trough there ($\sigma_\text{rep}$ = 0.143, against 0.252 at 3 m and 0.258 at 7 m); halving and
rebounding across 4 m of water is a sensor or sampling artefact, and leaving it in makes
$\mathbf{R}$ far too small at that depth. The smoother fills it.

!!! note "Honest caveat on smoothing"
    Averaged with equal weight per depth, `mean |log NIS|` slightly *favours* the unsmoothed table
    (0.225 vs 0.240, closer to 1 at 10 of 16 depths). That is entirely the 30–40 m tail, where both
    runs are over-damped and smoothing raises σ_rep further. Those depths carry 0.086–0.326 °C of
    free-run error against 1.0–1.2 °C at the thermocline, so the pooled (variance-weighted) verdict
    is the one to follow. **Smoothing is justified on thermocline calibration, not on accuracy.**

!!! danger "Superseded — the final runs do not smooth"
    This section reads NIS on upperlugano alone, where nothing independent says which table is
    *right*. The native-resolution fits on aegeri and greifensee do say, and they favour the
    unsmoothed table (§2.3). `A_final` and `C_final` therefore use it on all seven lakes, and the
    5 m result above is better read as evidence that thinning, not the raw fit, is what loses the
    trough.

**Filter implementation (C vs OpenDA) — see §6.**

!!! info "The scale does not need refitting when the table is smoothed"
    Smoothing redistributes σ_rep over depth but preserves the band means: the
    nosmooth/smooth rms ratio is 0.960 (0–4 m) and 0.993 (4 m+), so the fitted 1.25 / 2.23 would
    move only to 1.30 / 2.25. Holding the scale fixed is what makes C vs Cns a pure shape ablation.

---

## 6. Cross-engine validation (OpenDA)

The same observations and the same $\mathbf{R}$ are run through OpenDA's EnKF, so that a residual
difference is attributable to **the filter implementation** rather than to the error model.

### Making R identical, not merely similar

OpenDA's `TimeSeriesFormatterStochObserver` carries **one `standardDeviation` per time series**, so
the month axis used to be collapsed to an observation-weighted RMS. That was not a small
compromise: at 7 m the native σ spans 0.197 (mixed) to 0.520 (stratified) and OpenDA used 0.394 all
year — roughly **2× over-damped at the thermocline in winter**, ~25 % over-confident in summer,
worst exactly in the band that carries the residual skill.

The fix rests on one observation: **σ_rep is a step function in season, not a curve in month.**
`by_month` is a fan-out of `by_season` and $N_i = 1$, so the whole native σ model is
**16 depths × 2 seasons = 32 numbers**. Series are ours to define, so each depth is split into a
mixed and a stratified series, each with its own `standardDeviation` — all 32 values, exactly, with
no new observer class.

```xml
<timeSeries id="T_7m_mixed"      standardDeviation="0.1970242637">T_7m_mixed_real.csv</timeSeries>
<timeSeries id="T_7m_stratified" standardDeviation="0.5196784781">T_7m_stratified_real.csv</timeSeries>
```

Both model-side series point at the depth's single predictor file — the observer samples model
output only at its own observation times, so the two seasons never collide and the wrapper is
unchanged. There is no computational cost: the series never co-occur, so each analysis still sees
16 active observations.

The switch is **derived from the table, not configured**: deleting `sigma_rep*.json` reverts to the
scalar, a non-seasonal table falls back to the per-depth RMS, and a table that is genuinely monthly
or a lake with $N_i > 1$ declines the split rather than approximating it. Full design and guards in
`local/analyses&plans/sigma_rep_seasonal_openda.md`.

!!! warning "A blocker this uncovered"
    `OBS_TARGET_HOUR` was hardcoded to noon and the emitted day fraction to `+0.5`. Upperlugano's
    assimilated series is **06:00 only**, so the noon bin would have matched **zero rows** — the
    OpenDA run could not have worked at all. The hour is now taken from the series itself when it
    is already thinned to one reading per day.

### Result

OpenDA is **modestly but consistently more accurate** (0.598 vs 0.611 pooled, −28.0 % vs −26.5 %),
with lower bias and MAE, and the gain is concentrated in **11–21 m**:

| depth | C | OpenDA | Δ |
|--:|--:|--:|--:|
| 13 | 0.951 | **0.911** | −0.039 |
| 19 | 0.517 | **0.488** | −0.028 |
| 11 | 1.056 | **1.030** | −0.026 |
| 17 | 0.661 | **0.637** | −0.024 |
| 30 | **0.125** | 0.134 | +0.010 |

That is the band where C applies localization and OpenDA does not. C's mask is derived over the
stratified season and localizes harder than winter physics requires, so it discards information
OpenDA still uses — and the two small losses at 30 and 40 m are the other side of the same coin,
where unlocalized updates start to hurt. The per-depth NIS agrees: OpenDA sits at 0.93 / 0.74 at
30 / 40 m against C's 0.59 / 0.61, closer to 1 because it still updates there.

**OpenDA's lower spread is better calibration, not over-confidence.** Spread is 15 % lower
(0.0633 vs 0.0741) but the innovations shrink in proportion, so NIS stays at 1 — and OpenDA is
closer to 1 than C on mean-over-depths (0.994 vs 0.938), mixed (1.300 vs 1.335) and stratified
(0.980 vs 0.926).

!!! note "This is not a clean filter-vs-filter pair"
    C carries localization and OpenDA does not, so the honest algorithm comparison is against **B**
    (no localization: 0.631, NIS 1.081) — which OpenDA beats on both axes by more. Enabling
    OpenDA's Hamill taper would not fix this: it is a distance taper with one scalar radius, not
    the measured per-depth mask, so it would swap one confound for another.

### NIS for an OpenDA run

OpenDA writes no innovation or prior-spread files, so the budget and NIS plots cannot read it. They
do not have to: `Results/enkf_results.py` already logs `obs`, `pred_f` ($H\mathbf{x}^f$) and
`pred_f_std` (prior spread in observation space) per analysis time — the three terms of the
identity. `local/sigma/openda_nis.py` rebuilds NIS from them, using the run's own
`resolve_sigma_obs` so it is the native definition applied to OpenDA's numbers.

The result vectors carry **no depth labels** — they are in stochObserver series order, and under
the season split only half the series have data at any step. The tool therefore matches the obs
vector back against the assimilated CSV at every step and aborts on mismatch, rather than silently
pairing σ to the wrong depth.

```bash
python local/sigma/openda_nis.py args/experiments/upperlugano_openda_split.json \
    --run-root ~/openda-upperlugano-split --filter enkf
```

---

## 7. Final configuration — the worked example

The multi-lake final configuration is §10. This is the upperlugano run the experiment matrix of §4
was scored on, `args/experiments/upperlugano_0807_C.json`:

```json
{
  "engine": "python", "model": "simstrat", "algorithm": "EnKF",
  "window_mode": "obs", "n_members": 20, "inflation": 1.0, "rng_seed": 42,
  "start_date": "2025-01-01", "end_date": "2025-12-31",
  "sigma_obs": 0.5, "sigma_common": 0.075, "sigma_scale": 1.0,
  "localization": true,
  "lakes": {
    "upperlugano": {
      "obs_file": "observations/upperlugano/temperature_new_h06_1d.csv",
      "sigma_rep_file": "observations/upperlugano/sigma_rep_filtered.json",
      "sigma_rep_scale": { "0": 1.25, "4": 2.23 }
    }
  }
}
```

Supporting artefacts, all outside version control and all of which must travel with the run:

| artefact | produced by | kill switch |
|---|---|---|
| `filter/upperlugano.json` (`W_SEICHE` 27.3, `THERMO_DEPTH_MIN` 4.0) | `notebooks/calibrate_filter.py` | — |
| `observations/upperlugano/temperature_new_h06_1d.csv` | `notebooks/filter_observations.py` + `local/thin_obs.py` | point `obs_file` at the raw series |
| `observations/upperlugano/sigma_rep_filtered.json` | `notebooks/sigma_rep_from_obs.py` | delete → falls back to scalar `sigma_obs` |
| `localization/upperlugano.json` | `notebooks/localization.py` | delete, or `"localization": false` |

**Performance:** RMSE 0.831 → 0.611 °C against the raw hourly series (**−26.4 %**), improving all
16 observed depths; pooled NIS 1.03.

---

## 8. Known open items

These are measured, understood, and *not* addressed by any knob on this page.

**Spring ensemble under-dispersion.** Mixed-season NIS is 1.335 against 0.926 stratified, and April
carries 53 % of the mixed numerator on 18 % of its weight. April's spread is 0.085 against 0.12 in
summer while its innovations are summer-sized. This is $\mathbf{P}$, not $\mathbf{R}$: the fix is in
`perturbations/` or inflation, and no observation-error term reaches it. Raising $\mathbf{R}$ to
compensate would move the analysis in the *wrong* direction — spread and R shift the Kalman gain
oppositely.

**The 9–13 m band is the residual skill.** Every configuration leaves 0.95–1.06 °C there against a
free run of 1.0–1.2 °C — a gain of 5–13 %, against 60 % at the surface and 62 % at 30 m. That is
the thermocline, and it is where the remaining improvement lives.

**Below 30 m everything is over-damped.** NIS 0.34–0.66 at 35–40 m in all four filtered runs. The
two-band scale cannot see this, since both bands are fitted over the observed column as a whole.
Stakes are low (free-run error 0.086–0.326 °C) but it is a genuine miscalibration.

**Localization is derived over the stratified season only** and applied year-round, so it localizes
harder than winter physics requires. Mixed-season NIS moved 1.270 → 1.335 when it was switched on.
A deliberate trade, documented in `localization/upperlugano.json`.

---

## 9. Reproducing

One lake end to end. `--lakes all` runs every block in the config instead; it is **required** for a
multi-lake config, which otherwise fails with "no lake selected".

```bash
# 1. free run reference
python local/make_ref.py upperlugano --start 2025-01-01 --end 2026-01-01

# 2. filter — always reads observations/<lake>/temperature.csv, whatever obs_file says
python notebooks/filter_observations.py args/run_enkf.json --lake upperlugano

# 3. choose the hour, then thin to one instant per day
python local/hour_decision/best_hour.py --lakes upperlugano
python local/thin_obs.py --lake upperlugano --hour 6 \
    --in-file  observations/upperlugano/temperature_filtered.csv \
    --out-file observations/upperlugano/temperature_filtered_h06_1d.csv
#   fine chains add a depth grid:  --depths 1,2,3,...
#   a drifting sampling phase uses a window instead: --hour-window 0-6

# 4. fit sigma_rep — the SUB-DAILY filtered file, at the depths thinning kept
python notebooks/sigma_rep_from_obs.py args/run_enkf.json --lake upperlugano \
    --obs-file observations/upperlugano/temperature_filtered.csv \
    --smooth 0.5 --smooth-passes 2 --suffix _smooth

# 5. free ensemble run -> localization table (independent of sigma_rep)
python notebooks/localization.py --lake upperlugano --threshold 0.45

# 6-7. run at scale 1.0 with localization on, then fit the two-band scale from it
python src/assimilate.py args/experiments/upperlugano_0807_A.json --run-root ~/A-upperlugano
python local/sigma/fit_sigma_common.py args/experiments/upperlugano_0807_A.json \
    --run-root ~/A-upperlugano --max-steps 2 --eval-obs hourly

# 8. final run with the fitted scale
python src/assimilate.py args/experiments/upperlugano_0807_C.json --run-root ~/C-upperlugano
```

Reading step 7's output: it scores five candidate fits, and the one that matters is `k_scalar` /
`k_steps` — the multiplier on σ_rep, which feeds `sigma_rep_scale`. Read the **per-season** NIS
columns, not `NIS all`; a fit can hit pooled NIS = 1 by construction while being wrong in both
seasons separately. Score against raw hourly observations (`--eval-obs hourly`), never the filtered
series a run was trained on.

Diagnostics, written into the run directory:

```bash
python local/plotting/plot_multilake_budget.py <config> --run-root <dir> --out <dir>/budget.png
python local/plotting/plot_multilake_budget.py <config> --run-root <dir> --by-season \
    --out <dir>/budget_by_season.png
python local/plotting/plot_multilake_nis.py     <config> --run-root <dir> --out <dir>/nis.png
python local/plotting/plot_multilake_surface.py <config> --run-root <dir> --out <dir>/surface.png
# and the RMSE profile in four variants: {pct,degC} x {filtered,hourly}
python local/plotting/plot_multilake_rmse_profile.py <config> --run-root <dir> \
    --metric degC --eval-obs hourly --out <dir>/rmse_profile_degC_hourly.png
```

Read the **budget** plot as black against red: black to the right of red is under-dispersed, and the
band widths say whether spread or σ is responsible. Always quote the **hourly** RMSE variant.

---

## 10. All seven lakes

The runs that carry every calibrated value on this page, all over 2025 with 20 members.
**`C_seasonal_noloc` is the final one for now** — the numbers below are from
`~/C_seasonal_noloc.zip`, run 2026-08-14:

| | `A_final.json` | `C_final.json` | **`C_seasonal_noloc.json`** |
|---|---|---|---|
| σ_rep table | `sigma_rep_filtered_nosmooth.json` | same | same |
| `sigma_rep_scale` | 1.0 | fitted two-band, annual | **fitted two-band × season**, §2.5 |
| localization | off | on | **off** |
| `obs_operator` | `linear` | `linear` | `linear` |
| role | the run both scales are fitted from | first final run | **the current final run** |

!!! warning "Two differences from the worked example"
    All three configs use the **unsmoothed** σ_rep table, and all three set
    `"obs_operator": "linear"`. The linear operator spreads each observation row over the two
    bracketing state cells instead of putting 1.0 on the nearest, which matters because Simstrat
    centres cells at x.25/x.75 on its 0.5 m grid — so 135 of the 138 observation depths across
    these seven lakes fall exactly halfway between two cells, and `nearest` breaks that tie by
    array order. The default stays `nearest` so older runs reproduce bit-for-bit. Neither change
    is covered by the upperlugano matrix of §4–5.

### Per-lake summary — `C_seasonal_noloc`, the current final run

Scored against the **raw hourly** series, never the filtered one the run was steered toward, and at
the **assimilated depths only** (`--eval-obs hourly --assim-depths`), so a thinned lake is not scored
mostly where it was never told anything. 2025, 20 members, 223–364 analyses per lake.

| lake | free run | **C_seasonal_noloc** | improvement | depths improved | pooled NIS | *(C_final)* |
|---|--:|--:|--:|:--:|--:|--:|
| hallwil | 0.93 | **0.43** | −53.9 % | 19/19 | 1.32 | *0.40, −57 %* |
| geneva | 1.39 | **0.81** | −41.7 % | 27/27 | 1.10 | *0.81, −42 %* |
| murten | 0.72 | **0.45** | −36.5 % | 18/18 | 1.14 | *0.45, −37 %* |
| greifensee | 1.09 | **0.71** | −34.4 % | 16/17 | 1.31 | *0.70, −35 %* |
| maggiore | 1.54 | **1.01** | −34.2 % | 16/16 | 1.56 | *1.00, −35 %* |
| aegeri | 0.48 | **0.34** | −29.6 % | 20/22 | 1.25 | *0.34, −29 %* |
| upperlugano | 0.83 | **0.60** | −27.6 % | 15/16 | 1.10 | *0.60, −28 %* |

RMSE in °C. **Every lake improves by 28–54 %, and 131 of 135 scored depths improve.** The four
losses are all tiny and all in water the filter has nearly nothing left to correct: upperlugano
40 m (0.086 → 0.088), greifensee 12 m (0.659 → 0.670), aegeri 14 m (0.525 → 0.541) and aegeri 12 m
(0.564 → 0.682, the one real loss of the set, at the bottom of that lake's thermocline).

Against the assimilated series, the run's own scorecards:

| lake | n | bias | RMSE | MAE | mean spread | coverage 1σ | 2σ |
|---|--:|--:|--:|--:|--:|--:|--:|
| aegeri | 7136 | +0.017 | 0.265 | 0.141 | 0.035 | 0.151 | 0.293 |
| upperlugano | 5824 | +0.046 | 0.332 | 0.206 | 0.061 | 0.177 | 0.351 |
| hallwil | 6412 | −0.036 | 0.367 | 0.228 | 0.063 | 0.184 | 0.370 |
| murten | 6329 | −0.018 | 0.371 | 0.214 | 0.076 | 0.228 | 0.417 |
| geneva | 7731 | −0.036 | 0.490 | 0.305 | 0.050 | 0.108 | 0.213 |
| greifensee | 3694 | −0.014 | 0.567 | 0.366 | 0.079 | 0.164 | 0.317 |
| maggiore | 5840 | +0.270 | 0.790 | 0.535 | 0.050 | 0.070 | 0.146 |

Bias is under 0.05 °C on six of seven. **Maggiore is the exception at +0.27 °C**, and that is the
same defect §8 names: a bias no inflation of $\mathbf{R}$ can remove.

### Did the season axis work?

`C_seasonal_noloc` is the **self-consistency test**. The season-keyed scales were fitted by
`local/sigma/fit_sigma_common.py` on `A_final`, which ran localization off at
`sigma_rep_scale = 1.0`; this run therefore restores the fit's own assumptions, and pooled NIS
should land near 1.00 in *both* seasons and *both* bands for every lake.

| lake | all | mixed | stratified | mixed 0–4 m | mixed ≥ 4 m | strat 0–4 m | strat ≥ 4 m | \|mean d\|/rms d |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| upperlugano | 1.10 | 1.06 | 1.12 | 1.00 | 1.09 | 1.04 | 1.14 | 0.15 |
| geneva | 1.10 | 1.21 | 1.07 | 1.12 | 1.36 | 1.05 | 1.08 | 0.08 |
| murten | 1.14 | 1.01 | 1.24 | 0.96 | 1.10 | 0.99 | 1.37 | 0.05 |
| aegeri | 1.25 | 1.15 | 1.29 | 1.36 | 1.10 | 0.98 | 1.30 | 0.07 |
| greifensee | 1.31 | 1.09 | 1.36 | 1.02 | 1.12 | 1.07 | 1.42 | 0.02 |
| hallwil | 1.32 | 1.19 | 1.39 | 1.24 | 1.05 | 1.09 | 1.44 | 0.11 |
| maggiore | 1.56 | 1.26 | 1.61 | 1.11 | 1.36 | 0.99 | 1.88 | 0.34 |

**Partly.** 27 of the 28 season × band cells sit in 0.96–1.44 — the one outlier is maggiore's
stratified deep band at 1.88 — and the thing the axis was built for is fixed: **geneva's deep band,
1.51 mixed / 0.94 stratified under the annual scale, now reads 1.36 / 1.08.** The seasonal *spread*
is gone even where the level is not — no lake is now
mis-calibrated in opposite directions in its two seasons, which is what one annual $k$ could not
avoid.

**But the levels did not converge to 1.00**, and on the pooled number the run is slightly *further*
from 1 than `C_final` on four lakes (hallwil 1.11 → 1.32, greifensee 1.28 → 1.31, upperlugano
1.07 → 1.10, maggiore 1.52 → 1.56), flat on two and better on one (aegeri 1.30 → 1.25). Three
things account for that, and only the first two are understood:

1. **The fit is a one-shot linear solve, not a fixed point.** Raising $\mathbf{R}$ changes the
   analysis, hence the spread and the innovations it was fitted against. One pass gets close and
   stops.
2. **Maggiore is bias-dominated.** 34 % of its innovation magnitude is a persistent offset, and no
   inflation of $\mathbf{R}$ removes a bias — 1.88 in the stratified deep band is that offset, not
   a fit error. This was predicted before the run and is confirmed by it.
3. **The C_final comparison is confounded.** `C_final` ran with localization on (six of seven);
   this run has it off everywhere, so the pooled-NIS deltas above mix the season axis with
   localization. The clean pair for localization is `C_seasonal` vs `C_seasonal_noloc`, both with
   these scales — not yet scored here.

!!! note "Pooled and mean NIS diverge on hallwil"
    Pooled 1.32 against a mean-over-cells of 2.42. The pooled ratio is variance-weighted and is the
    one that inverts to a scale (§1); the gap says hallwil's miscalibration is concentrated in
    low-σ cells — the exact-zero σ_rep cells at 0.5 and 1 m that §2.3 flags. Aegeri diverges the
    other way (pooled 1.25, mean 0.84).

**Reproduce:**

```bash
python local/plotting/plot_multilake_nis.py args/experiments/C_seasonal_noloc.json \
    --run-root ~/C_seasonal_noloc --out nis.png
python local/plotting/plot_multilake_rmse_profile.py args/experiments/C_seasonal_noloc.json \
    --run-root ~/C_seasonal_noloc --metric degC --eval-obs hourly --assim-depths
```

### What the round before this changed

Three things were found and fixed after the upperlugano matrix of §4–5, each worth its own note
because none was visible on a single lake.

**The observation operator was reading the wrong depth.** Simstrat centres its grid cells at
x.25/x.75, so a round observation depth — 1 m, 10 m — falls *exactly halfway* between two cells.
That was true for **135 of 138 depths** across the seven lakes, the worst possible alignment, and
`H` was picking whichever cell won an arbitrary tie. In a thermocline that fabricates an innovation
of `gradient × 0.25 m` — up to 0.86 °C — which the filter cannot tell from a real error. Switching
`H` to interpolate between the two bracketing cells, the same thing Simstrat does when it writes its
own output, removed the losses it was causing and improved every lake.

**Thinning the observations made it worse before it made it better.** Reducing greifensee from 160
depths to 17 and aegeri from 239 to 22 was right — with 20 members the ensemble spans 19 directions,
so 160 correlated observations badly over-weight a profile. But choosing *round metres* put every
observation at the maximum offset, where the fine chains had previously averaged over a spread of
offsets. The thinning was correct; the depth grid was not.

**σ_rep depth smoothing was dropped.** It was flattening the thermocline peak and inflating the deep
tail, and once the operator was fixed it bought nothing. The tables are now the raw fit.

### Open items

**Localization is not in the final run at all.** The binary mask that broke murten in April is gone
— §2.6's Gaspari–Cohn taper is PSD and is applied to both $\mathbf{P}\mathbf{H}^{\!\top}$ and
$\mathbf{H}\mathbf{P}\mathbf{H}^{\!\top}$, which removes that failure mode rather than routing
around it. But `C_seasonal_noloc` still runs unlocalized on all seven, because that is the state the
scales were fitted in. **The one measurement missing from this page is `C_seasonal` vs
`C_seasonal_noloc`** — same scales, taper on or off — which would say what the taper is worth under
the season-keyed $\mathbf{R}$, and whether the deep water on the three Ticino/Léman lakes is better
free-running or updated. The localized run exists; it has not been scored here.

**One fit pass is not convergence.** Pooled NIS lands at 1.10–1.56, not 1.00, and §10's
self-consistency table shows the residual is systematic (every lake over-confident, none
over-damped) rather than noise. A second fit *from this run* is the obvious next step and is
legitimate now that $k \neq 1$ — the reason the loop was never closed on `C_final` (its $k$ was
fitted on an unlocalized system it did not run in) no longer applies here.

**Maggiore stays over-confident** at NIS 1.56, the only lake not close to 1, and its stratified deep
band at 1.88 is the worst cell of the 28. Its innovations are 34 % bias, and no scale on
$\mathbf{R}$ removes a bias, so the cause is upstream — most likely its 77.5 h filter window, which
also has the worst measured lag/removed trade of the seven (0.63, §2.1).

**Hallwil's exact-zero σ_rep cells.** Pooled 1.32 against mean-over-cells 2.42 points at the 0.5 and
1 m stratified cells where the estimator floored at 0 (§2.3) and $\sigma_i$ collapses to
$\sigma_\text{common}$. Neither the band split nor the season split can see them.
