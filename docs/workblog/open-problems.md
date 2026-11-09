# Open problems

*Last updated: November 2026.*

!!! abstract "In short"
    The observation side is in good shape; the open problems are on the model side. The main one is that the ensemble is too narrow, especially at depth. Most of the other problems below follow from it.

## 1. The ensemble is too narrow { #narrow-ensemble }

The problems we tried to solve with the observation uncertainty all point to an under-dispersed ensemble. In every calibration σ occupies almost the entire uncertainty budget, because the spread has little to contribute at depth ([Results](results.md#honest)). The single biggest limitation of the current setup: spread is generated only at the surface, through three forcing variables, and reaches depth only by mixing.

**What we know.** Parameter perturbation adds spread in the right place but only at amplitudes that cannot be justified ([Parameter perturbation](experiments.md#param-perturbation)). Perturbing air temperature adds nothing, particularly in winter where we were hoping to see an effect on the mixing structure ([Geneva CTD](experiments.md#geneva-ctd)).

**First answer: a learned model error.** A process noise q on the temperature profile, learned online together with the observation error R, makes the spread follow the error: spread/RMSE 0.26 → 0.98 on Upper Lugano and 0.22 → 0.76 on Ägeri (against the filtered observations), with slightly lower RMSE on both (−5 % and −8 % against the raw observations) ([Learned R and q](experiments.md#uncertainty-band), [Ägeri check](experiments.md#online-aegeri)).

**Suggested next steps.** Port it to `master` and test it on the other lakes; 7-day reforecasts to see whether the band holds at longer leads; the deep water in summer, where q drops to 0 and the band is still too narrow. Parameter spread at lower assimilation frequency remains an option.

## 2. The forcing ensemble is not centred { #uncentred-forcing }

The perturbation noise on the wind components U and V has zero mean, but the wind *speed* does not: every member is windier than the control. In the later CTD runs the ensemble-mean wind speed was 31–63 % above the control's ([Centred forcing perturbations](experiments.md#centred-forcing)), and the operational setup (`simplified_final`) uses the same uncentred perturbation. The ensemble mean therefore carries a systematic extra mixing that the daily analysis has to correct every day, and that accumulates when the observations are far apart.

**Suggested next step.** Rerun `simplified_final` with `center_perturbations` and compare RMSE, spread and NIS with the current results; repeat the Lugano deep-mixing run with centred forcing.

## 3. The summer thermocline { #thermocline }

Most places where the analysis is worse than the free run sit at the thermocline in summer ([Results](results.md#time-depth)). The error there is structural: the model misplaces the thermocline, and a forcing-generated ensemble rarely contains the right profile. Every σ variant we tried either left it unchanged or made it worse ([Experiments](experiments.md#sigma)).

The learned model error helps here first: on Ägeri the 12–14 m error drops from 0.45–0.60 °C with `simplified_final`, which at 12–12.4 m is even slightly worse than the free run, to 0.38–0.51 °C, below the free run at every level ([Ägeri check](experiments.md#online-aegeri)). The price is larger corrections at the thermocline (up to 1.4 °C on Ägeri and 2.0 °C on Upper Lugano).

**Suggested next steps.** Check whether the large corrections at the thermocline are physical or shocks (the day after each), and whether a step cap is needed; parameter spread at lower assimilation frequency.

## 4. The filter is over-cautious { #over-cautious }

With the seasonal σ the filter is systematically too cautious (NIS ≈ 0.6): it says it is less sure than it really is, and deep observations are trusted far too little. Calibrating σ fixes the consistency but not the accuracy ([Experiments](experiments.md#sigma)).

**Suggested next step.** Revisit σ once the ensemble carries more spread; σ should then not have to fill the whole budget!

This is now possible: with the [learned R and q](experiments.md#uncertainty-band) the model error is carried by the ensemble, and R takes only the misses that do not repeat from day to day. Without q (R split only) R stays physical and there is no runaway as with [adaptive σ](experiments.md#adaptive-sigma), but the filter swings to over-confident (NIS 1.76 on Ägeri). With q it is close to consistent (NIS 1.29 on Upper Lugano, 0.67 on Ägeri). Open: why Ägeri ends on the cautious side.

## 5. Lakes with surface data only { #surface-only }

Several lakes have only a surface station. Surface-only assimilation corrects the surface but damages the layers just below, between about 10 and 35 m against CTD ([Surface-only](experiments.md#surface-only)). Nothing we tried on the observation side fixed it.

**Suggested next step.** Localization limited to the mixed layer, so that surface information is not pushed below it; or, where possible, at least a few sensors over the upper 30–40 m (assimilating deeper sensor 15 m together or CTD).

## 6. Deep mixing { #deep-mixing }

Assimilating long-term CTD casts corrects the deep temperatures but makes deep mixing more frequent, 15 winters instead of 10 on Lake Lugano ([Deep mixing](experiments.md#deep-mixing)).

Part of this may come from the uncentred forcing ensemble ([problem 2](#uncentred-forcing)).

## 7. Ensemble size and evaluation period { #ensemble-size }

Twenty members is what the operational chain can afford, not a statistically defended choice. And every result is scored on 2025, the only year with profile observations on all seven lakes; the fitting-year check covers only three lakes ([Fitting year](experiments.md#fitting-year)).

**Suggested next step.** Score each new year as it becomes available, with the σ fitted on the previous ones. The learned R and q need no fitted σ; only the depth correlation of q comes from one year, and pooling more years would make it a lake property.

## 8. Physically defensible parameters { #parameters }

Estimating parameters with the filter does not converge: the parameters follow the seasons and absorb the model's seasonal bias ([Joint estimation](experiments.md#joint)).

**Suggested next step.** Slower updates; and compare the estimated distribution with a traditional calibration and learn model weaknesses.

## 9. 3D: horizontal spurious correlations { #3d }

In 3D one buoy can correct a whole region, but with 7 members spurious horizontal correlations are a real concern, and the first runs point to the need for vertical localization ([3D](experiments.md#3d)).

**Suggested next step.** Vertical and horizontal localization; assimilating the satellite surface temperature using scaling to adapt the vertical resolution.
