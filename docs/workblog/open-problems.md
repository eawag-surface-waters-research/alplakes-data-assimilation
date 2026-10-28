# Open problems

*Last updated: October 2026.*

!!! abstract "In short"
    The observation side is in good shape; the open problems are on the model side. The main one is that the ensemble is too narrow, especially at depth. Most of the other problems below follow from it.

## 1. The ensemble is too narrow { #narrow-ensemble }

The problems we tried to solve with the observation uncertainty all point to an under-dispersed ensemble. In every calibration σ occupies almost the entire uncertainty budget, because the spread has little to contribute at depth ([Results](results.md#honest)). The single biggest limitation of the current setup: spread is generated only at the surface, through three forcing variables, and reaches depth only by mixing.

**What we know.** Parameter perturbation adds spread in the right place but only at amplitudes that cannot be justified ([Parameter perturbation](experiments.md#param-perturbation)). Perturbing air temperature adds nothing, particularly in winter where we were hoping to see an effect on the mixing structure ([Geneva CTD](experiments.md#geneva-ctd)).

**Suggested next steps.** Parameter spread at lower assimilation frequency, where it already looks promising; justified random variability on the model state itself (process noise, see [Forecast uncertainty band](experiments.md#uncertainty-band)).

## 2. The forcing ensemble is not centred { #uncentred-forcing }

The perturbation noise on the wind components U and V has zero mean, but the wind *speed* does not: every member is windier than the control. In the later CTD runs the ensemble-mean wind speed was 31–63 % above the control's ([Centred forcing perturbations](experiments.md#centred-forcing)), and the operational setup (`simplified_final`) uses the same uncentred perturbation. The ensemble mean therefore carries a systematic extra mixing that the daily analysis has to correct every day, and that accumulates when the observations are far apart.

**Suggested next step.** Rerun `simplified_final` with `center_perturbations` and compare RMSE, spread and NIS with the current results; repeat the Lugano deep-mixing run with centred forcing.

## 3. The summer thermocline { #thermocline }

Most places where the analysis is worse than the free run sit at the thermocline in summer ([Results](results.md#time-depth)). The error there is structural: the model misplaces the thermocline, and a forcing-generated ensemble rarely contains the right profile. Every σ variant we tried either left it unchanged or made it worse ([Experiments](experiments.md#sigma)).

**Suggested next steps.** Parameter spread at lower assimilation frequency, where it already looks promising; justified random variability on the model state itself (process noise, see [Forecast uncertainty band](experiments.md#uncertainty-band)).

## 4. The filter is over-cautious { #over-cautious }

With the seasonal σ the filter is systematically too cautious (NIS ≈ 0.6): it says it is less sure than it really is, and deep observations are trusted far too little. Calibrating σ fixes the consistency but not the accuracy ([Experiments](experiments.md#sigma)).

**Suggested next step.** Revisit σ once the ensemble carries more spread; σ should then not have to fill the whole budget!

A cheaper intermediate step is the adaptive σ with safeguards: bounded between the 0.05 °C floor and the seasonal value, so that it can only trust the observations *more* than the seasonal σ does, never less. That would keep its gain where the seasonal σ is too cautious and stop the runaway seen on Ägeri ([Adaptive σ](experiments.md#adaptive-sigma)). Not tested yet.

## 5. Lakes with surface data only { #surface-only }

Several lakes have only a surface station. Surface-only assimilation corrects the surface but damages the layers just below, between about 10 and 35 m against CTD ([Surface-only](experiments.md#surface-only)). Nothing we tried on the observation side fixed it.

**Suggested next step.** Localization limited to the mixed layer, so that surface information is not pushed below it; or, where possible, at least a few sensors over the upper 30–40 m (assimilating deeper sensor 15 m together or CTD).

## 6. Deep mixing { #deep-mixing }

Assimilating long-term CTD casts corrects the deep temperatures but makes deep mixing more frequent, 15 winters instead of 10 on Lake Lugano ([Deep mixing](experiments.md#deep-mixing)).

Part of this may come from the uncentred forcing ensemble ([problem 2](#uncentred-forcing)).

## 7. Ensemble size and evaluation period { #ensemble-size }

Twenty members is what the operational chain can afford, not a statistically defended choice. And every result is scored on 2025, the only year with profile observations on all seven lakes; the fitting-year check covers only three lakes ([Fitting year](experiments.md#fitting-year)).

**Suggested next step.** Score each new year as it becomes available, with the σ fitted on the previous ones.

## 8. Physically defensible parameters { #parameters }

Estimating parameters with the filter does not converge: the parameters follow the seasons and absorb the model's seasonal bias ([Joint estimation](experiments.md#joint)).

**Suggested next step.** Slower updates; and compare the estimated distribution with a traditional calibration and learn model weaknesses.

## 9. 3D: horizontal spurious correlations { #3d }

In 3D one buoy can correct a whole region, but with 7 members spurious horizontal correlations are a real concern, and the first runs point to the need for vertical localization ([3D](experiments.md#3d)).

**Suggested next step.** Vertical and horizontal localization; assimilating the satellite surface temperature using scaling to adapt the vertical resolution.
