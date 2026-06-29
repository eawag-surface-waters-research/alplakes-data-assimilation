# Documentation: Data Assimilation Module for Simstrat

## 1-D Hydrodynamic Modelling with Data Assimilation

A couple of examples of 1-D hydrodynamic modelling that integrate DA as an essential step in their pipeline exist, showing the feasibility and multiple advantages of implementing such a module also for Simstrat. Kourzeneva (2014) provides an influential example using an extended Kalman filter (EKF)-based method to assimilate lake water surface temperature (LWST) observations into the lake model/parameterisation scheme FLake; the analysed variables are the mean water temperature (T), the bottom temperature (Tb), the mixed layer depth (h) and the shape factor (CT). Thomas et al. (2020) developed and tested a near-term iterative forecasting system that predicts the physical behaviour (hydrodynamics) of reservoirs while quantifying different sources of uncertainty and including DA as an integral process too. It explicitly partitions uncertainty into components such as model structure, parameters, and drivers, giving clearer insight into which factors most influence forecast reliability; meteorology is identified as a major source of uncertainty. Building on this framework, Wander et al. (2024) follows up by responding to the question: if initial conditions uncertainty isn't the dominant source of uncertainty, how often do you actually need to assimilate? For a 1-day-ahead forecast horizon, daily assimilation was the most skilled, while weekly data assimilation was most skilled at longer horizons (8–35 days). 3D examples are also available in the literature. These are still relevant, on one hand because we have the ambition to eventually extend our module to further models, and on the other hand because they allow us to understand how transferable processes are handled in these examples, such as the essential process of generating perturbations in forecast inputs. 

## Forcing Perturbation

Safin et al. (2022) avoid the perturbation problem altogether, driving each particle with a different member of the 21-member COSMO-E NWP ensemble; the ensemble itself represents forcing uncertainty, preserving physical cross-variable correlations without a statistical noise model. Simstrat requires a single station-based forcing time series, so we adopt the Baracchini-style (Baracchini et al., 2020) parametric perturbation approach. The forcing uncertainty has three components: station measurement error (small, negligible for MeteoSwiss stations), representativeness error (station vs. lake-mean conditions — the dominant term), and structural error in the flux computation. The spatial reanalysis provides a direct estimate of the representativeness component: averaging reanalysis grid cells over the lake gives a lake-mean time series, and the station-minus-lake-mean residuals sample the representativeness error distribution.

Perturbation statistics are derived from these residuals and we need to account for the fact that real-world forcing errors are not random in time. We generated meteorological forcing ensembles by modeling the residuals between reanalysis data (ICON) and in-situ observations for 2025. After preprocessing, including temporal alignment and transformation of wind speed and direction into Cartesian components $u = -V \sin\theta$ and $v = -V \cos\theta$, residuals were computed as $r_t = X_t^{rean} - X_t^{obs}$ for each variable $X \in \{U, V, G\}$ (zonal wind, meridional wind, global radiation). Assuming stationarity, each residual series was modeled independently as a first-order autoregressive process:

$r_t = \phi r_{t-1} + \epsilon_t$, where $\phi \approx \mathrm{corr}(r_t, r_{t-1})$ and $\epsilon_t \sim N(0, \sigma_\epsilon^2)$, with $\sigma_{\varepsilon} = \sigma_r (1 - \phi^2)^{1/2}$ chosen to preserve the observed variance.

Stochastic perturbation trajectories $\eta_t^{(m)}$ were then drawn for $m = 1, \dots, 20$ via $\eta_t^{(m)} = \phi\eta_{t-1}^{(m)} + \epsilon_t^{(m)}$ and added to the observed series to form ensemble members $X_t^{(m)} = X_t^{obs} + \eta_t^{(m)}$. Physical constraints were enforced by clipping global radiation to non-negative values $X_t^{(m)} = \max(0, X_t^{(m)})$ and suppressing perturbations during nighttime (i.e., when observed radiation is zero). Autoregressive model adequacy was assessed via autocorrelation consistency $\rho(k) = \phi^k$, residual Gaussianity, and ensemble spread diagnostics. In this first implementation cross-variable correlations are assumed negligible for this subset of variables, and spatial correlations are not considered because we use a single mean value from the lake reanalysis.

## The Theory Behind Particle Filters

The particle filter is a Monte Carlo method for solving the Bayesian filtering problem, where the goal is to sequentially estimate the posterior distribution $p(x_k \mid y_{1:k})$ of a hidden state $x_k$ given observations $y_{1:k}$. Instead of computing this distribution analytically, it is approximated by a set of $M$ weighted particles,

$$p_k(x) \approx \sum_{i=1}^M w_k^i \delta(x - x_k^i)$$, where $x_k^i$ are samples and $w_k^i$ are normalized weights. 

The algorithm alternates between a forecast (prediction) step and an analysis (update) step: particles are first propagated through the dynamical model $$x_{k+1}^i = M_{k+1}(x_k^i)$$, which approximates the prior $p(x_{k+1} \mid y_{1:k})$, and then their weights are updated using Bayes’ rule based on the likelihood of the new observation:

$$w_{k+1}^i \propto w_k^i \, p(y_{k+1} \mid x_{k+1}^i)$$, followed by normalization $\sum_{i=1}^M w_{k+1}^i = 1$. 

Over time, however, the weights tend to collapse onto a few particles (a phenomenon known as degeneracy), making the approximation inefficient; to mitigate this, a resampling step can be introduced, where particles are redrawn with probability proportional to their weights and reset to equal weights $w_{k+1}^i = 1/M$, effectively focusing computational effort on high-likelihood regions of the state space (bootstrap technique). In the limit $M \to \infty$, this sequential importance sampling scheme converges to the true Bayesian solution, making particle filters a powerful and flexible tool for nonlinear and non-Gaussian state estimation problems.

## Ensemble Kalman Filter: Details & Important Concepts

Sequential data assimilation methods (such as the Ensemble Kalman Filter) update the model state each time new observations become available. Instead of explicitly solving equations for the evolution of uncertainty, they approximate it by running an ensemble of model simulations, letting the system dynamics naturally generate error correlations. The forecast error covariance matrix P is estimated from the spread of the ensemble: its diagonal elements represent the variance (uncertainty) of temperature at each depth, while the off-diagonal elements capture how errors at different depths are correlated, reflecting physical processes like vertical mixing or stratification. When assimilating a temperature profile, the ensemble mean provides the best prior (forecast) estimate, and the covariance determines how observational information is propagated through the system. For instance, if only surface temperature is observed, the Kalman gain uses the covariance

$$K(z) = \frac{P^f(z, z_{\text{surf}})}{P^f(z_{\text{surf}}, z_{\text{surf}}) + R_{\text{surf}}}$$

to adjust subsurface layers through the analysis update

$$T^a_j(z) = T^f_j(z) + K(z)\,\bigl(y_{\text{surf}} - T^f_j(z_{\text{surf}})\bigr):$$ depths strongly correlated with the surface are updated more, while weakly correlated layers remain largely unchanged. After updating all ensemble members, the analysis mean represents the posterior temperature profile, optimally combining model and observations, and the updated ensemble spread reflects the remaining uncertainty. However, this approach is computationally expensive because the forecast model must be run for each ensemble member, and small ensemble sizes can lead to sampling errors and an underrepresentation of the true error covariance, degrading assimilation performance. It's not the type of perturbation that matters, but whether the ensemble represents realistic uncertainty in the system. Perturbing forcing (e.g., wind, radiation, precipitation) is popular because it injects variability in a physically meaningful, realistic-error way (errors propagate naturally through dynamics) and it creates flow-dependent errors (correctly correlated errors). So if the ensemble correctly samples the true uncertainty distribution, the method works regardless of how you generated it.

## Assimilation Strategy

For the MVP we aim to first use a particle-filter-type assimilation process. We start with particle filters because of the straight forward implementation, overall simplicity, and because it allows to assimilate the observed data without touching the model states of Simstrat that are stored in FORTRAN's binary format, which in the MVP results in a cleamner implementation. This method also removes the chance of producing unphisycal steps and in most cases avoids "jumps" in the temperature timeseries. For the theorethical background refer above. Conceptually N ensemble members are propagated forward, each driven by an independently perturbed forcing time series. Perturbations f(t) are based on statistics (standard deviation, autocorrelation) derived from the lake-mean reanalysis minus station residuals. At each observation time, the best-performing particle (or a set of likely particles) is identified and its state copied to all other particles (or states resampled based on likelyhood), which then continue to forecast with their own independent perturbed forcing sequences. The winning perturbations could be logged to diagnose any systematic forcing correction. 

A second implementation replaces the best-particle selection/survival of the fittest with an ensemble Kalman filter (see above for more details) update: the ensemble cross-covariance is used to compute a corrected state from all members, which is then propagated to the next observation cycle. Assimilation can essentially be applied to model parameters, inputs, forcing, or state — which can also be modified jointly. For our case the Simstrat model is already calibrated for Lake Lugano, so we see little reason to start updating calibrated parameters for now. Updating the forcing instead of the state can provide multiple advantages: physical consistency is ensured, and error attribution is more fitting. On the other hand, state correction is more agnostic about the error source, can better compensate for any kind of model deficiency, and has been shown to produce much more significant improvements compared to updating only parameters and forcing. A negative of the latter could be the generation of shocks in the temperature time series, which however can be limited by frequent assimilation of observations and by keeping the corrections small. Given the scarcity of published EnKF-based forcing correction examples for lakes and oceans, however, we need to consider the state update as the more standard strategy forward.

## Filtering Internal Wave 

The assimilation of observed lake temperature profiles into one-dimensional hydrodynamic lake models such as Simstrat can be challenged by unresolved sub-daily variability in the water column. In particular, temperature observations near and below the thermocline frequently exhibit short-term oscillations caused by internal waves and basin-scale seiches. These processes induce rapid vertical displacements of isotherms, leading to strong temporal fluctuations in measured temperatures at fixed depths.

Because internal wave dynamics are not represented in Simstrat, the model cannot reproduce these observed sub-daily oscillatory behaviors in temperature profiles. As a result, direct assimilation of high-frequency temperature observations may introduce inconsistencies between the model state and the observations. This mismatch can destabilize the assimilation procedure, generate spurious corrections, and reduce the reliability of the assimilated results

Therefore, a key challenge is to develop assimilation strategies that account for unresolved internal wave variability while preserving the physical consistency and numerical stability of the lake model. To address this issue, we develop an adaptive low-pass filter whose window size varies both in depth and time as a function of stratification strength. The filter window at each depth and timestep is defined as the sum of two components: a baseline temporal smoothing term and an additional adaptive component that increases in regions and periods of strong stratification, where internal wave activity is expected to be more pronounced.

**Step 0 — Gradient smoothing**

Raw local gradients are first smoothed with a causal `GRAD_SMOOTH_H`-hour trailing mean (parameter $W_s$) to reduce noise before driving the window:

```math
\bar{g}(z,t) = \frac{1}{W_s}\int_{t-W_s}^{t} \left|\frac{\partial T}{\partial z}(z,\tau)\right| d\tau
```

Depths shallower than `THERMO_DEPTH_MIN` are set to zero (surface layer dominated by solar heating, not internal waves).

**Component 1 — Gradient-driven (thermocline)**

```math
W_\text{grad}(z,t) = \text{clip}\!\left(\frac{W_\text{MAX} \cdot \bar{g}(z,t)}{G_\text{MAX}},\ W_\text{MIN},\ W_\text{MAX}\right)
```

`G_MAX` is the gradient value that maps to `W_MAX` (default: 95th percentile of $\bar{g}$ across thermocline depths, auto-computed).

**Component 2 — Depth floor (below thermocline)**

```math
W_\text{floor}(z,t) = \text{clip}\!\left(\frac{z - z_{tc}(t)}{\max\!\left(D_\text{ref} - z_{tc}(t),\; 1\right)},\ 0,\ 1\right) \cdot (W_\text{DEEP} - W_\text{MIN})
```

where $z_{tc}(t) = \arg\max_z \bar{g}(z,t)$ is the time-varying thermocline depth (depth of peak smoothed gradient). This component is zero when the peak gradient falls below `THERMO_GRAD_MIN` (no active stratification, e.g. winter). The $\max(\cdot, 1)$ in the denominator guards against division by zero when $z_{tc} \geq D_\text{ref}$.

**Final window**

```math
W(z,t) = \text{clip}\!\left(W_\text{grad}(z,t) + W_\text{floor}(z,t),\ W_\text{MIN},\ W_\text{MAX}\right)
```

| Zone | Dominant component | Typical window |
|---|---|---|
| $z <$ `THERMO_DEPTH_MIN` | none ($\bar{g} = 0$ forced) | $W_\text{MIN}$ |
| thermocline | $W_\text{grad}$ | up to $W_\text{MAX}$ |
| below thermocline | $W_\text{floor}$ ramps with depth | $W_\text{MIN}$ → $W_\text{DEEP}$ |

Applied as a causal trailing box filter (no lookahead, online-compatible).

![Adaptive Filter](../images/adaptive_filter.png)

raw hourly temperature in thin transparent blue against the adaptively
filtered signal in red. The gap between the two lines represents the high-frequency variability the filter removed at
that depth — narrow near the surface (short window, little smoothing), wider near and below the thermocline (longer
window, more aggressive smoothing). The last plot makes the seasonal behaviour of the filter directly visible — windows grow during summer
stratification (strong thermocline gradient activates both the gradient-driven and depth-floor components) and collapse
toward W_MIN in winter when the water column is well-mixed.

We provide this filtering approach as an optional component that can be integrated into the assimilation workflow of alplakes_da. At the current stage, the results do not show a substantial improvement from the filtering procedure, likely because the assimilation already applies averaged corrections over a one-day window, which partially mitigates the impact of sub-daily oscillations.

Nevertheless, this implementation serves as a proof of concept and demonstrates how adaptive filtering strategies can be flexibly incorporated within the framework. We consider this filtering step a promising direction for future research, particularly for higher-frequency assimilation setups or applications where unresolved internal wave variability has a stronger impact. However, due to project priorities and time constraints, further development and evaluation of the approach are left for future work.

## Extension to 3D Models

When expanding our module to include 3D models, a couple of considerations are needed. First, there is a need for spatio-temporal forcing perturbation accounting both for spatial and temporal correlations. Second, observation location matters much more: in 3D, an observation at one location primarily constrains the state near that location, and localization becomes essential to avoid spurious correlations and keep computational costs low. Finally, ensemble size becomes a serious constraint — examples in 3D use around 20 ensemble members, while for 1D they use up to 400 because the computational load is small.

## References

Baracchini, T., Chu, P. Y., ˇSukys, J., Lieberherr, G., Wunderle, S., W¨uest, A., and Bouffard, D.
(2020). Data assimilation of in situ and satellite remote sensing data to 3d hydrodynamic lake
models: a case study using delft3d-flow v4. 03 and openda v2. 4. Geoscientific Model Development,
13(3):1267–1284.

Kourzeneva, E. (2014). Assimilation of lake water surface temperature observations using an extended
kalman filter. Tellus A: Dynamic Meteorology and Oceanography, 66(1):21510.

Safin, A., Bouffard, D., Ozdemir, F., Ramon, C. L., Runnalls, J., Georgatos, F., Minaudo, C., and
ˇSukys, J. (2022). A bayesian data assimilation framework for lake 3d hydrodynamic models with a
physics-preserving particle filtering method using spux-mitgcm v1. Geoscientific Model Development,
15(20):7715–7730.

Thomas, R. Q., Figueiredo, R. J., Daneshmand, V., Bookout, B. J., Puckett, L. K., and Carey, C. C.
(2020). A near-term iterative forecasting system successfully predicts reservoir hydrodynamics and
partitions uncertainty in real time. Water Resources Research, 56(11):e2019WR026138.

Wander, H. L., Thomas, R. Q., Moore, T. N., Lofton, M. E., Breef-Pilz, A., and Carey, C. C. (2024).
Data assimilation experiments inform monitoring needs for near-term ecological forecasts in a eutrophic
reservoir. Ecosphere, 15(2):e4752.
