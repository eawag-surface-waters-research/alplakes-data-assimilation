## Literature

**Baracchini, T., Chu, P.Y., Šukys, J., Lieberherr, G., Wunderle, S., Wüest, A., Bouffard, D. (2020). *Data assimilation of in situ and satellite remote sensing data to 3D hydrodynamic lake models: a case study using Delft3D-FLOW and OpenDA*** — EnKF with Delft3D; Lake Geneva demonstration.   
 PDF / article: [https://doi.org/10.5194/gmd-13-1267-2020](https://doi.org/10.5194/gmd-13-1267-2020) (GMD). [GMD+1](https://gmd.copernicus.org/articles/13/1267/2020/?utm_source=chatgpt.com) 

**Important paper # 1: Similarity and approach**

**Brief Summary:**

The paper develops a flexible data assimilation (DA) framework that combines in situ observations, satellite remote sensing, and 3D hydrodynamic numerical simulations to resolve the wide range of spatiotemporal scales involved in lake dynamics. The case study is Lake Geneva, one of the largest freshwater lakes in western Europe. Using an ensemble Kalman filter (EnKF), the approach accounts for both model and observational uncertainties, assimilating in situ temperature profiles and AVHRR (Advanced Very High Resolution Radiometer) lake surface water temperature (LSWT) data into the 3D Delft3D-FLOW model. The open-source DA platform OpenDA serves as the integration environment for the assimilation. Results show DA effectively improved model performance across a broad range of spatiotemporal scales and physical processes, reducing overall temperature errors by 54%. Specific improvements included:
•	Better representation of upwelling events 
•	Improved thermocline structure and mixed layer depth throughout the water column.
•	Physically coherent updates even in areas with missing satellite coverage, due to the covariance propagation of the EnKF.
•	Better capture of summer LSWT variability that the control run missed.
With a localization scheme, an ensemble size of 20 members was found sufficient to derive covariance matrices yielding satisfactory results. This is notable for computational feasibility in near-real-time operational systems. The entire framework was developed with near-real-time operational lake monitoring in mind, including integration into a platform (meteolakes.ch). 

**takeaway: big improvements 3D lake modelling of large lake and its processes through assimilation in complex 3D model where large biases exist using an ensemble method to integrate in situ and satellite data in an operational ready fashion. Perturbation of forcing and updating of the temperature.**

**Method:** Data assimilation framework for 3D hydrodynamic lake models by coupling Delft3D-FLOW v4.03 (z-layer, 100 vertical levels, 450 m horizontal grid, κ-ε turbulence closure) with OpenDA v2.4 through a new file-based black-box wrapper, and applied it to Lake Geneva for the year 2017. An Ensemble Kalman Filter (EnKF) updates the 3D temperature state only; stochasticity is injected not into the state but into the wind forcing (u and v components), using OpenDA's spatiotemporally correlated noise model whose standard deviation and correlation scales were derived from the COSMO-E ensemble reanalysis. Two observation streams were assimilated: 31 in-situ vertical temperature profiles at stations SHL2 and GE3, and 128 quality-filtered AVHRR satellite LSWT images (of 3372 available). Observation-error variances were set from instrument precision plus a 48-hour temporal-variability window for in-situ data, and a 1 °C threshold for AVHRR. A Gaspari–Cohn localization with a 15 km cutoff suppressed spurious long-range covariances, and an ensemble-size sensitivity study was performed.

**Major results:** DA reduced overall temperature RMSE by 54 % and MAE by 60 % relative to the calibrated control run. Improvements were consistent across surface and deepwater, eliminating a warm bias in the 5–25 m mixed-layer and correcting thermocline depth. Spatial structures; gyres, complex thermal gradients, and transient upwellings; were preserved and sharpened; a September upwelling underestimated by 5 °C in the control was reduced to a 2.5 °C error after DA. Crucially, 20 ensemble members were sufficient thanks to localization, making the approach viable for near-real-time operational forecasting (e.g., meteolakes.ch). 


**Kourzeneva, E. (2014). *Assimilation of lake water surface temperature observations using an Extended Kalman Filter* (Tellus A)** — EKF into FLake (1-D parameterisation); satellite \+ in situ LSWT assimilation.   
 Article / PDF: [https://a.tellusjournals.se/articles/10.3402/tellusa.v66.21510](https://a.tellusjournals.se/articles/10.3402/tellusa.v66.21510?utm_source=chatgpt.com) . [a.tellusjournals.se+1](https://a.tellusjournals.se/articles/10.3402/tellusa.v66.21510?utm_source=chatgpt.com) 

**Important paper # 2: influential**

The analysed variables are the mean water temperature (T), the bottom temperature (Tb), the mixed layer depth (h) and the shape factor (CT). EKF: numerically calculated corrections.

**Brief Summary:**

This paper develops a new extended Kalman filter (EKF)-based method to assimilate lake water surface temperature (LWST) observations into the lake model/parameterisation scheme FLake (Freshwater Lake), and implements it into the stand-alone offline version of FLake. FLake is widely used in numerical weather prediction (NWP) and climate modelling, and is included in operational NWP runs at some national weather service centres. The mixed and non-mixed regimes in lakes are treated separately by the EKF algorithm. The timing of the ice period is indicated implicitly: no ice is assumed if the water surface temperature is being measured. Numerical experiments are performed using operational in situ observations for 27 lakes and merged observations (in situ plus satellite) for 4 lakes in Finland. This was an early and influential paper in lake data assimilation. As Baracchini et al. noted, Kourzeneva (2014) used an EKF to assimilate lake surface water temperature into a one-dimensional two-layer freshwater lake model, leading to significant improvements over the free model run. A key finding echoed in subsequent work is that spring–early summer observations play a key role in improving model performance during the warming period, with implications for water quality modelling and phytoplankton bloom prediction.

**takeaway: big improvements in NWP representation of lakes through assimilation in simple 1D model where large biases are known using a non-ensemble method to integrate in situ and satellite data. Perturbation of states but no ensemble!**

**Methods: ** The study developed a new Extended Kalman Filter (EKF) algorithm to assimilate lake water surface temperature (LWST) observations into the FLake lake parameterisation scheme, implemented in its stand-alone offline version. FLake is a self-similarity-based two-layer model with up to 12 prognostic variables (mixed-layer temperature and depth, bottom temperature, thermocline shape factor, ice/snow thickness and temperature, etc.), which makes the control vector small enough for EKF to be computationally tractable; a key advantage over ensemble methods for NWP applications. Because the mixed and non-mixed (stratified) regimes have different active state variables, they are treated separately by the EKF, and the algorithm is re-initialised (B matrix reset) whenever the regime switches, to avoid divergence from nonlinearity jumps across regimes.Ice timing is handled implicitly: if a surface temperature is observed, the lake is assumed ice-free. Jacobians of the model operator M and observation operator H are computed by finite differences, requiring one perturbed offline FLake run per control variable, with perturbation sizes chosen by preliminary sensitivity tests. Experiments were run for 27 Finnish lakes using SYKE in-situ LWST (EKF-S) and for 4 lakes using merged SYKE + MODIS satellite data (EKF-M), compared against a free run (FR) initialised from a typical late-autumn mixed profile.

**Major results:** The free run suffered from a well-known FLake warm bias, overestimating summer mixed-layer temperature by up to 5 °C (e.g., Lake Inarijärvi); the EKF-S experiment brought TML much closer to observations across all 27 lakes. Cross-validation with the merged dataset confirmed that satellite LWST adds useful information where in-situ coverage is sparse. Impact on autumn/winter ice and snow thickness was negligible because SYKE observations are unavailable just before ice onset, so the filter falls back to the free-run background; in spring the free run produced ice break-up dates that were too late, and early-spring observations were identified as particularly valuable for correcting the subsequent warming trajectory. The paper highlighted practical challenges — regime-switch handling, Jacobian perturbation tuning, and observation gaps around freeze-up — as the main obstacles to operational deployment in NWP systems like HIRLAM.

**Safin, A., et al. (2022). *A Bayesian data assimilation framework for lake 3-D hydrodynamic models (SPUX–MITgcm)*** — physics-preserving particle filtering; 3-D MITgcm; Lake Geneva.   
 PDF / article: [https://gmd.copernicus.org/articles/15/7715/2022/](https://gmd.copernicus.org/articles/15/7715/2022/) . [GMD](https://gmd.copernicus.org/articles/15/7715/?utm_source=chatgpt.com)[Research Collection](https://www.research-collection.ethz.ch/bitstreams/78b1a39d-1ba6-4747-8760-c03d10fe43bd/download?utm_source=chatgpt.com) 

**Comment: Framework example, sophisticated method sampling, focus primarely on satellite data. Possibly we could take as an inspiration when starting to handle satellite images to increase the satellite data use and skin to bulk conversion and when dealing with the 3D model since kind of continues from where Baracchini left + Handling of the selected trajectories as new inputs! The updates are sampled using a EMCEE (Monte Carlo type) algorithm: evolving walkers + Particle filter trajectories for each updated parameter (stochasticity of weather) + bootstrapping all particles are resampled (bootstrapped) according to their observational likelihoods. Model states are deleted and replaced by some other state from a different trajectory. Limited ability, model states are not modified, parameter-forcing trajectories sampled. Similar filtering of forcing!**

**Brief Summary:**

This paper presents a Bayesian inference framework for a 3D hydrodynamic model of Lake Geneva, combining stochastic weather forcing with high-frequency observational datasets. It couples a Bayesian inference package (SPUX) with the hydrodynamics package MITgcm into a single framework, SPUX-MITgcm.  The paper explicitly positions itself as a methodological advance over the earlier EnKF-based work. It notes that while the ensemble Kalman filter used by Baracchini et al. achieved a 54 % temperature error reduction, due to the limitation of that assimilation scheme, only about 3.7 % of available LSWT images were used. The new framework aims to exploit a much larger fraction of the satellite data. Framework: 
1) Parameter inference via EMCEE: The model relies on the ensemble affine invariant sampler (EMCEE) to calibrate distributions of physical model parameters — particularly well suited for nonlinear parameters — providing a more informative and accurate parameter estimation than standard inference methods, albeit at higher computational expense. 
2) Physics-preserving particle filter: To increase confidence in the sampling algorithm, a particle filter method provides trajectories consistent with the hydrodynamic model, where intermediate model state posteriors are resampled in accordance with their respective observational likelihoods. Importantly, the filter does not modify model states (they are only deleted or replicated), so predictions do not exhibit the shocks generated by some DA schemes. 
3) BiLSTM neural network for bulk-to-skin temperature conversion: A bi-directional long short-term memory (BiLSTM) neural network is developed to estimate lake skin temperature from a 27-hour history of hydrodynamic bulk temperature predictions and atmospheric data, also quantifying associated uncertainty. This is necessary because AVHRR measures skin temperature while hydrodynamic models predict bulk temperature — a mismatch that prior studies handled with restrictive quality filtering.
The DA improvements were more modest than in Baracchini et al.: the overall improvement in RMSE and MAE across the various datasets was 4–15 %. However, the framework used 798 AVHRR images (compared to ~124 in the Baracchini study), and did so without requiring manual image-by-image quality thresholding. The BiLSTM network achieved a 33 % reduction in RMSE for the test set, though in the assimilation run, it increased RMSE by about 10 %, most likely due to differences between the training data and the assimilation process.
The authors are candid about the method's trade-offs: the particle filter provides a relatively small improvement to model predictions in contrast to other popular DA schemes, but at no cost to the quality of the physical model. However, this approach requires a highly robust hydrodynamic model, as its corrective powers are limited. The approach is also quite computationally costly; simulations ran at the Swiss National Supercomputing Center over approximately 3 months. This paper sits neatly between Baracchini et al. (2020) and the broader literature: it swaps Gaussian-assumption-based EnKF for a fully Bayesian particle MCMC approach — gaining physics consistency and non-Gaussian flexibility, but trading some correction power and incurring far greater computational cost. The authors suggest that a cheaper parameter optimization method combined with an improved particle filter would be the more productive path forward for operational use.

The particle filter improvement over the control run is small.
The method is extremely computationally expensive
Use the COSMO-E ensemble members directly as forcing realizations
The BiLSTM bulk-to-skin observation operator: accepts nearly all quality-flagged pixels and let the BiLSTM network figure out how much to trust each one at assimilation time
Particle filtering degenerates badly with dense, informative observations

**Anderson, J., Collins, S., et al. (various). *Operational ensemble DA for short-term lake forecasts (FLAREr toolset)*** — open-source R tools (FLAREr) and implementations for ensemble DA in lake forecasting.   
 FLAREr docs / code: [https://flare-forecast.org/FLAREr/](https://flare-forecast.org/FLAREr/?utm_source=chatgpt.com) . [FLARE](https://flare-forecast.org/FLAREr/?utm_source=chatgpt.com) 

**Important paper # 3:** 

Updates Temperature + model parameters using EnKF. 441! ensembles built in a non very clear manner...

FLAREr is very interesting. Many papers demonstrating the use. Good for inspiration.
example paper where the framework was first presented): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR026138
The study “A Near-Term Iterative Forecasting System Successfully Predicts Reservoir Hydrodynamics and Partitions Uncertainty in Real Time” by R. Quinn Thomas and colleagues presents a data-driven, real-time forecasting system that continuously updates and improves predictions of reservoir hydrodynamics.

**Brief Summary:**
Objective: To develop and test a near-term iterative forecasting system that predicts the physical behavior (hydrodynamics) of reservoirs while quantifying different sources of uncertainty.
Approach: The system combines data assimilation (regularly incorporating new measurements) with iterative model updating, allowing forecasts to adjust in real time as new data arrive.
Key innovation: It explicitly partitions uncertainty into components such as model structure, parameters, and drivers (e.g., weather inputs), giving clearer insight into which factors most influence forecast reliability.
Results: The method effectively improved accuracy in predicting reservoir temperature and water column dynamics over various time horizons.
Implications: This framework can be scaled for use in environmental management, water quality forecasting, and adaptive reservoir operations — offering a generalizable approach for ecological forecasting systems.

**Architecture**

FLARE couples three components into a daily iterative cycle:

In-situ sensors — a thermistor chain plus an inflow discharge sensor stream data continuously to the cloud, where they're picked up daily by the workflow.

General Lake Model (GLM) — a 1D process-based hydrodynamic model that runs the forward simulation, driven by NOAA Global Ensemble Forecast System (GEFS) 16-day meteorological forecasts (21 members).

Ensemble Kalman Filter (EnKF) — runs at each daily timestep to update both the model state (water temperature profile) and three sensitivity-selected GLM parameters (a longwave radiation scaling factor and two sediment temperature parameters) using the most recent observations. State augmentation is used so the filter estimates parameters and states jointly.

**Uncertainty partitioning:** the key methodological contribution. This is what the paper's title emphasizes and what makes it foundational. FLARE represents and propagates five distinct sources of uncertainty through the ensemble, and it can quantify each one's contribution by selectively turning sources on and off:

Driver (meteorological) uncertainty: each FLARE ensemble member is paired with one of the 21 NOAA GEFS members, propagating weather forecast spread directly into the lake model.

Initial condition uncertainty: spread across ensemble members on day 0 of each forecast, set either by the previous day's forecast or by the post-DA analysis.

Process uncertainty: random Gaussian noise added to water temperature predictions at each daily timestep, spatially correlated across depths.

Parameter uncertainty: each member draws from distributions for the three EnKF-tuned parameters; the spread across members represents how well-constrained each parameter is.

Observation uncertainty: the sensor measurement error feeds into the EnKF's R matrix and determines how strongly the filter trusts each observation.

**Major results: ** 

FLARE successfully produced skillful 16-day water temperature forecasts in real time at FCR. Forecast skill degraded with horizon as expected, but stayed within useful bounds across the 16-day window for most depths. The uncertainty partitioning revealed the headline finding that drove all subsequent FLARE work: process uncertainty and meteorological driver uncertainty dominated total forecast variance, with initial condition uncertainty playing only a minor role except at the very shortest horizons. Parameter uncertainty contributed measurably but less than process and driver. This finding is the opposite of the situation in numerical weather prediction, where IC uncertainty dominates and frequent DA always pays off. It set up the question that Wander et al. (2024) later answered systematically: if IC uncertainty isn't the dominant source, how often do you actually need to assimilate? The Thomas paper also showed that EnKF-based parameter tuning was essential — running FLARE with constant parameters (only updating initial conditions) substantially degraded skill, because GLM out of the box is not well-calibrated to FCR's specific dynamics.

**Interesting: 1D model framework, ensemble sizes for computationally light simulations ~ 400, uncertainity partitioning, various considerations useful for us.**

**Thomas, S.M., et al. (2020). *Data assimilation experiments inform monitoring needs for near-term ecological forecasts in a eutrophic reservoir (FLARE system)*** — FLARE forecasting system; ensemble DA for water temperature and short-term forecasts.   
 Article: [https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.4752](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.4752?utm_source=chatgpt.com) . [ESAJournals](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.4752?utm_source=chatgpt.com)[VTechWorks](https://vtechworks.lib.vt.edu/bitstream/10919/104566/1/Advancing%20lake%20and%20reservoir%20water%20quality%20management%20with%20near%20term%20iterative%20ecological%20forecasting.pdf?utm_source=chatgpt.com) 

Authors got allucinated: correct ones Heather L. Wander et. al. 2024

**Important paper # 4: Practical dimension, focusing on the needs, application of FLARE**

**Brief Summary:**
Many forecasting systems have been developed using high temporal frequency (minute to hourly resolution) data streams for assimilation, but this approach may be cost-prohibitive or impossible for variables that lack high-frequency sensors or have high data latency. Rather than asking how to do DA, the paper asks a more practical question: how often do you actually need to assimilate data to produce skillful forecasts? Starting in June 2020, real-time water column data were recorded in Beaverdam Reservoir, Virginia. Multiple temperature sensors were deployed at 1 m intervals from the surface to the sediment, and a multi-parameter sonde monitored water temperature at 1.5 m at the deepest site. Sensors collected data every 10 minutes, which were then assimilated into the FLARE (Forecasting Lake And Reservoir Ecosystems) system at different rates. DA experiments: The data assimilation frequencies tested were daily, weekly, fortnightly, or monthly, and the forecast horizons ranged from 1-, 7-, and 35-day-ahead forecasts. Observations were selectively withheld to simulate lower-frequency monitoring scenarios, allowing a clean comparison of DA frequency effects. What assimilation frequency produces the most skilful forecasts; how skill varies across depth and season (mixed vs. stratified); and how DA frequency influences total forecast uncertainty and the contribution of initial condition uncertainty. For a 1-day-ahead forecast horizon, daily assimilation was the most skilled. Weekly data assimilation was most skilled at longer horizons (8–35 days). Overall, the study notes a trend of lower-frequency data assimilation outperforming daily assimilation as the forecast horizon increased. The study concludes that weekly water temperature observations are likely "good enough" to set up a skillful forecasting system for many management applications, while daily assimilation would be most useful for applications requiring high forecast accuracy in deeper waters or at shorter forecast horizons. Where other studies focused on physical limnology and 3D model state correction for large, deep lakes, Wander et al. operate in the ecological forecasting tradition — using a 1D process model (FLARE), targeting a small eutrophic drinking water reservoir, and asking practical monitoring design questions relevant to water managers. Key insight: you don't necessarily need high-frequency data streams to produce useful forecasts, and that the optimal frequency depends on the forecast horizon. This has direct implications for sensor deployment decisions and monitoring costs.

**Method:** FLARE applied to Beaverdam Reservoir (BVR), a small (0.28 km²), shallow (11 m), dimictic, eutrophic drinking-water reservoir in southwestern Virginia, for the full year 2021. The 1D General Lake Model (GLM v3.3.0) was driven by NOAA GEFS 1–35-day meteorological forecasts (30 ensemble members) and forecasts were generated at 0.5 m vertical intervals from 0.1 to 11 m. The EnKF used 256 ensemble members to avoid spurious correlations, with state augmentation tuning three sensitivity-selected parameters: a longwave radiation scaling factor and epilimnetic and hypolimnetic sediment temperature parameters. Process noise SD was set to 0.75 °C, observation noise SD to 0.1 °C. A 35-day spin-up period (Nov–Dec 2020) calibrated parameters before the focal year. Skill was evaluated against withheld sensor observations using RMSE (with a 2 °C "skillful" threshold) and CRPS, separately for surface (0–2.5 m), middle (2.6–8.4 m), and bottom (8.5–11 m) layers, and for the mixed (118 days) vs. stratified (241 days) periods.

**Results:**
**Q1 — Which DA frequency gives the most skillful forecasts?** Aggregated over the year, daily DA wins at 1–7-day horizons, but weekly DA wins at 8–35-day horizons, and weekly is the best overall across the largest number of horizons. Daily DA actually crosses the 2 °C unskillful threshold by day 28, while weekly, fortnightly, and monthly never do. The mechanism: daily DA causes parameter overfitting — daily updates pushed the longwave and sediment temperature parameters to noisier, more variable trajectories that diverged from the more stable values reached under weekly/fortnightly/monthly DA. A control experiment with constant (untuned) parameters but daily initial-condition updates was substantially worse (mean RMSE 3.12 °C vs. 1.95 °C), confirming that parameter tuning matters; just not too often.

**Q2 — How does skill vary across space (depth) and time (mixed vs. stratified)?** Overall mean RMSE was 1.50 °C, with 1-day-ahead 0.81 °C, 7-day 1.15 °C, and 35-day 1.94 °C. Bottom-water forecasts were consistently the most skillful (1.13 °C aggregated) because hypolimnetic temperatures change slowly under stratification. Surface forecasts were the worst (1.78 °C) because they respond rapidly to meteorological forcing. The stratified period was slightly more skillful than the mixed period (1.43 vs. 1.56 °C aggregated), again because day-to-day variability is lower once stratification sets in. The crossover horizon at which weekly DA overtook daily DA depended on depth: >5 days at the surface, but >8 days at mid- and bottom depths.

**Q3 — How does DA frequency affect total uncertainty, and what fraction comes from initial conditions?** This is the conceptually important question. Total forecast variance was largest under monthly DA and smallest under daily DA at the 1-day horizon (2.38 °C vs. 0.60 °C), but the curves converged by the end of the 35-day horizon, meaning the benefit of frequent DA is concentrated at short horizons. To quantify the role of initial conditions specifically, they re-ran the forecasts with and without initial-condition uncertainty included and compared variances. The result: under daily DA, initial-condition uncertainty contributed only 0.01% of total uncertainty at the 1-day horizon, essentially negligible. Under weekly/fortnightly/monthly DA at 1 day, it contributed 54–71%. By day 10 it dropped below 1% across all DA frequencies for the entire mixed period and for surface depths in the stratified period; only mid- and bottom-water stratified forecasts retained ~11% IC uncertainty out to 10–20 days. The implication is fundamental: for this system, initial conditions are not the dominant source of uncertainty at most horizons; process, parameter, and driver uncertainty must dominate. This is the opposite of weather forecasting, where IC uncertainty dominates and frequent DA always helps. It explains why weekly DA can match or beat daily DA: there is simply not much IC uncertainty left to remove.

**Uncertainty partitioning approach (the four sources)** FLARE represents uncertainty through the 256-member ensemble across four channels: (1) driver uncertainty — each FLARE member is paired with one of the 30 NOAA GEFS meteorological ensemble members; (2) initial-condition uncertainty — spread across the 256 members on day 0 of each forecast, set either by the previous day's forecast or by the post-DA analysis; (3) process uncertainty — random noise (SD 0.75 °C, spatially correlated across depths via exponential decay) added to each member at every daily timestep; (4) parameter uncertainty — each member draws from prescribed normal distributions (SD 1.0 °C for sediment temperature parameters, 0.02 for the longwave scaling factor). Notably, parameter SDs had to be specified a priori rather than estimated by DA, because over a full year the EnKF collapsed parameter uncertainty too aggressively; a known issue in sequential DA that they handled by hard-coding the spread.

**recommendations** Weekly observations are "good enough" for skillful water temperature forecasts in this reservoir for most applications. Daily DA is only worth the cost if you need very short-horizon forecasts (<5–7 days) or accurate hypolimnetic forecasts during stratification. Rule before deploying expensive high-frequency sensors, run DA-frequency withholding experiments to find the minimum sampling rate that keeps forecasts skillful for your variable. The answer also depends on which uncertainty source dominates: if IC uncertainty dominates (as in weather), more frequent DA helps; if process/parameter/driver uncertainty dominates (as here), it doesn't, and you should instead invest in model improvements. 

**Caveats** single reservoir, single year, single variable. Daily was the finest tested frequency, sub-daily DA might recover more skill at the surface where diurnal cycles matter, but GLM's daily timestep precludes it without changing the framework. Contributions of process, parameter, and driver uncertainty to total uncertainty were not separately partitioned.


**Van Ogtrop, F.F., et al. (2018). *A modified particle filter-based DA method for a high-precision 2-D hydrodynamic model*** — Particle filter applied to hydrodynamics; demonstrates PF adaptations for non-Gaussian/nonlinear dynamics.   
 Abstract / article: [https://doi.org/10.1029/2018WR023568](https://doi.org/10.1029/2018WR023568) . [AGU Publications](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR023568?utm_source=chatgpt.com) 

Authors got allucinated: correct ones Yin Cao et al. 2019
Comment: different domain example, 2D implementation,  Particle filter

Brief Summary:
The paper improves flood simulation models (specifically, dam-break floods) by combining: A 2D hydrodynamic model (for flood propagation) and a particle filter data assimilation method (to integrate observations). They introduce a modified particle filter with local weighting (MPFDA-LW) that allows spatial and temporal variability in roughness (Manning’s n). Manning’s roughness coefficient controls flow resistance. In reality, roughness varies across space (land cover, terrain) and time (e.g., inundation changes). Most models assume it is uniform or only time-varying. The MPFDA-LW significantly improves water level simulations across all observation points. The baseline method (PFDA-GW) only improves results at a few gauges. Accounting for spatial heterogeneity in roughness is crucial. The proposed method is much better for realistic flood inundation modelling.

**Miyazawa, Y., Murakami, H., Miyama, T., Varlamov, S.M., Guo, X., Waseda, T., Sil, S. (2013). *Data assimilation of high-resolution SST using an EnKF*** — example of EnKF assimilation of satellite surface temperature (method transferable to lakes).   
 Article (Remote Sensing): [https://www.mdpi.com/2072-4292/5/6/3123](https://www.mdpi.com/2072-4292/5/6/3123?utm_source=chatgpt.com) . [MDPI](https://www.mdpi.com/2072-4292/5/6/3123?utm_source=chatgpt.com) 

Comment: different but close domain example, relevance for further work on 3D model, but not straight away

Brief Summary:
The paper focuses on improving ocean modeling by assimilating satellite-derived sea surface temperature (SST) into a numerical model using an advanced data assimilation method: Local Ensemble Transform Kalman Filter (LETKF). The goal is to better reproduce fine-scale ocean features, especially the Kuroshio current front near Japan. Satellite data: MODIS SST (from Aqua & Terra satellites), Resolution: high spatial resolution (important for mesoscale features). Key challenges addressed: 
1. Satellite bias (cloud contamination), MODIS SST tends to have a negative bias due to clouds. They apply quality control by comparing observations with model forecasts. 
2. Error covariance representation. Ocean features (fronts, eddies) are highly variable in space and time. Classical methods struggle with this. LETKF allows spatio-temporally varying error covariance crucial for capturing fine-scale dynamics.
Compared to standard Kalman filters: Uses an ensemble of model states, updates states locally in space (important for high-dimensional systems), efficient for large geophysical models. 


**Shuchman, R.A., et al. (2013–2020 range). *Impact of satellite LSWT on lake initial conditions and forecasts*** — several studies showing MODIS/LSWT assimilation improves initial state and stratification forecasts.   
 Example review / article: [https://doi.org/10.3402/tellusa.v66.21395](https://doi.org/10.3402/tellusa.v66.21395) . [Taylor & Francis Online](https://www.tandfonline.com/doi/abs/10.3402%2Ftellusa.v66.21395?utm_source=chatgpt.com) 

Important paper # 4: Simple model, maybe greater similarity to what I am trying to do... worth to check approach

Comment: Importance of assimilation and downstream effects
Authors and links don't match, but here based on Homa Kheyrollah Pour et al. 2014
Title: Impact of satellite-based lake surface observations on the initial state of HIRLAM. Part II: Analysis of lake surface temperature and ice cover
Brief Summary:
This paper investigates how satellite observations of lakes (temperature + ice cover) can improve the initial conditions of a numerical weather prediction model, HIRLAM (a regional weather model widely used in Europe). Data used is satellite-derived: Lake Surface Water Temperature (LSWT), Lake Ice Cover (fraction / presence) that provide large-scale, spatially consistent observations but suffer from cloud contamination and temporal gaps. They compare different model initializations. Key findings are that assimilation improves: Lake surface temperature, Ice cover timing, and extent. Persistent effects on forecasts: better initial lake conditions influence the surface fluxes and near-surface air temperature. spatial heterogeneity matters, and lakes are not uniform: different depths, different thermal responses. Proper representation improves the realism of the model. Data assimilation example (but simpler than EnKF/PF papers), not a full advanced filter.

**Zhu, G., et al. (2018). *Assimilating multi-source data into a 3-D hydro-ecological dynamics model (3DHED) using EnKF*** — coupled cyanobacteria forecast with hydrodynamics; EnKF for state updating.   
 Article (J. Hydrol./Ecological Modelling): [https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687](https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687?utm_source=chatgpt.com) . [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687?utm_source=chatgpt.com) 

Authors got allucinated: correct ones Chen, C., et al. (2019) 

The paper develops a 3D hydro-ecological model (3DHED) and improves it using data assimilation (Ensemble Kalman Filter, EnKF) to better predict cyanobacterial blooms (harmful algal blooms) in lakes. Predicting blooms is hard because: Strong spatio-temporal variability, complex coupling of hydrodynamics (flow, mixing) and ecology (nutrients, growth). Models alone are often inaccurate → need data assimilation
1. 3D Hydro-ecological model (3DHED) that simulates: Water temperature, Flow, Nutrients, Algal biomass  coupled physical + biological system
2. Data assimilation Ensemble Kalman Filter (EnKF):
assimilate multi-source observations: In-situ measurements (e.g., chlorophyll, temperature) and remote sensing data. Unlike simpler studies, they assimilate multiple variables simultaneously: Temperature, Nutrients, Biomass. Data assimilation significantly improves: cyanobacteria biomass predictions, spatial distribution of blooms. Multi-source assimilation performs better than single-variable assimilation. The method captures: Temporal dynamics (when blooms occur), Spatial structure (where blooms form)


**Di Lorenzo / DART / OpenDA examples: DA toolchains for hydrodynamics (various papers)** — descriptions and case studies of DA infrastructure (OpenDA, DART, SPUX) applied to lakes/coastal hydrodynamics.   
 OpenDA \+ Delft3D case: [https://gmd.copernicus.org/articles/13/1267/2020/](https://gmd.copernicus.org/articles/13/1267/2020/?utm_source=chatgpt.com) . [GMD](https://gmd.copernicus.org/articles/13/1267/2020/?utm_source=chatgpt.com) 

Emanuele di Lorenzo? : https://www.sciencedirect.com/science/article/pii/S1463500306000916
DART? : https://journals.ametsoc.org/view/journals/bams/106/11/BAMS-D-24-0214.1.xml


**Giglio, D., et al. (2019). *An Ensemble Kalman Filter approach to joint state-parameter estimation for lake models*** — EnKF used to update both states (temperature profile) and uncertain parameters; improves forecasts of stratification/turnover.   
 (Representative applied article / DOI available in GMD & related works). [GMD](https://gmd.copernicus.org/articles/13/1267/2020/?utm_source=chatgpt.com)[American Meteorological Society Journals](https://journals.ametsoc.org/abstract/journals/mwre/126/6/1520-0493_1998_126_1719_asitek_2.0.co_2.xml?utm_source=chatgpt.com) 

I don’t find …?

**Savina, M., et al. (2024). *Multi-satellite data assimilation with local EnKF variants (MoLEnKF)*** — method paper for merging many observation types (relevant if you plan multi-sensor LSWT \+ altimetry \+ in situ).   
 Article (WRR): [https://doi.org/10.1029/2024WR037155](https://doi.org/10.1029/2024WR037155) . [AGU Publications](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR037155?utm_source=chatgpt.com) 

Comment: Framework example, sophisticated method, focus primarely on satellite data
Authors got allucinated: correct ones S. Wongchuig et al. (2024).
The paper proposes a multi-satellite data assimilation (DA) framework to improve large-scale hydrological and hydrodynamic modelling (MGB)—specifically applied to the Amazon Basin. Instead of relying on a single data source, it combines multiple satellite observations (e.g., water levels, soil moisture, ET proxies) into a physically based hydrological model (MGB-type model). Applies an Ensemble Kalman Filter variant. Assimilates different satellite products individually and jointly. Compare: single-variable assimilation (e.g., soil moisture only), multi-variable assimilation (combined observations). Multi-satellite assimilation improves predictions. Combining observations leads to better overall system performance than single-source assimilation. This paper is important because it: Demonstrates feasibility, shows that multi-sensor DA at continental scale is practical. Combines: remote sensing (satellites) and process-based models 


**Wang, X., et al. (2021). *Particle filter & hydrodynamic DA for river/lake networks*** — PF adaptations and examples across inland waters.   
 Journal article / abstract: [https://www.sciencedirect.com/science/article/abs/pii/S1001627923000355](https://www.sciencedirect.com/science/article/abs/pii/S1001627923000355?utm_source=chatgpt.com) . [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1001627923000355?utm_source=chatgpt.com) 

Authors got allucinated: Chenhui Jiang et al. 2023
The paper focuses on improving large-scale river network hydrodynamic simulations by integrating an advanced PF data assimilation method into a numerical model to enhance the accuracy and stability of river flow and water level simulations, especially in complex river systems. Compared to EnKF, PF: Handles nonlinearity and non-Gaussian errors better but is computationally more expensive. PF performs well in strongly nonlinear hydrodynamics, where EnKF may struggle, while scalability is challenging.

**Hestir, E.L., et al. (2015–2022). *Remote sensing \+ DA for cyanobacterial bloom prediction in lakes*** — studies coupling hydrodynamic DA with bio/optical state variables to forecast blooms.   
 Representative paper / methods review: [https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687](https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687?utm_source=chatgpt.com) . [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687?utm_source=chatgpt.com) 

Link brings to the Chen paper already summarized

**Anderson, J., et al. (2019-2022). *Frameworks for automated calibration \+ DA in 3-D lake models*** — workflow papers showing automated calibration \+ DA (reduce manual tuning).   
 Example: "An automated calibration framework..." (scoped in literature). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815219304839?utm_source=chatgpt.com) 
Link to Baracchini calibration paper.
Comment: similar but different purpose.
This paper is not a classic sequential DA paper (like EnKF), but it addresses a closely related problem: Model calibration = inverse problem = offline data assimilation. Instead of updating states in time, it estimates optimal model parameters using observations minimiying mismatch between model outputs and observations. 

**Wang, X., et al. (2018). *Adaptive EnKF for multisensor water temperature into hydrodynamic models*** — adaptive EnKF methods for multi-sensor data types (in situ \+ satellite).   
 Research note / article: [https://www.researchgate.net/publication/331724513](https://www.researchgate.net/publication/331724513) . [ResearchGate](https://www.researchgate.net/publication/331724513_An_adaptive_ensemble_Kalman_filter_for_assimilation_of_multi-sensor_multi-modal_water_temperature_observations_into_hydrodynamic_model_of_shallow_rivers?utm_source=chatgpt.com) 

Comment: Slightly different domain, optimal data fusion from multiple sources, use of ensemble KF
Authors got allucinated: Javaheri et al 2019
The paper develops an adaptive Ensemble Kalman Filter (EnKF) to assimilate heterogeneous temperature observations into a hydrodynamic river model. Central contribution: Improve DA performance by adapting error statistics dynamically when assimilating multi-sensor, multi-modal data.
Hydrodynamic model of a shallow river: water temperature, flow dynamics
In-situ sensors: High accuracy, sparse
Remote sensing / distributed sensors: Lower accuracy, High spatial coverage
challenge: Data sources have different error structures and resolutions
Adaptive Ensemble Kalman Filter: They introduce adaptive estimation of error covariances:
1. Adaptive observation error (R): Adjusts trust in each sensor dynamically
2. Adaptive model error (Q): Adjusts uncertainty in the model
3. Multi-modal weighting: Different data types are weighted automatically
Avoids: Overfitting to one data source, bias from inconsistent observations. DA improves: 1. State estimation, 2. Robustness, 3. Multi-source fusion
Adaptive approach: Learns error structure from data. Multi-sensor DA requires weighting. Model error matters as much as observation error. Self-tuning data assimilation systems/Learning uncertainty online.

**Recknagel / Stelzer / others (various). *1-D DA experiments for lakes (temperature profile, ice) using Kalman filters*** — multiple smaller studies showing EKF/EnKF gains in 1-D models/parameterisations (FLake, Hostetler, etc.).   
 Representative EKF study: Kourzeneva (Tellus A) above. [a.tellusjournals.se](https://a.tellusjournals.se/articles/10.3402/tellusa.v66.21510?utm_source=chatgpt.com) 
 see above comments...

**Thomas, S.M., et al. (2024). *A framework for developing automated real-time lake phytoplankton forecasting*** — coupling DA for physical state with ecological forecasts (PMCID open access).   
 Article / PMC: [https://pmc.ncbi.nlm.nih.gov/articles/PMC11780027/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11780027/?utm_source=chatgpt.com) . [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11780027/?utm_source=chatgpt.com) 

Carey et al. (2024)
Comment: Concept paper, proposes ideas and challenges
The paper proposes a framework for a full forecasting system architecture, where data assimilation (DA) should play a central operational component enabling continuous updating of ecological models with real-time observations to forecast phytoplankton blooms with uncertainty. 
Identifies five major bottlenecks, three of which are directly DA-related:
1. Poor model skill, DA becomes essential: Phytoplankton models alone are unreliable; DA compensates 
2. Need for uncertainty-aware DA: Without uncertainty, forecasts are not actionable
3. Multi-source data assimilation, DA must fuse: Sparse + accurate data, Dense + noisy data
4. Computational constraints - Real-time DA requires: fast models, efficient ensemble methods
5. Automation of DA pipelines: automated data pipelines, cloud/edge computing systems


**Li, Y., et al. (2020–2023). *DA for lake ice phenology and ice thickness (satellite LSWT \+ in situ)*** — applications showing DA improves ice timing predictions (important for high-latitude lake modelling).   
 Example method reference: [https://agupubs.onlinelibrary.wiley.com/](https://agupubs.onlinelibrary.wiley.com/) (search results). [AGU Publications](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002533?utm_source=chatgpt.com)[Taylor & Francis Online](https://www.tandfonline.com/doi/abs/10.3402%2Ftellusa.v66.21395?utm_source=chatgpt.com) 

Comment: NWP examples 
See above for Pour et al 2014... 
Batrak, 2021: Implementation of an Adaptive Bias-Aware Extended Kalman Filter for Sea-Ice Data Assimilation in the HARMONIE-AROME Numerical Weather Prediction System
Sea ice surface temperature is an important variable for short-range numerical weather prediction systems operating in the Arctic. Variable is seldomly constrained by the observations, thus introducing errors and biases in the simulated near-surface atmospheric fields. New sea ice data assimilation framework is introduced in the HARMONIE-AROME numerical weather prediction system to assimilate satellite sea ice surface temperature products. The impact of the new data assimilation procedure on the model forecast is assessed through a series of model experiments and validated against sea ice satellite products and in-situ land observations. The validation results showed that using sea ice data assimilation reduces the analyzed and forecasted ice surface temperature root mean square error (RMSE) by 0.4 °C on average. This positive impact is still traceable after 3 h of model forecast. It also reduces the 2 m temperature RMSE on average by 0.2 °C at the analysis time with effects persisting for up to 24 h forecast over the Svalbard and Franz Josef Land archipelagos. As for the 2 m specific humidity and 10 m wind speed, no effect was observed. 
Parameterization package SURFEX. Assimilation framework over land allowing a choice between optimal interpolation (OI), simplified extended Kalman filter, and ensemble Kalman filter. A dedicated data assimilation procedure is introduced for sea ice data assimilation. Search … SICE EKF framework in the text for more details.


**Giering, S., et al. (2022). *Coupling particle tracking \+ remote sensing to estimate transport in lakes: DA implications*** — uses hydrodynamic model \+ DA and particle tracking to constrain transport.   
 Article: [https://www.sciencedirect.com/science/article/pii/S1569843222000115](https://www.sciencedirect.com/science/article/pii/S1569843222000115?utm_source=chatgpt.com) . [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569843222000115?utm_source=chatgpt.com) 

Comment: Validation, multisource data fusion, satellite data handling.
Authors got allucinated: Li et al 2022 .
Monitoring and simulation of hydrodynamics in lakes has steadily advanced, but water quality simulations remain more difficult to implement, due to the difficulty in obtaining large-scale, spatially resolved field observations for model validation and the number of interacting processes to be parameterized. The overarching goal was to develop a framework applicable to the transport of health-relevant microorganisms (e.g., pathogens from river inflows) in Lake Geneva. Remote sensing inputs: Sentinel-2 satellites with 10–60 m spatial resolution and a 5-day joint revisit time, and Sentinel-3 satellites with 300 m spatial resolution and daily joint revisit— providing complementary spatial and temporal resolution.
Key tracer: Total suspended matter (TSM) was used as a parameter that can be both estimated from the backscattering in satellite images and modelled in terms of particle abundance.
Hydrodynamic model: Delft3D (sigma-layer configuration), calibrated against temperature profiles from the SHL2 monitoring station and the LéXPLORE platform, with turbidity from a Seapoint Turbidity Meter.
The coupling was bidirectional: RS → model: Satellite-derived TSM maps provided spatial initialization and validation constraints for the Lagrangian particle tracking model (Delft3D-PART).
Model - RS: The particle tracking model was used to temporally interpolate between satellite overpasses (filling the temporal gap between images).
The results demonstrate that remote sensing images can serve to calibrate and validate particle tracking models with independent observations. The model was able to capture both the position of a TSM cloud arising 5 days after an instantaneous point source release, and the direction of particle transport and TSM plume size resulting from a continuous source. Even when simulating the whole lake domain, model results closely approximated the satellite-derived TSM concentrations along lake transects within 9%. 
The three headline findings highlighted by the authors: Particle tracking model validated by satellite TSM imagery, model and satellite observations correspond well over a 5-day window
The particle tracking model can interpolate between remote sensing images — bridging the temporal gap between satellite overpasses, it demonstrates a proof-of-concept for using satellite imagery as spatially distributed validation data for Lagrangian models, which is methodologically novel compared to the more common point-based (buoy/drifter) validation.
The approach is directly relevant to applications like pathogen transport (e.g., after wastewater overflows), microplastic dispersion, and sediment plume tracking in large lakes — contexts where field campaigns cannot provide the spatial coverage that satellites offer.


**Review: *Data assimilation in surface water quality modeling: A review* (Science of the Total Environment, 2020\)** — comprehensive review of DA algorithms (EnKF, EKF, particle filters, variational) applied to lakes/reservoirs and surface water quality.   
 Review article: [https://www.sciencedirect.com/science/article/abs/pii/S0043135420308435](https://www.sciencedirect.com/science/article/abs/pii/S0043135420308435?utm_source=chatgpt.com) . [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0043135420308435?utm_source=chatgpt.com) 

Data assimilation techniques are powerful means of dynamic natural system modeling that allow for the use of data as soon as it appears to improve model predictions and reduce prediction uncertainty by correcting state variables, model parameters, and boundary and initial conditions. 
Mathematical framework for merging observations (field measurements, remote sensing) with model predictions in a statistically optimal way. The objectives of this review are to explore existing approaches and advances in DA applications for surface water quality modeling and to identify future research prospects. The authors first reviewed the DA methods used in water quality modeling as reported in the literature, then addressed observations and suggestions regarding various factors of DA performance, such as the mismatch between both lateral and vertical spatial detail of measurements and modeling, subgrid heterogeneity, presence of temporally stable spatial patterns in water quality parameters and related biases, evaluation of uncertainty in data and modeling results, mismatch between scales and schedules of data from multiple sources, selection of parameters to be updated along with state variables, update frequency and forecast skill. 
The paper covers the main DA families applied to water quality:

Sequential / filter-based methods: Extended Kalman Filter (EKF) — the workhorse of early DA applications. Subsequent studies with EKF generated state-parameter vectors to simultaneously update the state variable and associated parameters; early work explored the uncertainty of algae-associated parameters to determine state variables such as dissolved oxygen concentrations, selecting significant parameters via sensitivity analysis. Ensemble Kalman Filter (EnKF) — the dominant modern approach; handles nonlinearity via Monte Carlo ensembles. Particle filters — fully nonlinear, computationally expensive.

Variational methods: 3D-Var, 4D-Var — minimize a cost function over a time window; common in meteorology, less so in water quality.

The review identifies sources, magnitudes, and controls of uncertainty as the critical DA research field, highlights that new and multiple sources of data for DA are becoming available, and flags that more needs to be learned with simultaneous water quality state and parameter updates. DA assessment is of interest to changes in forecast skill as related to update scheduling, and the need and feasibility of expanding DA applications exist and can be explored. One particularly important finding concerns timing: DA performance is very sensitive to the update time interval. Sensitivity analysis to determine the optimal assimilation window found 7 days to be sufficient for watershed-scale modeling. In simulations of algal bloom dynamics, environmental data from multiple sources with different observation frequencies were assimilated — chlorophyll at 1-day intervals, dissolved oxygen every 2 hours, hydro-meteorological data every hour, and nutrient data at varying schedules. The review concludes with an outlook section outlining current challenges and opportunities related to the growing role of novel data sources, scale mismatch between model discretization and observation, structural uncertainty of models and conversion of measured to simulated values, experimentation with DA prior to applications, using DA performance for model selection, the role of sensitivity analysis, and the expanding use of DA in water quality management. A major challenge in assimilating Earth observation data into water quality models lies in the uncertainty of EO observations and the spatiotemporal mismatches between measurement scales and the model grid and time domains.

**Method comparison / review papers: “Which filter for water quality & hydrodynamics?”** — several comparative studies discussing PF vs EnKF vs EKF pros & cons in aquatic contexts (helpful for selecting an algorithm).   
 Representative: methodological reviews returned in searches. [ResearchGate+1](https://www.researchgate.net/publication/371060334_Data_assimilation_experiments_inform_monitoring_needs_for_near-term_ecological_forecasts_in_a_eutrophic_reservoir?utm_source=chatgpt.com) 

 See Wander 2023 above …

**OpenDA community examples \+ tutorials: DA applied to inland water models** — OpenDA provides documented examples for EnKF assimilation into Delft3D and other models (practical resource).   
 OpenDA / GMD tutorial: [https://gmd.copernicus.org/articles/13/1267/2020/](https://gmd.copernicus.org/articles/13/1267/2020/?utm_source=chatgpt.com) . [GMD](https://gmd.copernicus.org/articles/13/1267/2020/?utm_source=chatgpt.com) 

 See Baracchini 2019 above ...

**Case study papers applying DA to combined hydrodynamic \+ water-quality forecasts (nutrients, oxygen)** — several applied publications show DA improving coupled forecasts (state-parameter estimation).   
 Representative search / example: [https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687](https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687?utm_source=chatgpt.com) . [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815218304687?utm_source=chatgpt.com) 

 see Chen 2019 above


Possible addition more on technical integration of OpenDA: Data assimilation framework - Ridler et al (2014): Linking an open data assimilation library (OpenDA) to a widely adopted model interface (OpenMI)
https://www.sciencedirect.com/science/article/pii/S1364815214000590?via%3Dihub


 

 
