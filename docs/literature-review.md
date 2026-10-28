# Literature review

!!! abstract "In short"
    Five papers shaped the module. Baracchini et al. (2020) is the closest precedent: an EnKF with forcing perturbation, 20 members, on Lake Geneva. Safin et al. (2022) show the particle-filter alternative and its cost. Kourzeneva (2014) is the early 1D lake reference. Thomas et al. (2020) and Wander et al. (2024) come from the FLARE forecasting system and shaped how we think about uncertainty sources and assimilation frequency. Further reading follows at the end.

## Core papers

### Baracchini et al. (2020): EnKF for a 3D model of Lake Geneva { #baracchini-2020 }

Baracchini, T., Chu, P. Y., Šukys, J., Lieberherr, G., Wunderle, S., Wüest, A., and Bouffard, D. (2020). Data assimilation of in situ and satellite remote sensing data to 3D hydrodynamic lake models: a case study using Delft3D-FLOW v4.03 and OpenDA v2.4. *Geoscientific Model Development*, 13(3), 1267–1284. [doi:10.5194/gmd-13-1267-2020](https://doi.org/10.5194/gmd-13-1267-2020)

**Summary.** A data assimilation framework that combines in situ profiles, satellite surface temperature and a 3D hydrodynamic model, built for near-real-time operation (meteolakes.ch). An EnKF in OpenDA assimilates in situ temperature profiles and AVHRR lake surface water temperature (LSWT) into Delft3D-FLOW for Lake Geneva.

**Method.** Delft3D-FLOW v4.03 (z-layer, 100 vertical levels, 450 m grid, κ-ε closure) coupled to OpenDA v2.4 through a file-based black-box wrapper, for 2017. The EnKF updates the 3D temperature state only. Spread is injected into the wind forcing (u and v), with OpenDA's spatiotemporally correlated noise model, whose standard deviation and correlation scales come from the COSMO-E ensemble. Assimilated: 31 in situ profiles at SHL2 and GE3, and 128 quality-filtered AVHRR images (of 3372). Observation error from instrument precision plus a 48-hour variability window for in situ data, and a 1 °C threshold for AVHRR. Gaspari–Cohn localization with a 15 km cutoff; an ensemble-size sensitivity study.

**Results.** RMSE −54 % and MAE −60 % against the calibrated control run, consistent from the surface to the deep water. A warm bias in the 5–25 m layer is removed and the thermocline depth corrected. Gyres, thermal fronts and upwellings are preserved and sharpened: a September upwelling underestimated by 5 °C is reduced to a 2.5 °C error. With localization, 20 members were enough.

**Relevance for us.** The closest precedent to our setup: forcing perturbation, state update, 20 members, operational aim. The starting point for the 3D extension.

### Safin et al. (2022): physics-preserving particle filter for Lake Geneva { #safin-2022 }

Safin, A., Bouffard, D., Ozdemir, F., Ramon, C. L., Runnalls, J., Georgatos, F., Minaudo, C., and Šukys, J. (2022). A Bayesian data assimilation framework for lake 3D hydrodynamic models with a physics-preserving particle filtering method using SPUX-MITgcm v1. *Geoscientific Model Development*, 15(20), 7715–7730. [gmd.copernicus.org/articles/15/7715/2022](https://gmd.copernicus.org/articles/15/7715/2022/)

**Summary.** A Bayesian framework coupling the inference package SPUX with MITgcm for Lake Geneva, positioned as an advance on Baracchini et al. (2020), which could use only about 3.7 % of the available LSWT images. Three components:

1. **Parameter inference with EMCEE**, an affine-invariant ensemble sampler, suited to nonlinear parameters but expensive.
2. **A physics-preserving particle filter.** Trajectories are resampled by their observational likelihood; states are only deleted or replicated, never modified, so there are no shocks.
3. **A BiLSTM network for the bulk-to-skin conversion**, estimating skin temperature from 27 hours of bulk temperature and meteorology, with its uncertainty, so that nearly all quality-flagged pixels can be used.

**Results.** RMSE and MAE improve by only 4–15 %, but with 798 AVHRR images instead of about 124 and without image-by-image thresholding. The BiLSTM reduces RMSE by 33 % on its test set but increases it by about 10 % in the assimilation run. About 3 months on the Swiss National Supercomputing Centre. The authors suggest cheaper parameter optimisation plus an improved particle filter for operational use.

**Relevance for us.**

- Small improvement over the control run, at a very high computational cost.
- Particle filtering degenerates with dense, informative observations, as our PF does ([Method](workblog/method.md)).
- Each member is driven by a COSMO-E ensemble member: a possible alternative to statistical forcing noise.
- The bulk-to-skin observation operator is a template for satellite assimilation.

### Kourzeneva (2014): EKF for the 1D lake scheme FLake { #kourzeneva-2014 }

Kourzeneva, E. (2014). Assimilation of lake water surface temperature observations using an extended Kalman filter. *Tellus A: Dynamic Meteorology and Oceanography*, 66(1), 21510. [doi:10.3402/tellusa.v66.21510](https://doi.org/10.3402/tellusa.v66.21510)

**Summary.** An early and influential lake DA paper: an extended Kalman filter (EKF) assimilates lake water surface temperature into FLake, the two-layer parameterisation used in numerical weather prediction. The analysed variables are the mean water temperature, the bottom temperature, the mixed-layer depth and the shape factor.

**Method.** The small control vector of FLake makes an EKF tractable, an advantage over ensemble methods in NWP. Mixed and stratified regimes are treated separately, and the filter is re-initialised when the regime switches. Ice is handled implicitly: an observed surface temperature means no ice. Jacobians by finite differences, one perturbed FLake run per control variable. 27 Finnish lakes with in situ data, 4 with merged in situ and MODIS data.

**Results.** The free run has FLake's known summer warm bias, up to 5 °C in the mixed layer; the EKF brings it close to the observations on all 27 lakes. Satellite data help where in situ coverage is sparse. Little effect on ice, because observations are missing just before freeze-up. Spring and early-summer observations are the most valuable for the warming period.

**Relevance for us.** A simple 1D model with a known bias, corrected by assimilation of state variables without an ensemble.

### Thomas et al. (2020): FLARE and uncertainty partitioning { #thomas-2020 }

Thomas, R. Q., Figueiredo, R. J., Daneshmand, V., Bookout, B. J., Puckett, L. K., and Carey, C. C. (2020). A near-term iterative forecasting system successfully predicts reservoir hydrodynamics and partitions uncertainty in real time. *Water Resources Research*, 56(11), e2019WR026138. [doi:10.1029/2019WR026138](https://doi.org/10.1029/2019WR026138) · Code: [FLAREr](https://flare-forecast.org/FLAREr/)

**Summary.** FLARE (Forecasting Lake And Reservoir Ecosystems), a daily iterative forecasting system for Falling Creek Reservoir:

- **Sensors**: a thermistor chain and an inflow sensor, streamed to the cloud.
- **Model**: the 1D General Lake Model (GLM), driven by the 21 members of the NOAA GEFS 16-day forecast.
- **Filter**: an EnKF that updates the temperature profile and three parameters (a longwave scaling factor and two sediment temperatures) by state augmentation.

**Uncertainty partitioning**, the main contribution. Five sources, each switchable on and off:

- *driver*: one GEFS member per FLARE member;
- *initial condition*: the spread on day 0;
- *process*: Gaussian noise on the temperature at each daily step, correlated over depth;
- *parameter*: spread of the three estimated parameters;
- *observation*: sensor error in the EnKF's R.

**Results.** Skilful 16-day temperature forecasts in real time. Process and driver uncertainty dominate the forecast variance; initial conditions matter only at the shortest horizons. That is the opposite of weather forecasting, where initial conditions dominate. Running with fixed parameters degraded skill substantially.

**Relevance for us.** A 1D operational forecasting system, with ensembles of about 400 members for a light model. Process noise as a source of spread is the basis of our [forecast uncertainty band](workblog/experiments.md#uncertainty-band).

### Wander et al. (2024): how often to assimilate { #wander-2024 }

Wander, H. L., Thomas, R. Q., Moore, T. N., Lofton, M. E., Breef-Pilz, A., and Carey, C. C. (2024). Data assimilation experiments inform monitoring needs for near-term ecological forecasts in a eutrophic reservoir. *Ecosphere*, 15(2), e4752. [doi:10.1002/ecs2.4752](https://doi.org/10.1002/ecs2.4752)

**Summary.** If initial conditions are not the dominant uncertainty, how often do we need to assimilate? FLARE on Beaverdam Reservoir (0.28 km², 11 m deep, dimictic, eutrophic), 2021, with temperature sensors every metre. Assimilation daily, weekly, fortnightly or monthly, by withholding observations; forecasts 1 to 35 days ahead.

**Method.** GLM v3.3.0 driven by 30 NOAA GEFS members, 0.5 m vertical resolution. EnKF with 256 members and state augmentation of three parameters (longwave scaling, two sediment temperatures). Process noise 0.75 °C, observation noise 0.1 °C. Scored with RMSE (2 °C "skilful" threshold) and CRPS, separately for surface, middle and bottom layers and for mixed and stratified periods.

**Results.**

- **Frequency.** Daily assimilation is best at 1–7 days, weekly at 8–35 days and overall. Daily assimilation crosses the 2 °C threshold by day 28; the others never do. Daily updates over-fit the parameters, which become noisier than with weekly updates. Fixed parameters with daily state updates are much worse (RMSE 3.12 against 1.95 °C).
- **Depth and season.** Bottom waters are the most skilful (1.13 °C), the surface the least (1.78 °C); stratified slightly better than mixed. The horizon at which weekly overtakes daily is longer at depth (> 8 days) than at the surface (> 5 days).
- **Initial-condition share.** With daily assimilation, initial conditions are 0.01 % of the 1-day forecast variance; with less frequent assimilation, 54–71 %. Beyond 10 days, below 1 % almost everywhere.
- **Parameters.** Their spread had to be fixed a priori, because over a year the EnKF collapsed it.

**Recommendations of the authors.** Weekly observations are "good enough" for most uses; daily assimilation pays off for short horizons or for the hypolimnion in summer. Run withholding experiments before investing in high-frequency sensors.

**Caveats.** One reservoir, one year, one variable; daily was the finest frequency tested.

**Relevance for us.** Supports daily assimilation for short-horizon accuracy, and weekly assimilation for longer forecasts ([When to assimilate](workblog/method.md#timing)).

## Further reading

Papers of wider or partial relevance.

**Lakes and reservoirs**

- **Baracchini, T., Hummel, S., Verlaan, M., Cimatoribus, A., Wüest, A., and Bouffard, D. (2020).** An automated calibration framework and open source tools for 3D lake hydrodynamic models. *Environmental Modelling & Software*, 134, 104787. [doi:10.1016/j.envsoft.2020.104787](https://doi.org/10.1016/j.envsoft.2020.104787). Calibration as offline assimilation: Delft3D-FLOW and OpenDA coupled for automated parameter estimation, tested on Lake Geneva and Greifensee.
- **Kheyrollah Pour, H., Rontu, L., Duguay, C., Eerola, K., and Kourzeneva, E. (2014).** Impact of satellite-based lake surface observations on the initial state of HIRLAM. Part II: Analysis of lake surface temperature and ice cover. *Tellus A*, 66, 21395. [doi:10.3402/tellusa.v66.21395](https://doi.org/10.3402/tellusa.v66.21395). Satellite lake surface temperature assimilated into the initial state of a weather model.
- **Chen, C., et al. (2019).** Assimilating multi-source data into a three-dimensional hydro-ecological dynamics model using Ensemble Kalman Filter. *Environmental Modelling & Software*, 117, 188–199. [doi:10.1016/j.envsoft.2019.03.028](https://doi.org/10.1016/j.envsoft.2019.03.028). An EnKF in a 3D hydro-ecological model (3DHED) for cyanobacterial biomass in Lake Taihu; adding remote sensing to the in situ data improves the spatial patterns.
- **Li, et al. (2022).** Coupling remote sensing and particle tracking to estimate trajectories in large water bodies. [EPFL Infoscience](https://infoscience.epfl.ch/record/293836). Sentinel-2/3 suspended matter maps validate a Delft3D particle-tracking model of Lake Geneva, which in turn fills the gaps between overpasses.
- **Carey et al. (2024).** A framework for developing a real-time lake phytoplankton forecasting system to support water quality management in the face of global change. *Ambio*. [doi:10.1007/s13280-024-02076-7](https://doi.org/10.1007/s13280-024-02076-7). A concept paper for an automated real-time forecasting system with quantified uncertainty, with DA as a central component.

**Other domains**

- **Javaheri et al. (2019).** An adaptive ensemble Kalman filter for assimilation of multi-sensor, multi-modal water temperature observations into hydrodynamic model of shallow rivers. *Journal of Hydrology*. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169419302318). In situ sensors (dense in time) and Landsat-7 (dense in space) combined in an adaptive EnKF; deals with the skin–bulk temperature mismatch.
- **Miyazawa, Y., et al. (2013).** Data assimilation of the high-resolution sea surface temperature obtained from the Aqua-Terra satellites (MODIS-SST) using an ensemble Kalman filter. *Remote Sensing*, 5(6), 3123–3139. [doi:10.3390/rs5063123](https://doi.org/10.3390/rs5063123). An LETKF with spatiotemporally varying covariances for the Kuroshio front; quality control removes most of the cloud-induced cold bias, and the ensemble spread screens erroneous data. Relevant for 3D and satellite work later.
- **Wongchuig, S., et al. (2024).** Multi-satellite data assimilation for large-scale hydrological-hydrodynamic prediction: proof of concept in the Amazon basin. *Water Resources Research*, 60(8). [doi:10.1029/2024WR037155](https://doi.org/10.1029/2024WR037155). A multi-observation local EnKF (MoLEnKF) for water surface elevation, water storage, flood extent and soil moisture; water surface elevation gives the largest gains (discharge error −38 %).
- **Cao, Y., et al. (2019).** A modified particle filter-based data assimilation method for a high-precision 2-D hydrodynamic model considering spatial-temporal variability of roughness: simulation of dam-break flood inundation. *Water Resources Research*, 55. [doi:10.1029/2018WR023568](https://doi.org/10.1029/2018WR023568). Local weighting with spatially varying roughness improves all gauges, global weighting only a few.
- **Improving hydrodynamic modeling of river networks by incorporating data assimilation using a particle filter (2023).** *International Journal of Sediment Research*, 38(5), 711–723. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1001627923000355). A particle filter for the Yangtze–Dongting river–lake network, comparing Gaussian and Cauchy likelihoods.
- **Batrak (2021).** Implementation of an adaptive bias-aware extended Kalman filter for sea-ice data assimilation in the HARMONIE-AROME numerical weather prediction system. Ice surface temperature RMSE −0.4 °C and 2 m temperature −0.2 °C, persisting up to 24 h.

**Reviews and tools**

- **Cho, Pachepsky, Ligaray, Kwon and Kim (2020).** Data assimilation in surface water quality modeling: a review. *Water Research*, 186, 116307. [doi:10.1016/j.watres.2020.116307](https://doi.org/10.1016/j.watres.2020.116307). DA methods in water quality modelling, factors affecting their performance (scale mismatch, data and model uncertainty), and open challenges.
- **Ridler et al. (2014).** Data assimilation framework: linking an open data assimilation library (OpenDA) to a widely adopted model interface (OpenMI). *Environmental Modelling & Software*, 57, 76–89. [doi:10.1016/j.envsoft.2014.02.008](https://doi.org/10.1016/j.envsoft.2014.02.008). Technical integration of OpenDA, tested with an EnKF in MIKE SHE.
