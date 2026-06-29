# Data Assimilation for Lake Hydrodynamic Models
 
Integrating observations into operational lake models to improve forecast accuracy for the [Alplakes](https://www.alplakes.eawag.ch) platform.
 
## Overview
 
This repository implements a data assimilation (DA) framework for lake hydrodynamic models operated as part of the [Alplakes](https://www.alplakes.eawag.ch) project at [Eawag](https://www.eawag.ch). The framework blends in situ measurements and satellite remote sensing data into three-dimensional hydrodynamic simulations, correcting model states in near-real time to produce more accurate lake forecasts across the European Alpine region.
 
Alplakes provides operational hydrodynamic simulations for over 200 lakes, combining 1D ([Simstrat](https://www.eawag.ch)) and 3D ([Delft3D-FLOW](https://oss.deltares.nl/web/delft3d), [MITgcm](https://mitgcm.org/)) models with remote sensing products. Data assimilation is the mechanism that bridges the gap between model predictions and observed reality, reducing forecast errors and quantifying uncertainty.

