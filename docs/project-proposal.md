# Data Assimilation Framework for Operational Lake Modelling in Canton Ticino

James Runnalls, Eawag

Damien Bouffard, Eawag & UNIL

Martin Schmid, Eawag

## Project Overview

We propose developing a **data assimilation framework** to enhance the accuracy and reliability of operational one-dimensional (1D) lake models in Canton Ticino. This framework will integrate real-time observations with numerical models to improve predictions of lake thermal structure, providing water managers with more precise forecasting capabilities. Through its flexible, open-source architecture and scalable design, the framework will establish a standardized foundation for operational lake forecasting that can be deployed across Switzerland.

## Project Description

### Problem statement

While the operational models integrated into Alplakes successfully capture day-to-day variations in lake dynamics, they suffer from small inaccuracies, which can limit the use of such models. The errors result from:

* Simplified physics and model parameters  
* Inaccurate forcing data (meteorology) that cannot accurately represent the conditions for the entire lake

We have two choices: focusing on consistency (continuous model runs) or accuracy (frequent re-initialization). Re-initializing models to match observations creates discontinuous jumps in the timeseries, breaking the physical consistency needed for trend analysis and scenario studies. Yet allowing models to run freely means accepting growing inaccuracy over time. 

### The Missing Link: Data Assimilation

To solve this challenge, we propose the use of a data assimilation tool that will create a continuous connection between models and observations, maintaining both accuracy and consistency over any timescale. Currently, the observation network and models operate in parallel but rarely communicate. Observations are used to validate model performance after the fact, but not to correct models during operation. This means we're not extracting full value from either investment—models don't benefit from real-time observations, and observations don't inform predictions beyond initial conditions. 

## Current Operational Capabilities

Eawag and Canton Ticino have already invested significantly in lake monitoring and modelling infrastructure, providing the needed data and models to allow:

* **Providing lake hydrodynamic forecasting capabilities:** 5-day predictions for planning and decision-making.  
* **Providing continuous timeseries**: Structured data pipeline and gap filling between observations with physically consistent estimates   
* **Scenario analysis**: Testing "what-if" situations for management decisions   
* **Climate projections**: Understanding long-term trends and impacts 

Currently, the system integrates:

### In situ observation

* Operational in-situ monitoring network with temperature chains and meteorological stations deployed across major lakes   
* Automated data transmission and storage systems providing near real-time observations 

### Satellite Data 

* Routine acquisition of Landsat Collection 2 (Landsat-8, Landsat-9) thermal imagery   
* Processing chains for lake surface temperature and water quality parameters   
* Archive of historical satellite observations for validation 

### Operational Models

* 1D hydrodynamic models (Simstrat) running daily for major lakes   
* Automated forecasting system producing 5-day predictions   
* Web-based visualization platform for model outputs ([Alplakes](https://www.alplakes.eawag.ch/))   
* Established workflows for model initialization and forcing data ([Operational-Simstrat](https://github.com/Eawag-AppliedSystemAnalysis/operational-simstrat)) 

### Technical Infrastructure

* Dedicated computational resources for model simulations   
* REST API for managing observations and model outputs   
* Operational staff familiar with current systems 

This existing infrastructure means we can focus resources entirely on developing and implementing the data assimilation layer, rather than building basic capabilities from scratch. 

## Proposed Solution

We will develop **AlplakesDA** \- a modular, open-source Python library for operational lake data assimilation. This library will provide a production-ready toolkit that seamlessly integrates observations with lake models. 

### Core Library Features

* **Plug-and-play data assimilation** for Simstrat (extendable to other models)   
* **Automated observation processing** from multiple sources (in-situ, satellite, meteorological)   
* **Ensemble-based uncertainty quantification** with configurable ensemble sizes   
* **Real-time quality assurance** for incoming observations   
* **Flexible assimilation scheduling**    
* **Performance monitoring** with validation metrics and diagnostic outputs 

The library will be configuration-driven, requiring no code changes for operational adjustments. 

## Technical Approach 

### Modular Architecture

AlplakesDA will be organized into independent, replaceable modules: 

* **Observation Handlers**: Processors for satellite (Sentinel, Landsat), in-situ (CTD, thermistor chains), and meteorological data sources   
* **Model Interfaces**: Wrapper for Simstrat, with abstract base classes for easy addition of new models   
* **Assimilation Core**: OpenDA integration providing Ensemble Kalman Filter with localization and inflation   
* **Quality Assurance**: Comprehensive QA/QC including range checks, spike detection, spatial consistency tests, and climatological validation   
* **Observation Operators**: Tools for interpolation, averaging, and variable transformation between model and observation spaces   
* **Diagnostics Suite**: Validation metrics, visualization tools, and automated reporting 

### OpenDA Integration

The library will leverage OpenDA, a proven open-source data assimilation platform used operationally worldwide. OpenDA provides robust, well-tested implementations of: 

* Ensemble Kalman Filter (EnKF) with various flavors   
* Localization methods to handle spatially sparse observations   
* Parallel ensemble management for computational efficiency   
* Established interfaces to multiple modelling systems 

### Quality Assurance Pipeline

All observations will pass through a multi-stage QA process: 

* Physical bounds checking (temperature, velocity ranges)   
* Temporal consistency analysis (spike and drift detection) 

### Extensibility by Design

The modular structure ensures easy expansion: 

* **New observation types**: Drone measurements, citizen science data, or IoT sensors can be added by implementing standard interfaces   
* **Additional models**: 3D models (Delft3D, MITgcm) or other 1D models can be integrated through model wrappers   
* **Alternative DA methods**: Variational approaches or particle filters can be added alongside EnKF   
* **Multiple lakes**: Configuration files allow deployment to new water bodies without code modification   
   

## Expected Impact

By adding data assimilation to our existing capabilities, we expect: 

* **Significant reduction in model errors** for lake temperature    
* **Quantified uncertainty** enabling risk-based decision-making   
* **Reduced manual intervention** through automated parameter updating   
* **Increased usage of data by stakeholders** due to improved reliability  

## Conclusion

This project represents a logical and cost-effective evolution of Canton Ticino's lake monitoring and prediction capabilities. By adding data assimilation to our already-operational observation and modelling systems, we can dramatically improve forecast accuracy without duplicating infrastructure investments. The modular approach ensures that current operations continue uninterrupted while we enhance their capabilities. 
