## 1. Computer Setup
 
- Create a GitHub account and request access to the [alplakes-da](https://github.com/eawag-surface-waters-research/alplakes-da) repository
- Request an admin user account from IT to enable installation of custom programs
- Install and configure [Claude Code](https://docs.anthropic.com/en/docs/claude-code) in Visual Studio Code
- Install and configure [Docker](https://www.docker.com/)
 
## 2. Literature Review
 
- Use an LLM of your choice to summarise papers from the literature review; check for any gaps in coverage and include the summaries in the documentation
- Select the top 5 most relevant papers and produce a comparison of their methods and results. Document your findings in literature-review.md 
 
## 3. Analysis of Proposal

- Critically review the proposed method below, outlining the pros and cons of editing forcing data rather than modifying the model parameters directly
- Explore how this approach could be extended to 3D hydrodynamic models

### Proposed Simstrat Data Assimilation Approach (MVP)
 
The current plan for the Simstrat minimum viable product is as follows:
 
1. Generate model input files covering the entire simulation period
2. Import in-situ observation data, down-sampling high-resolution datasets to a manageable temporal resolution
3. Divide the simulation into chunks defined by the periods between in-situ data points
4. Use [OpenDA](https://www.openda.org/) to produce an ensemble of forcing files from the Simstrat inputs — this will require information on meteorological input uncertainty and some experimentation
5. Run all ensemble members and select the best-performing configuration
6. Repeat for each chunk, recording the adjustments made to the meteorological data
 
## 3. Experience with Modelling Software
 
- Complete the PEAK course exercises on running Simstrat. [download](https://eawagch-my.sharepoint.com/:f:/g/personal/james_runnalls_eawag_ch/IgDtFcNzTR_hT5E1jmzMEq9hAfsFJZbe1CyR_W5mVF_8EE8?e=kVcqDB)
- Run a model simulation using [operational-simstrat](https://github.com/Eawag-AppliedSystemAnalysis/operational-simstrat)
- Run a calibration using [operational-simstrat](https://github.com/Eawag-AppliedSystemAnalysis/operational-simstrat)
- Run a 3D model using [alplakes-simulations](https://github.com/eawag-surface-waters-research/alplakes-simulations)
 
## 4. Planning
 
- Review the structure of [lake-calibrator](https://github.com/eawag-surface-waters-research/lake-calibrator) — consider how this structure could be adapted for the data assimilation use case and whether it serves as a good starting point
- Create a task breakdown for delivering the MVP, covering areas such as meteorological uncertainty quantification, code development, and in-situ data collection
 
