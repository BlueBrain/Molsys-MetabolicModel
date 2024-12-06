# A Multiscale Electro-Metabolic Model of a Rat Neocortical Circuit

This repository contains all the information related to the paper:

**"A Multiscale Electro-Metabolic Model of a Rat Neocortical Circuit Reveals the Impact of Ageing on Central Cortical Layers"**  
by Sofia Farina, Alessandro Cattabiani, Darshan Mandge, Polina Shichkova, James B. Isbister, Jean Jacquemier, James G. King, Henry Markram, and Daniel Keller.

---

## Contents

### 1. Figures
- The `./Figures` folder contains Jupyter notebooks for generating the figures presented in the paper.
- The `./Figures/Code4DataExtraction` folder contains Python scripts used to extract data from the simulations.

### 2. Config
- `./config/node_sets.json`: Contains the IDs of the 27,962 neurons in the microcircuit (testNGVSSCX is the name of the neuron set).
- `./config/circuit_config.json`: Contains the configuration to build the circuit. You will need to provide paths to:
  - **Neuronal morphologies**: [Download here](https://zenodo.org/records/8155899).
  - **Synaptic connections**: [Download here](https://zenodo.org/records/11113043).
  - **Neuronal electrical models**: Use the `emodels` folder in this repository.

### 3. emodels
- This folder contains the electrical models of the neurons.

### 4. Metabolism
- Contains the Julia code for the Young and Aged simulations of the metabolic model, based on Shichkova et al. (2023). These simulations are also integrated into the `Multiscale Orchestrator` (see below).
- Includes the log files for the simulations described in the article.

### 5. Simulations
- The `./simulations` folder contains `simulation_config.json` files for the three simulations presented in the paper. These files define the settings and stimuli to be run using the `Multiscale Orchestrator`.

---

## Simulations and Data
The results of the three simulations presented in the article, along with the extracted data used to generate the figures, can be downloaded here: [10.5281/zenodo.14187063](https://doi.org/10.5281/zenodo.14187063).

---
## Instructions to Run Notebooks in the Figures Folder

1. Create a Python environment using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   
---
## Instructions to Run Simulations Using the Multiscale Orchestrator

1. **Set Up the Multiscale Orchestrator**  
   Follow the instructions in the [Multiscale Orchestrator documentation](https://multiscalerun.readthedocs.io/stable/) to set up the environment. Use the template provided in the repository:  
   [multiscale_run/templates/rat_sscxS1HL_V10](https://github.com/BlueBrain/MultiscaleRun/tree/main).

2. **Generate the Required Files**  
   After setting up a new simulation using the template, the following files will be generated:
   ```bash
   ├── circuit_config.json
   ├── msr_config.json
   ├── node_sets.json
   └── simulation_config.json
   ├── postproc.ipynb
   └── simulation.sbatch

4. **Update the Files**  
Replace the generated files with the corresponding ones provided in this repository:
   ```bash
   ├── ./config/circuit_config.json (Before using it update the path with emodels morphologies, synaptic connection)
   ├── ./config/node_sets.json (ensure you are using multiscale_run/templates/rat_sscxS1HL_V10)
   └── ./simulations/simulation_config_young.json
   └── ./simulations/simulation.sbatch

## Additional information
mod file for NEURON can be found in this [repository](https://github.com/BlueBrain/neurodamus-models/tree/main/neocortex/mod/metabolism)
