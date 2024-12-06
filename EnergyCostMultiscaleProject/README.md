# EnergyCostMultiscale_run

This repository contains scripts and configurations for simulating electrophysiological and metabolic models of the rat neocortex to compute the energy cost during a simulation.

authors: S. farina, J. Coogan, P. Kumbhar

## Simulation Types

### `sim_without_metabolism`
This simulation runs only [Neurodamus](https://doi.org/10.5281/zenodo.8075201) using the [Multiscale Orchestrator](https://github.com/BlueBrain/MultiscaleRun/tree/main).

### `sim_with_metabolism`
This simulation integrates the electrophysiological model with metabolic dynamics, based on the approach described in Shichkova et al. (2023).

## How to Run the Simulations

Follow the instructions provided in the [Multiscale Orchestrator Documentation](https://multiscalerun.readthedocs.io/stable/).

### Steps:
1. Create a new simulation using the `rat_sscxS1HL_V10` template.
2. Replace the following files in the template with the ones provided in this repository:
   - `circuit_config.json`
   - `node_sets.json`
   - `simulation_config.json`
   - `simulation.sbatch`

For further details, refer to Farina et al. (2025).

## How to Compute the Simulation's Power Usage

1. Submit the simulation job as usual.
2. Wait for the job to start. **Note:** This requires monitoring the job status, but it is manageable as it only needs to be done a few times.
3. Once the job has started, run the attached `power_mon.py` script.
4. The Python script will generate a CSV file named `power_usage.csv` containing the power usage data.

## References
- Shichkova et al. (2023) "Breakdown and rejuvenation of aging brain energy metabolism"
- Farina et al. (2024) "A Multiscale Electro-Metabolic Model of a Rat Neocortical Circuit Reveals the Impact of Ageing on Central Cortical Layers"
- Neurodamus: https://doi.org/10.5281/zenodo.8075201