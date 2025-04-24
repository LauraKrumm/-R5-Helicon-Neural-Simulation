# R5-Helicon Neural Simulation

This repository contains code for simulating and analyzing the interaction between R5 and Helicon neuron populations under different time-of-day conditions. The simulation is based on the Brian2 neural simulator framework.

## System Requirements

### Software Dependencies
- Python 3.8 or higher
- brian2 2.4.2 or higher
- numpy 1.20.0 or higher
- matplotlib 3.4.0 or higher
- scipy 1.7.0 or higher
- pandas 1.3.0 or higher

### Operating Systems
The code has been tested on:
- macOS 12.0 and higher
- Ubuntu 20.04

### Hardware Requirements
- No non-standard hardware is required
- At least 8GB of RAM is recommended for faster simulations

## Installation Guide

1. Clone this repository:
```bash
git clone https://github.com/your-username/R5-Helicon-Neural-Simulation.git
cd R5-Helicon-Neural-Simulation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Typical installation time: ~5 minutes on a standard desktop computer.

## Demo

To run a demonstration of the model:

```bash
python src/run_simulation.py --demo
```

This will run a shortened simulation (1 iteration instead of 15) with predefined parameters and generate example plots.

Expected output:
- Spike raster plots for R5 and Helicon populations
- Power spectral density (PSD) plots
- Cross-correlation analysis between the two populations

Expected run time for demo: ~2 minutes on a standard desktop computer.

## Instructions for Use

### Running the Full Simulation

To run the full simulation with default parameters:

```bash
python src/run_simulation.py
```

This will run the simulation for both 'morning' and 'night' conditions with 15 iterations each.

### Customizing Parameters

You can modify parameters using command-line arguments:

```bash
# Run only morning simulation
python src/run_simulation.py --simulation_times morning

# Run only night simulation
python src/run_simulation.py --simulation_times night

# Run with 5 iterations instead of 15
python src/run_simulation.py --runtimes 5
```

Additional parameters can be modified directly in the `run_simulation.py` file:
- Various neuron parameters (tau, connectivity strengths, etc.)

### Analyzing Results

After running the simulation, results are saved to the `data/simulation_results` directory. The plotting functions in `R5_Hel_plots.py` can be used to visualize the results:

```python
import R5_Hel_plots

# Plot power spectral density
R5_Hel_plots.plot_PSD(daytime='morning', label_diver_neuron='drv_off')

# Plot correlation between R5 and Helicon populations
R5_Hel_plots.corr_coef_R5_Hel(label_diver_neuron='drv_off')
```

## Reproduction Instructions

To reproduce the quantitative results in the manuscript:

1. Run the full simulation for both day and night conditions:
   ```bash
   python src/run_simulation.py
   ```

2. After the simulation completes, the key figures from the manuscript will be automatically generated and saved to the `docs/figures` directory.