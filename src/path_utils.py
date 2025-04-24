"""
Path utilities for the R5-Helicon neural simulation.

This module defines the paths to data files, simulation results, 
and figure output directories.
"""

import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the root directory (one level up from current script if in src/)
root_dir = os.path.dirname(current_dir)

# Paths to data directories
path_to_connectome = os.path.join(root_dir, 'data', 'connectome', 'connectome.npy')
path_to_files = os.path.join(root_dir, 'data', 'simulation_results', '')
path_to_figures = os.path.join(root_dir, 'docs', 'figures', '')

# Create directories if they don't exist
for directory in [os.path.dirname(path_to_connectome), path_to_files, path_to_figures]:
    os.makedirs(directory, exist_ok=True)