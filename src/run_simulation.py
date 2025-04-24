#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R5-Helicon Neural Simulation

This script runs simulations of coupled R5 and Helicon neuron populations
under different time-of-day conditions (morning and night).

"""

import argparse
import os
import numpy as np
from brian2 import *

# Import local modules
from R5_Hel_definitions import run_and_plot, connectome, creat_visual_inputs
import R5_Hel_plots
import path_utils

def main(args):
    """
    Run the R5-Helicon neural simulation with the specified parameters.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(path_utils.path_to_files), exist_ok=True)
    os.makedirs(path_utils.path_to_figures, exist_ok=True)
    
    # Set parameters based on demo mode
    save_results = 'true'  # only save files if 'true'
    if args.demo:
        print("Running in demo mode with reduced iterations")
        runtimes = 1
        simulation_times = ['morning']
    else:
        runtimes = args.runtimes
        simulation_times = args.simulation_times.split(',')
    
    print(f"Running simulation with {runtimes} iterations for conditions: {simulation_times}")

    for sim_time in simulation_times:
        print(f"\nSimulating {sim_time} condition...")

        # Load connectome
        R5_connectome, R5_Hel_connectome, Hel_R5_connectome, Hel_Hel_connectome = connectome()

        # R5 PARAMETERS
        N_r5 = 20
        tau_syn_ex = 4 * ms
        tau_syn_inh = 20 * ms
        I_ex_R5 = 0.002 * R5_connectome
        I_inh_R5 = -0.0004 * R5_connectome

        # HELICON PARAMETERS
        N_hel = 4
        tau_hel = 100 * ms
        sig_hel = 5 * mvolt / ms
        if sim_time == 'morning':
            I_0_hel = 4.5 * mvolt / ms
        if sim_time == 'night':
            I_0_hel = -0.75 * mvolt / ms
        I_Hel = 0.00016 * Hel_Hel_connectome

        # Visual input simulation
        vis_ampl = 0
        move_mean = 0
        vis_filt = creat_visual_inputs()

        # R5 - HELICON PARAMETERS
        if sim_time == 'morning':
            I_ex_R5_Hel = 0.008 * R5_Hel_connectome
            I_inh_R5_Hel = -0.0016 * R5_Hel_connectome
        if sim_time == 'night':
            I_ex_R5_Hel = 0.008 * R5_Hel_connectome
            I_inh_R5_Hel = 0 * R5_Hel_connectome

        # HELICON - R5 PARAMETERS
        I_Hel_R5 = 0.00032 * Hel_R5_connectome

        print('Hel_Hel_connectome', np.mean(Hel_Hel_connectome), np.std(Hel_Hel_connectome))
        print('Hel_R5_connectome', np.mean(Hel_R5_connectome), np.std(Hel_R5_connectome))
        print('I_Hel_R5', np.mean(I_Hel_R5), np.std(I_Hel_R5))
        print('I_Hel', np.mean(I_Hel), np.std(I_Hel))

        # DRIVER NEURON
        tau_from_dr = 4 * ms
        I_dr_R5 = 0
        I_dr_Hel = 0
        N_drive = 1
        N = N_r5 + N_hel + N_drive
        N_r5_arr = np.arange(0, N_r5, 1)
        N_hel_arr = np.arange(N_r5, N_r5 + N_hel, 1)
        label_diver_neuron = 'drv_off'

        dt = 1
        runtime = 60000 if not args.demo else 10000  # Shorter runtime for demo
        t_cut = int((runtime / dt) / 3)

        print(f'Simulation time: {sim_time}')
        offsets = []
        Hel_R5_corr_coefs = []
        mean_corr_coefs_r5 = []
        mean_corr_coefs_hel = []
        
        for r_num in range(runtimes):
            print(f'Run {r_num+1}/{runtimes}')
            offset, Hel_R5_corr_coef, mean_corr_coef_r5, mean_corr_coef_hel, spikemon_dr_neur = \
                run_and_plot(N_r5, tau_syn_ex, tau_syn_inh, I_ex_R5, I_inh_R5, N_hel, tau_hel, sig_hel, I_0_hel, I_Hel,
                             I_ex_R5_Hel, I_inh_R5_Hel, I_Hel_R5, tau_from_dr, I_dr_R5, I_dr_Hel, runtime, dt, sim_time,
                             t_cut, r_num, save_results, label_diver_neuron, vis_filt, vis_ampl, move_mean)

            offsets.append(offset)
            Hel_R5_corr_coefs.append(Hel_R5_corr_coef)
            mean_corr_coefs_r5.append(mean_corr_coef_r5)
            mean_corr_coefs_hel.append(mean_corr_coef_hel)

        if save_results == 'true':
            np.save(path_utils.path_to_files + f'offsets_{sim_time}_{label_diver_neuron}.npy', offsets)
            np.save(path_utils.path_to_files + f'Hel_R5_corr_coefs_{sim_time}_{label_diver_neuron}.npy', Hel_R5_corr_coefs)
            np.save(path_utils.path_to_files + f'mean_corr_coefs_r5_{sim_time}_{label_diver_neuron}.npy', mean_corr_coefs_r5)
            np.save(path_utils.path_to_files + f'mean_corr_coefs_hel_{sim_time}_{label_diver_neuron}.npy', mean_corr_coefs_hel)

        print(f'Offsets_{sim_time}', offsets)

    # Generate plots
    print("\nGenerating plots...")
    if 'morning' in simulation_times:
        R5_Hel_plots.plot_PSD(daytime='morning', label_diver_neuron='drv_off')
        R5_Hel_plots.plot_PSD_db(daytime='morning', label_diver_neuron='drv_off')
        R5_Hel_plots.compound_signal_R5_Hel(daytime='morning', label_diver_neuron='drv_off')
        
    if 'night' in simulation_times:
        R5_Hel_plots.plot_PSD(daytime='night', label_diver_neuron='drv_off')
        R5_Hel_plots.plot_PSD_db(daytime='night', label_diver_neuron='drv_off')
        R5_Hel_plots.compound_signal_R5_Hel(daytime='night', label_diver_neuron='drv_off')

    # Plot comparison plots if both conditions were simulated
    if 'morning' in simulation_times and 'night' in simulation_times:
        R5_Hel_plots.corr_coef_R5_Hel(label_diver_neuron='drv_off')
        R5_Hel_plots.plot_day_night(type='r5', label_diver_neuron='drv_off')
        R5_Hel_plots.plot_day_night(type='hel', label_diver_neuron='drv_off')

    print("Simulation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='R5-Helicon Neural Simulation')
    parser.add_argument('--demo', action='store_true', help='Run a short demo with reduced iterations')
    parser.add_argument('--runtimes', type=int, default=15, help='Number of simulation iterations')
    parser.add_argument('--simulation_times', type=str, default='night,morning', 
                        help='Comma-separated list of conditions to simulate')
    args = parser.parse_args()
    
    main(args)