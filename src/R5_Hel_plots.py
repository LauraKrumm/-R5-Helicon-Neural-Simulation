"""
R5-Helicon Neural Simulation: Plotting Functions

This module provides visualization and analysis functions for R5 and Helicon
neuron simulation data, including spike plots, power spectral density,
correlation analyses, and compound signal plots.

The plots maintain consistent styling with cadetblue for R5 neurons and
mediumslateblue for Helicon neurons.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from pylab import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import path_utils
import pandas as pd

# Set consistent plotting parameters
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2


def plot_spike_raster_plot(spikemon_r5, spikemon_hel, daytime, r_num):
    """
    Generate a spike raster plot showing R5 and Helicon neuron activity.
    
    Parameters
    ----------
    spikemon_r5 : brian2.SpikeMonitor
        Spike monitor for R5 neurons
    spikemon_hel : brian2.SpikeMonitor
        Spike monitor for Helicon neurons
    daytime : str
        Time condition ('morning' or 'night')
    r_num : int
        Run number identifier
        
    Returns
    -------
    None
        The figure is saved to disk
    """
    plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(11, 1)
    ax1 = plt.subplot(gs[0:6, :])
    
    # Plot R5 spikes in cadetblue
    plt.plot(spikemon_r5.t, spikemon_r5.i, 'o', color='cadetblue')
    
    # Plot Helicon spikes in mediumslateblue (offset by 20 for visualization)
    plt.plot(spikemon_hel.t, spikemon_hel.i + 20, 'o', color='mediumslateblue')
    
    # Configure axis appearance
    plt.tick_params(left=False, labelleft=False)
    plt.xticks(fontsize=8)
    ax1.set_xlim(0, 10)  # Show first 10 seconds
    ax1.set_xlabel('time [s]', fontsize=13)
    
    # Customize spine appearance
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.spines["left"].set_visible(False)
    
    plt.title(f'Spikes - {daytime}', fontsize=13, loc='left')
    
    # Save the figure
    output_path = f'{path_utils.path_to_figures}spikerasterplt_{daytime}_{r_num}.png'
    plt.savefig(output_path, dpi=300)


def plot_connectome(connectome):
    """
    Visualize the connectome matrix showing connections between neuron populations.
    
    Parameters
    ----------
    connectome : ndarray
        Connectome matrix showing connection strengths
        
    Returns
    -------
    None
        The figure is saved to disk
    """
    plt.figure()
    ax = plt.gca()
    plt.imshow(connectome, cmap='Blues')
    
    # Add labels for different connection regions
    ax.text(10, 10, 'R5-R5', fontsize=12, verticalalignment='center', 
            horizontalalignment='center', color='k')
    ax.text(21.6, 10, 'R5-ExR1', fontsize=10, verticalalignment='center', 
            horizontalalignment='center', color='w')
    ax.text(10, 22, 'ExR1-R5', fontsize=12, verticalalignment='center', 
            horizontalalignment='center', color='k')
    ax.text(21.6, 22, 'ExR1-ExR1', fontsize=8, verticalalignment='center', 
            horizontalalignment='center', color='k')
    
    # Add colorbar and title
    clb = plt.colorbar()
    clb.set_label('# synaptic connections', fontsize=15)
    plt.title('connectome', fontsize=15)
    
    # Save the figure
    plt.savefig(path_utils.path_to_figures + 'connectome.svg')


def plot_PSD(daytime, label_diver_neuron):
    """
    Plot power spectral density (PSD) for R5 and Helicon populations.
    
    Parameters
    ----------
    daytime : str
        Time condition ('morning' or 'night')
    label_diver_neuron : str
        Label for the driver neuron mode
        
    Returns
    -------
    None
        The figure is saved to disk
    """
    # Load data files
    frequency = glob.glob(path_utils.path_to_files + f'frequencies_{daytime}_{label_diver_neuron}.npy')
    PSD_R5_files = glob.glob(path_utils.path_to_files + f'PSD_r5_{daytime}_{label_diver_neuron}_*.npy')
    PSD_Hel_files = glob.glob(path_utils.path_to_files + f'PSD_hel_{daytime}_{label_diver_neuron}_*.npy')

    f = np.load(frequency[0])
    
    # Load all PSD data from multiple runs
    PSD_R5 = np.zeros((len(PSD_R5_files), len(f)))
    PSD_Hel = np.zeros((len(PSD_Hel_files), len(f)))
    for n in range(len(PSD_R5_files)):
        PSD_R5[n, :] = np.load(PSD_R5_files[n])
        PSD_Hel[n, :] = np.load(PSD_Hel_files[n])

    # Create figure with custom layout
    fig = plt.figure(figsize=(4.8, 4))
    axs = fig.add_axes([0.25, 0.2, 0.7, 0.75])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    # Set normalization factor depending on daytime
    if daytime == 'night':
        norm = 4
    else:
        norm = 1

    # Plot Helicon PSD with uncertainty
    axs.plot(f[1:-1], (np.mean(PSD_Hel, axis=0)[1:-1]) / 4, 
             color='mediumslateblue', linewidth=4, label='Helicon')
    axs.fill_between(f[1:-1],
                     (np.mean(PSD_Hel, axis=0)[1:-1] - np.std(PSD_Hel, axis=0)[1:-1]) / 4,
                     (np.mean(PSD_Hel, axis=0)[1:-1] + np.std(PSD_Hel, axis=0)[1:-1]) / 4,
                     color='mediumslateblue', alpha=0.3)
    
    # Plot R5 PSD with uncertainty
    axs.plot(f[1:-1], (np.mean(PSD_R5, axis=0)[1:-1]) / 20 / norm, 
             color='cadetblue', linewidth=2.5, label='R5')
    axs.fill_between(f[1:-1],
                     (np.mean(PSD_R5, axis=0)[1:-1] - np.std(PSD_R5, axis=0)[1:-1]) / 20 / norm,
                     (np.mean(PSD_R5, axis=0)[1:-1] + np.std(PSD_R5, axis=0)[1:-1]) / 20 / norm,
                     color='cadetblue', alpha=0.4)

    # Set axis labels and formatting
    axs.set_xlabel('frequency [Hz]', fontsize=22)
    axs.set_ylabel('average PSD', fontsize=22)
    axs.yaxis.set_tick_params(which='major', size=7, width=2)
    axs.xaxis.set_tick_params(which='major', size=7, width=2)
    axs.set_xlim(0.3, 3)
    
    # Set y-axis limits based on daytime
    if daytime == 'night':
        axs.set_ylim(-5, 700)
    else:
        axs.set_ylim(-5, 100)
    
    # Save the figure
    plt.savefig(path_utils.path_to_figures + f'PSD_{daytime}_{label_diver_neuron}.svg', format='svg')


def corr_coef_R5_Hel(label_diver_neuron):
    """
    Plot correlation coefficients between R5 and Helicon neurons for morning vs night.
    
    Parameters
    ----------
    label_diver_neuron : str
        Label for the driver neuron mode
        
    Returns
    -------
    None
        The figure is saved to disk
    """
    # Load correlation coefficient data
    corr_coef_night = np.load(path_utils.path_to_files + f'Hel_R5_corr_coefs_night_{label_diver_neuron}.npy')
    print('corr_coef_night:', corr_coef_night)
    corr_coef_day = np.load(path_utils.path_to_files + f'Hel_R5_corr_coefs_morning_{label_diver_neuron}.npy')

    # Create bar plot comparing morning and night
    fig = plt.figure(figsize=(4, 4))
    axs = fig.add_axes([0.3, 0.2, 0.7, 0.7])
    x = ['morning', 'night']
    y = [np.mean(corr_coef_day), np.mean(corr_coef_night)]
    c = [np.std(corr_coef_day), np.std(corr_coef_night)]

    # Format plot appearance
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.bar(x, y, color=['red', 'midnightblue'], width=0.6, alpha=0.8)
    axs.errorbar(x, y, yerr=c, fmt=".", color="k")
    axs.set_ylim(0, 1)
    
    # Format y-axis ticks
    axs.yaxis.set_tick_params(which='major', size=7, width=2)
    axs.yaxis.set_tick_params(which='minor', size=5, width=2)
    axs.yaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(0, 1.2, 0.2)))
    axs.yaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(0, 1, 0.1)))
    
    axs.set_ylabel('correlation coefficient')
    plt.title('theory', fontsize=22, loc='left')
    
    # Save the figure
    plt.savefig(path_utils.path_to_figures + f'corr_coef_R5_Hel_label_diver_neuron.svg', format='svg')


def compound_signal_R5_Hel(daytime, label_diver_neuron):
    """
    Plot compound signals (summed activity) for R5 and Helicon populations.
    
    Parameters
    ----------
    daytime : str
        Time condition ('morning' or 'night')
    label_diver_neuron : str
        Label for the driver neuron mode
        
    Returns
    -------
    None
        The figures are saved to disk
    """
    # Load simulation data
    runtime = np.load(path_utils.path_to_files + f'runtime_{daytime}_{label_diver_neuron}.npy')
    conv_matrix_hel = np.load(path_utils.path_to_files + f'compound_hel_{daytime}_{label_diver_neuron}_0.npy')
    conv_matrix_r5 = np.load(path_utils.path_to_files + f'compound_r5_{daytime}_{label_diver_neuron}_0.npy')

    # Configuration for R5 and Helicon plots
    plot_for = ['R5', 'Hel']
    matrices = [conv_matrix_r5, conv_matrix_hel]
    normalize = [20, 4]
    y_lim_min = [-3, 0]

    # Create separate plots for R5 and Helicon
    for p, conv_m, norm, ymin in zip(plot_for, matrices, normalize, y_lim_min):
        if p == 'R5':
            colour = 'navy'
        if p == 'Hel':
            colour = 'mediumslateblue'
            
        # Create figure with custom layout
        fig = plt.figure(figsize=(5, 2))
        ax2 = fig.add_axes([0.2, 0.4, 0.7, 0.5])
        
        # Plot normalized compound signal
        plt.plot(np.linspace(0, runtime, len(np.sum(conv_m, axis=0))) * 0.001, 
                 np.sum(conv_m, axis=0) / norm,
                 linewidth=3, color=colour, label='Helicon')
        
        # Configure plot appearance
        plt.tick_params(bottom=False, labelbottom=False)
        ax2.set_xlim(0, (runtime * 10 ** -3) - 10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.yaxis.set_tick_params(which='major', size=7, width=2)
        ax2.yaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(0, 30, 10)))
        ax2.set_ylim(ymin, 20)
        ax2.set_xlim(20, 30)
        
        # Save the figure
        plt.savefig(path_utils.path_to_figures + f'compund_signal_{p}_{daytime}_{label_diver_neuron}.svg', 
                    format='svg')


def plot_day_night(type, label_diver_neuron='drv_off'):
    """
    Compare power spectral density between morning and night for a specific neuron type.
    
    Parameters
    ----------
    type : str
        Neuron type to plot ('r5' or 'hel')
    label_diver_neuron : str, optional
        Label for the driver neuron mode, by default 'drv_off'
        
    Returns
    -------
    None
        The figure is saved to disk
    """
    # Load data files
    frequency = glob.glob(path_utils.path_to_files + f'frequencies_morning_{label_diver_neuron}.npy')
    PSD_files_day = glob.glob(path_utils.path_to_files + f'PSD_{type}_morning_{label_diver_neuron}_*.npy')
    PSD_files_night = glob.glob(path_utils.path_to_files + f'PSD_{type}_night_{label_diver_neuron}_*.npy')

    # Set color based on neuron type
    if type == 'hel':
        colour = 'mediumslateblue'
    else:
        colour = 'cadetblue'

    # Load frequency data
    f = np.load(frequency[0])
    
    # Load PSD data for morning and night
    PSD_day = np.zeros((len(PSD_files_day), len(f)))
    PSD_night = np.zeros((len(PSD_files_night), len(f)))
    for n in range(len(PSD_files_day)):
        PSD_day[n, :] = np.load(PSD_files_day[n])
        PSD_night[n, :] = np.load(PSD_files_night[n])

    # Create figure with custom layout
    fig = plt.figure(figsize=(4.8, 4))
    axs = fig.add_axes([0.25, 0.2, 0.7, 0.75])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    # Set normalization factors based on neuron type
    if type == 'r5':
        norm = 4
        norm2 = 20
    else:
        norm = 1
        norm2 = 4

    # Plot morning PSD with uncertainty
    axs.plot(f[1:-1], (np.mean(PSD_day, axis=0)[1:-1]) / norm2, 
             color=colour, linewidth=4, label='Helicon')
    axs.fill_between(f[1:-1],
                     (np.mean(PSD_day, axis=0)[1:-1] - np.std(PSD_day, axis=0)[1:-1])/ norm2,
                     (np.mean(PSD_day, axis=0)[1:-1] + np.std(PSD_day, axis=0)[1:-1])/ norm2,
                     color=colour, alpha=0.3)

    # Plot night PSD with uncertainty
    axs.plot(f[1:-1], (np.mean(PSD_night, axis=0)[1:-1]) / norm2 / norm, 
             color=colour, linewidth=2.5, label='R5')
    axs.fill_between(f[1:-1],
                     (np.mean(PSD_night, axis=0)[1:-1] - np.std(PSD_night, axis=0)[1:-1]) / norm2 / norm,
                     (np.mean(PSD_night, axis=0)[1:-1] + np.std(PSD_night, axis=0)[1:-1]) / norm2 / norm,
                     color=colour, alpha=0.4)

    # Configure plot appearance
    axs.set_xlabel('frequency [Hz]', fontsize=22)
    axs.set_ylabel('average PSD', fontsize=22)
    axs.yaxis.set_tick_params(which='major', size=7, width=2)
    axs.xaxis.set_tick_params(which='major', size=7, width=2)
    axs.set_xlim(0.3, 3)
    
    # Set y-axis limits based on neuron type
    if type == 'hel':
        axs.set_ylim(-5, 700)
    else:
        axs.set_ylim(-5, 700)
    
    # Save the figure
    plt.savefig(path_utils.path_to_figures + f'PSD_{type}_{label_diver_neuron}.svg', format='svg')


def plot_PSD_db(daytime, label_diver_neuron):
    """
    Plot power spectral density (PSD) in decibels for R5 and Helicon populations.
    
    Parameters
    ----------
    daytime : str
        Time condition ('morning' or 'night')
    label_diver_neuron : str
        Label for the driver neuron mode
        
    Returns
    -------
    None
        The figure is saved to disk
    """
    # Load data files
    frequency = glob.glob(path_utils.path_to_files + f'frequencies_{daytime}_{label_diver_neuron}.npy')
    PSD_R5_files = glob.glob(path_utils.path_to_files + f'PSD_r5_{daytime}_{label_diver_neuron}_*.npy')
    PSD_Hel_files = glob.glob(path_utils.path_to_files + f'PSD_hel_{daytime}_{label_diver_neuron}_*.npy')

    # Load frequency data
    f = np.load(frequency[0])
    
    # Load PSD data from multiple runs
    PSD_R5 = np.zeros((len(PSD_R5_files), len(f)))
    PSD_Hel = np.zeros((len(PSD_Hel_files), len(f)))
    for n in range(len(PSD_R5_files)):
        PSD_R5[n, :] = np.load(PSD_R5_files[n])
        PSD_Hel[n, :] = np.load(PSD_Hel_files[n])

    # Create figure with custom layout
    fig = plt.figure(figsize=(4.8, 4))
    axs = fig.add_axes([0.25, 0.2, 0.7, 0.75])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    # Convert to dB scale (10*log10) and plot Helicon PSD with uncertainty
    axs.plot(f[1:-1], np.log10((np.mean(PSD_Hel, axis=0)[1:-1]) / 4) * 10, 
             color='mediumslateblue', linewidth=4, label='Helicon')
    axs.fill_between(f[1:-1],
                     np.log10((np.mean(PSD_Hel, axis=0)[1:-1] - np.std(PSD_Hel, axis=0)[1:-1]) / 4) * 10,
                     np.log10((np.mean(PSD_Hel, axis=0)[1:-1] + np.std(PSD_Hel, axis=0)[1:-1]) / 4) * 10,
                     color='mediumslateblue', alpha=0.3)
    
    # Convert to dB scale and plot R5 PSD with uncertainty
    axs.plot(f[1:-1], np.log10((np.mean(PSD_R5, axis=0)[1:-1]) / 20) * 10, 
             color='cadetblue', linewidth=2.5, label='R5')
    axs.fill_between(f[1:-1],
                     np.log10((np.mean(PSD_R5, axis=0)[1:-1] - np.std(PSD_R5, axis=0)[1:-1]) / 20) * 10,
                     np.log10((np.mean(PSD_R5, axis=0)[1:-1] + np.std(PSD_R5, axis=0)[1:-1]) / 20) * 10,
                     color='cadetblue', alpha=0.4)

    # Configure plot appearance
    axs.set_xlabel('frequency [Hz]', fontsize=22)
    axs.set_ylabel('average PSD (dB)', fontsize=22)
    axs.yaxis.set_tick_params(which='major', size=7, width=2)
    axs.xaxis.set_tick_params(which='major', size=7, width=2)
    axs.xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(0, 5, 1)))
    axs.set_xlim(0.3, 3)
    axs.set_ylim(-5, 40)
    
    # Save the figure
    plt.savefig(path_utils.path_to_figures + f'PSD_{daytime}_{label_diver_neuron}_db.svg', format='svg')