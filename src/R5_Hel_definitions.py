"""
R5-Helicon Neural Simulation: Core Functions

This module provides the primary simulation functions for R5 and Helicon
neuron populations, including neuron models, connectivity setup, visual inputs,
and simulation analysis functions.

The module implements Izhikevich-type neuron models for both R5 and Helicon
populations, with parameters that differ between morning and night conditions.
"""

import matplotlib.pyplot as plt
from brian2 import *
import numpy as np
import matplotlib.mlab as ml
from random import randrange
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, lfilter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import random

# Import local modules
import path_utils
from R5_Hel_plots import plot_spike_raster_plot


def connectome():
    """
    Load and process the neural connectome for the simulation.
    
    This function loads a pre-defined connectome from the specified path and
    extracts connectivity matrices for different neuron populations:
    - R5-R5 connections
    - R5-Helicon connections
    - Helicon-R5 connections
    - Helicon-Helicon connections
    
    Returns
    -------
    tuple
        Four 1D arrays representing the flattened connectivity matrices:
        (R5_connectome, R5_Hel_connectome, Hel_R5_connectome, Hel_Hel_connectome)
    """
    connectome_tmp = np.load(path_utils.path_to_connectome)
    connectome = np.vstack((connectome_tmp, np.zeros((len(connectome_tmp)))))
    connectome = np.hstack((connectome, np.zeros((len(connectome), 1))))

    # Extract R5-R5 connections (20x20 neurons)
    R5 = connectome[:20, :20]
    R5_connectome = R5.flatten()
    
    # Extract R5-Helicon connections (20x4 neurons)
    R5_Hel = connectome[0:20, 20:24]
    R5_Hel_connectome = R5_Hel.flatten()
    
    # Extract Helicon-R5 connections (4x20 neurons)
    Hel_R5 = connectome[20:24, 0:20]
    Hel_R5_connectome = Hel_R5.flatten()
    
    # Extract Helicon-Helicon connections (4x4 neurons)
    Hel_Hel = connectome[20:24, 20:24]
    Hel_Hel_connectome = Hel_Hel.flatten()
    
    return R5_connectome, R5_Hel_connectome, Hel_R5_connectome, Hel_Hel_connectome


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a butterworth bandpass filter.
    
    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    fs : float
        Sampling rate in Hz
    order : int, optional
        Filter order, by default 5
        
    Returns
    -------
    tuple
        Butterworth filter coefficients (b, a)
    """
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """
    Apply a butterworth bandpass filter to data.
    
    Parameters
    ----------
    data : ndarray
        Input signal to be filtered
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    fs : float
        Sampling rate in Hz
    order : int
        Filter order
        
    Returns
    -------
    ndarray
        Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def creat_visual_inputs():
    """
    Create filtered white noise for visual inputs to the simulation.
    
    Generates Gaussian white noise and applies a bandpass filter to limit
    the frequency content to 0.1-5 Hz range.
    
    Returns
    -------
    ndarray
        Filtered white noise signal for visual input
    """
    # Create white noise for the duration of the simulations (60 sec)
    dt = 1  # ms
    runtime = 60000  # 60 sec
    mean = 0
    std = 10
    samples = numpy.random.normal(mean, std, size=runtime * dt)

    # Bandpass Filter white noise between 0.1 and 5 Hz
    fs = 1000.0  # Sample rate (Hz)
    lowcut = 0.1  # Low cutoff frequency (Hz)
    highcut = 5   # High cutoff frequency (Hz)
    vis_filt = butter_bandpass_filter(samples, lowcut, highcut, fs, order=2)
    
    return vis_filt


def _neurons(N_r5, tau_syn_ex, tau_syn_inh, I_ex_R5, I_inh_R5, N_hel, tau_hel, sig_hel, I_0_hel, I_Hel, I_ex_R5_Hel,
            I_inh_R5_Hel, I_Hel_R5, tau_from_dr, I_dr_R5, I_dr_Hel, runtime, dt, daytime, vis_filt, vis_ampl,
             move_mean, r_num):
    """
    Create and simulate the neural network with R5 and Helicon populations.
    
    This function sets up the neuron models, their connections, and runs
    the simulation for the specified duration. It uses the Brian2 simulator
    for neural dynamics.
    
    Parameters
    ----------
    N_r5 : int
        Number of R5 neurons
    tau_syn_ex : brian2.Quantity
        Time constant for excitatory synapses
    tau_syn_inh : brian2.Quantity
        Time constant for inhibitory synapses
    I_ex_R5 : ndarray
        Excitatory connection strengths within R5 population
    I_inh_R5 : ndarray
        Inhibitory connection strengths within R5 population
    N_hel : int
        Number of Helicon neurons
    tau_hel : brian2.Quantity
        Time constant for Helicon neurons
    sig_hel : brian2.Quantity
        Noise amplitude for Helicon neurons
    I_0_hel : brian2.Quantity
        Base current for Helicon neurons
    I_Hel : ndarray
        Connection strengths within Helicon population
    I_ex_R5_Hel : ndarray
        Excitatory connection strengths from R5 to Helicon
    I_inh_R5_Hel : ndarray
        Inhibitory connection strengths from R5 to Helicon
    I_Hel_R5 : ndarray
        Connection strengths from Helicon to R5
    tau_from_dr : brian2.Quantity
        Time constant for driver neuron connections
    I_dr_R5 : float
        Connection strength from driver to R5
    I_dr_Hel : float
        Connection strength from driver to Helicon
    runtime : int
        Simulation duration in ms
    dt : int
        Simulation time step in ms
    daytime : str
        Simulation mode ('morning' or 'night')
    vis_filt : ndarray
        Visual input filter
    vis_ampl : float
        Visual input amplitude
    move_mean : float
        Mean value for visual input
    r_num : int
        Run number identifier
        
    Returns
    -------
    tuple
        All simulation monitors and data objects:
        (spikemon_r5, spikemon_hel, M_r5_v, M_g_syn_inh, M_g_syn_ex, M_hel_v, 
        M_g_from_r5_inh, M_g_from_r5_ex, M_dr, M_g_from_hel, spikemon_dr_neur)
    """
    start_scope()
    defaultclock.dt = 1 * ms

    ####################
    #   R5 neurons     #
    ####################

    # Set Izhikevich model parameters based on daytime
    if daytime == 'morning':
        I_0_r5 = 0.34
        sig_r5 = 0.02
        a_r5 = 0.02 / ms
        b_r5 = 0.2 / ms
        c_r5 = -65 * mvolt
        d_r5 = 6 * mvolt / ms
    if daytime == 'night':
        I_0_r5 = 0.3
        sig_r5 = 0.08
        a_r5 = 0.02 / ms
        b_r5 = 0.3 / ms
        c_r5 = -50 * mvolt
        d_r5 = 1.6 * mvolt / ms

    # R5 Izhikevich equation and conductance coupling
    eqs_r5 = '''
        dv/dt = ((0.04/ms/mvolt)*v**2+(5/ms)*v+140*mvolt/ms-u)*0.05+I+g_syn_ex+g_syn_inh+g_from_hel+g_dr : volt 
        du/dt = (a_r5*(b_r5*v-u))*0.05 : volt/second 
        dI_syn_R5_ex/dt = -I_syn_R5_ex/tau_syn_ex :volt/second
        dg_syn_ex/dt = (I_syn_R5_ex-g_syn_ex)/tau_syn_ex :volt/second
        dI_syn_R5_inh/dt = -I_syn_R5_inh/tau_syn_inh :volt/second
        dg_syn_inh/dt = (I_syn_R5_inh-g_syn_inh)/tau_syn_inh :volt/second    
        dI_from_hel/dt = -I_from_hel/tau_hel :volt/second
        dg_from_hel/dt = (I_from_hel-g_from_hel)/tau_hel :volt/second
        I :volt/second
        dI_from_dr/dt = -I_from_dr/tau_from_dr :volt/second
        dg_dr/dt = (I_from_dr-g_dr)/tau_from_dr :volt/second
        '''
        
    # R5 reset of Izhikevich model
    reset_r5 = '''
        v = c_r5
        u += d_r5
        I = I_0_r5*mvolt/ms + randn()*sig_r5*mvolt/ms 
        '''
        
    # Create population of R5 neurons
    r5 = NeuronGroup(N_r5, eqs_r5, threshold='v > -10*mvolt', reset=reset_r5, method='euler')

    # Assign each neuron a different current
    r5.I = np.linspace(I_0_r5 - sig_r5, I_0_r5 + sig_r5, N_r5) * mvolt / ms

    # Set initial conditions
    V_0_r5 = np.zeros(N_r5)
    u_0_r5 = np.zeros(N_r5)
    for n in range(N_r5):
        V_0_r5[n] = randrange(-90, -30)
        u_0_r5[n] = randrange(-11, 0)

    r5.v = V_0_r5 * mvolt
    r5.u = u_0_r5 * volt / second
    r5.I_syn_R5_ex = 0 * volt / second
    r5.I_syn_R5_inh = 0 * volt / second
    r5.g_syn_inh = 0 * volt / second
    r5.g_syn_ex = 0 * volt / second
    r5.I_from_hel = 0 * volt / second
    r5.g_from_hel = 0 * volt / second
    r5.I_from_dr = 0 * volt / second
    r5.g_dr = 0 * volt / second

    #####################
    #  Helicon neurons  #
    #####################

    # Create a timed array for visual input
    ta = TimedArray((vis_filt - move_mean) * vis_ampl * mvolt / ms, dt=1 * ms)

    # Helicon Izhikevich parameters for bursting
    a_hel = 0.02 / ms
    b_hel = 0.2 / ms
    c_hel = -65 * mvolt
    d_hel = 6 * mvolt / ms

    # Izhikevich equation and conductance coupling for Helicon neurons
    eqs_hel = '''
        dv/dt = ((0.04/ms/mvolt)*v**2+(5/ms)*v+140*mvolt/ms-u)*0.05+g_hel+g_from_r5_ex+g_from_r5_inh+g_dr+I+I_vis : volt 
        du/dt = (a_hel*(b_hel*v-u))*0.05 : volt/second 
        dI_hel/dt = -I_hel/tau_hel :volt/second
        dg_hel/dt = (I_hel-g_hel)/tau_hel :volt/second
        dI_from_r5_inh/dt = -I_from_r5_inh/tau_syn_inh :volt/second
        dg_from_r5_inh/dt = (I_from_r5_inh-g_from_r5_inh)/tau_syn_inh :volt/second
        dI_from_r5_ex/dt = -I_from_r5_ex/tau_syn_ex :volt/second
        dg_from_r5_ex/dt = (I_from_r5_ex-g_from_r5_ex)/tau_syn_ex :volt/second
        I = I_0_hel + sig_hel*randn() : volt/second (constant over dt)
        I_vis = ta(t) :  volt/second
        dI_from_dr/dt = -I_from_dr/tau_from_dr :volt/second
        dg_dr/dt = (I_from_dr-g_dr)/tau_from_dr :volt/second
        '''

    # Reset of Izhikevich model for Helicon neurons
    reset_hel = '''
        v = c_hel
        u += d_hel
        '''

    # Create population of Helicon neurons
    hel = NeuronGroup(N_hel, eqs_hel, threshold='v > -10*mvolt', reset=reset_hel, method='euler')

    # Set initial conditions for Helicon neurons
    V_0_hel = np.zeros(N_hel)
    u_0_hel = np.zeros(N_hel)
    for n in range(N_hel):
        V_0_hel[n] = randrange(-80, -40)
        u_0_hel[n] = randrange(-11, 0)

    hel.v = V_0_hel * mvolt
    hel.u = u_0_hel * volt / second
    hel.I_hel = 0 * volt / second
    hel.g_hel = 0 * volt / second
    hel.I_from_r5_ex = 0 * volt / second
    hel.I_from_r5_inh = 0 * volt / second
    hel.g_from_r5_ex = 0 * volt / second
    hel.g_from_r5_inh = 0 * volt / second
    hel.I_from_dr = 0 * volt / second
    hel.g_dr = 0 * volt / second

    #####################
    #   Driver neuron   #
    #####################

    # Driver neuron parameters
    a_dr = 0.02 / ms
    b_dr = 0.2 / ms
    c_dr = -65 * mvolt
    d_dr = 6 * mvolt / ms
    I_0 = 0.45 * mvolt / ms

    # Driver neuron equations
    eqs_dr = '''
        dv/dt = ((0.04/ms/mvolt)*v**2+(5/ms)*v+140*mvolt/ms-u)*0.05+I_0 : volt 
        du/dt = (a_dr*(b_dr*v-u))*0.05 : volt/second 
        '''

    # Driver neuron reset
    reset = '''
        v = c_dr
        u += d_dr
        '''

    # Create driver neuron
    dr_neuron = NeuronGroup(1, eqs_dr, threshold='v > -10*mvolt', reset=reset, method='euler')
    
    # Set initial conditions for driver neuron
    dr_neuron.v = -70 * mvolt
    dr_neuron.u = 0 * volt / second

    #####################
    #     Coupling      #
    #####################

    # Synapses R5 - R5
    S_ex_R5 = Synapses(r5, r5, model='I_spike_ex :volt/second', on_pre='I_syn_R5_ex += I_spike_ex')
    S_ex_R5.connect()
    S_ex_R5.I_spike_ex = I_ex_R5 * mvolt / ms

    S_inh_R5 = Synapses(r5, r5, model='I_spike_inh :volt/second', on_pre='I_syn_R5_inh += I_spike_inh')
    S_inh_R5.connect()
    S_inh_R5.I_spike_inh = I_inh_R5 * mvolt / ms

    # Synapses Helicon - Helicon
    S_hel = Synapses(hel, hel, model='I_spike :volt/second', on_pre='I_hel += I_spike')
    S_hel.connect()
    S_hel.I_spike = I_Hel * mvolt / ms

    # Synapses R5 - Helicon
    S_ex_r5_hel = Synapses(r5, hel, model='I_spike_ex :volt/second', on_pre='I_from_r5_ex += I_spike_ex')
    S_ex_r5_hel.connect()
    S_ex_r5_hel.I_spike_ex = I_ex_R5_Hel * mvolt / ms

    S_inh_r5_hel = Synapses(r5, hel, model='I_spike_inh :volt/second', on_pre='I_from_r5_inh += I_spike_inh')
    S_inh_r5_hel.connect()
    S_inh_r5_hel.I_spike_inh = I_inh_R5_Hel * mvolt / ms

    # Synapses Helicon - R5
    S_hel_r5 = Synapses(hel, r5, model='I_spike :volt/second', on_pre='I_from_hel += I_spike')
    S_hel_r5.connect()
    S_hel_r5.I_spike = I_Hel_R5 * mvolt / ms

    # Synapses driver neuron - R5
    S_dr_r5 = Synapses(dr_neuron, r5, model='I_spike :volt/second', on_pre='I_from_dr += I_spike')
    S_dr_r5.connect()
    S_dr_r5.I_spike = I_dr_R5 * mvolt / ms

    # Synapses driver neuron - Helicon
    S_dr_hel = Synapses(dr_neuron, hel, model='I_spike :volt/second', on_pre='I_from_dr += I_spike')
    S_dr_hel.connect()
    S_dr_hel.I_spike = I_dr_Hel * mvolt / ms

    #####################
    #    Monitoring     #
    #####################

    # R5 monitor variables
    M_r5_v = StateMonitor(r5, 'v', record=True)
    M_g_syn_inh = StateMonitor(r5, 'g_syn_inh', record=True)
    M_g_syn_ex = StateMonitor(r5, 'g_syn_ex', record=True)
    M_g_from_hel = StateMonitor(r5, 'g_from_hel', record=True)

    # Helicon monitor variables
    M_hel_v = StateMonitor(hel, 'v', record=True)
    M_g_from_r5_inh = StateMonitor(hel, 'g_from_r5_inh', record=True)
    M_g_from_r5_ex = StateMonitor(hel, 'g_from_r5_ex', record=True)

    # Driver neuron monitor
    M_dr = StateMonitor(dr_neuron, 'v', record=True)

    # Record spike times
    spikemon_r5 = SpikeMonitor(r5, record=True)
    spikemon_hel = SpikeMonitor(hel, record=True)
    
    # Run the simulation
    run(runtime * ms)

    # Plot spike raster plot
    plot_spike_raster_plot(spikemon_r5, spikemon_hel, daytime, r_num)

    # Record the spiketimes of driver neuron
    spikemon_dr_neur = SpikeMonitor(dr_neuron)
    firing_rate = spikemon_dr_neur.num_spikes / second
    print('Firing rate driver neuron:', firing_rate)

    return (spikemon_r5, spikemon_hel, M_r5_v, M_g_syn_inh, M_g_syn_ex, M_hel_v, 
            M_g_from_r5_inh, M_g_from_r5_ex, M_dr, M_g_from_hel, spikemon_dr_neur)


def _gaussian(x, mu, sig, bin_width):
    """
    Create a Gaussian kernel.
    
    Parameters
    ----------
    x : ndarray
        Input values
    mu : float
        Mean of the Gaussian
    sig : float
        Standard deviation of the Gaussian
    bin_width : float
        Width of bins for normalization
        
    Returns
    -------
    ndarray
        Gaussian kernel values
    """
    return ((((np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))) * 1 / (sig * np.sqrt(2 * np.pi))) / bin_width)


def _convolve(spikemon, gaus, M_v, N):
    """
    Convolve spike trains with a Gaussian kernel.
    
    Parameters
    ----------
    spikemon : brian2.SpikeMonitor
        Spike monitor containing spike times
    gaus : ndarray
        Gaussian kernel for convolution
    M_v : brian2.StateMonitor
        State monitor for time reference
    N : int
        Number of neurons
        
    Returns
    -------
    ndarray
        Matrix of convolved spike trains
    """
    # Create a spike matrix (neurons x time)
    spike_matrix = np.zeros((N, len(M_v.t)))
    for n in range(N):
        times_per_neuron = spikemon.t[spikemon.i == n]
        spike_matrix[n, :] = np.in1d(np.array(M_v.t), np.array(times_per_neuron))

    # Convolve each neuron's spike train with the Gaussian
    for i in range(N):
        conv_oscill = np.convolve(spike_matrix[i, :], gaus, mode='full')
        if i == 0:
            conv_matrix = np.array(conv_oscill)
        else:
            conv_matrix = np.vstack((conv_matrix, conv_oscill))

    return conv_matrix


def _correlation_coefficient(t_cut, spikemon, conv_matrix, N):
    """
    Calculate correlation coefficients between neurons.
    
    Parameters
    ----------
    t_cut : int
        Index to cut off initial transient period
    spikemon : brian2.SpikeMonitor
        Spike monitor
    conv_matrix : ndarray
        Matrix of convolved spike trains
    N : int
        Number of neurons
        
    Returns
    -------
    tuple
        (mean_corr_coef, corr_coef_matrix)
    """
    corr_coef_matrix = np.zeros((N, N))

    for i in range(1, N):
        for j in range(i):
            # To avoid NaN when one oscillator never fires, set pearson coeff to zero
            if np.sum(conv_matrix[j, :]) == 0 or np.sum(conv_matrix[i, :]) == 0:
                corr_coef_matrix[i, j] = 0
            else:
                # Only use the last time steps for correlation (cut off initial transient)
                corr = pearsonr(conv_matrix[i, t_cut:], conv_matrix[j, t_cut:])
                corr_coef_matrix[i, j] = corr[0]

    # Calculate mean correlation coefficient (excluding zeros)
    mean_corr_coef = np.mean(corr_coef_matrix[corr_coef_matrix != 0])

    return mean_corr_coef, corr_coef_matrix


def _crosscorr(datax, datay, lag=0, wrap=False):
    """
    Calculate lag-N cross correlation between two time series.
    
    Parameters
    ----------
    datax : pandas.Series
        First time series
    datay : pandas.Series
        Second time series
    lag : int, default 0
        Lag in time steps
    wrap : bool, default False
        Whether to wrap around at edges
        
    Returns
    -------
    float
        Cross correlation coefficient
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def run_and_plot(N_r5, tau_syn_ex, tau_syn_inh, I_ex_R5, I_inh_R5, N_hel, tau_hel, sig_hel, I_0_hel, I_Hel, I_ex_R5_Hel,
                 I_inh_R5_Hel, I_Hel_R5, tau_from_dr, I_dr_R5, I_dr_Hel, runtime, dt, daytime, t_cut,
                 r_num, save_results, label_diver_neuron, vis_filt, vis_ampl, move_mean):
    """
    Run the simulation and perform analysis including correlation and PSD.
    
    This is the main function to call for running the simulation. It executes
    the simulation, calculates correlations between populations, and performs
    spectral analysis.
    
    Parameters
    ----------
    N_r5 : int
        Number of R5 neurons
    tau_syn_ex : brian2.Quantity
        Time constant for excitatory synapses
    tau_syn_inh : brian2.Quantity
        Time constant for inhibitory synapses
    I_ex_R5 : ndarray
        Excitatory connection strengths within R5 population
    I_inh_R5 : ndarray
        Inhibitory connection strengths within R5 population
    N_hel : int
        Number of Helicon neurons
    tau_hel : brian2.Quantity
        Time constant for Helicon neurons
    sig_hel : brian2.Quantity
        Noise amplitude for Helicon neurons
    I_0_hel : brian2.Quantity
        Base current for Helicon neurons
    I_Hel : ndarray
        Connection strengths within Helicon population
    I_ex_R5_Hel : ndarray
        Excitatory connection strengths from R5 to Helicon
    I_inh_R5_Hel : ndarray
        Inhibitory connection strengths from R5 to Helicon
    I_Hel_R5 : ndarray
        Connection strengths from Helicon to R5
    tau_from_dr : brian2.Quantity
        Time constant for driver neuron connections
    I_dr_R5 : float
        Connection strength from driver to R5
    I_dr_Hel : float
        Connection strength from driver to Helicon
    runtime : int
        Simulation duration in ms
    dt : int
        Simulation time step in ms
    daytime : str
        Simulation mode ('morning' or 'night')
    t_cut : int
        Index to cut off initial transient period
    r_num : int
        Run number identifier
    save_results : str
        Whether to save results ('true' or 'false')
    label_diver_neuron : str
        Label for the driver neuron mode
    vis_filt : ndarray
        Visual input filter
    vis_ampl : float
        Visual input amplitude
    move_mean : float
        Mean value for visual input
        
    Returns
    -------
    tuple
        (offset, Hel_R5_corr_coef, mean_corr_coef_r5, mean_corr_coef_hel, spikemon_dr_neur)
    """
    # Run coupled R5-Hel system
    spikemon_r5, spikemon_hel, M_r5_v, M_g_syn_inh, M_g_syn_ex, M_hel_v, M_g_from_r5_inh, M_g_from_r5_ex, M_dr, \
    M_g_from_hel, spikemon_dr_neur = _neurons(
        N_r5, tau_syn_ex, tau_syn_inh, I_ex_R5, I_inh_R5, N_hel, tau_hel, sig_hel,
        I_0_hel, I_Hel, I_ex_R5_Hel, I_inh_R5_Hel, I_Hel_R5, tau_from_dr, I_dr_R5,
        I_dr_Hel, runtime, dt, daytime, vis_filt, vis_ampl, move_mean, r_num)

    # Create Gaussian filter for R5 population
    bin_width = 1 / 1000
    sig_r5_gauss = 100
    x_values_r5 = np.arange(-1000, 1000, dt)
    gauss_r5 = _gaussian(x_values_r5, 0, sig_r5_gauss, bin_width)

    # Create Gaussian filter for Helicon population
    sig_hel_gauss = 100
    x_values_hel = np.arange(-1000, 1000, dt)
    gauss_hel = _gaussian(x_values_hel, 0, sig_hel_gauss, bin_width)

    # Convolve spike trains with Gaussian filters
    conv_matrix_r5 = _convolve(spikemon_r5, gauss_r5, M_r5_v, N_r5)
    conv_matrix_hel = _convolve(spikemon_hel, gauss_hel, M_hel_v, N_hel)

    # Calculate correlation coefficient between R5 and Helicon population
    Hel_R5_corr_coef = pearsonr(np.sum(conv_matrix_r5, axis=0), np.sum(conv_matrix_hel, axis=0))
    print(f'Correlation coefficient between Helicon and R5: {Hel_R5_corr_coef[0]}')

    # Calculate correlation coefficients within R5 and Helicon populations
    mean_corr_coef_r5, corr_coef_matrix_r5 = _correlation_coefficient(t_cut, spikemon_r5, conv_matrix_r5, N_r5)
    mean_corr_coef_hel, corr_coef_matrix_hel = _correlation_coefficient(t_cut, spikemon_hel, conv_matrix_hel, N_hel)
    print('Mean correlation coefficient R5:', mean_corr_coef_r5)
    print('Mean correlation coefficient Helicon:', mean_corr_coef_hel)

    # Time lagged correlation coefficient (only after system has stabilized)
    conv_matrix_r5_cropped = conv_matrix_r5[:, t_cut:]
    conv_matrix_hel_cropped = conv_matrix_hel[:, t_cut:]

    # Calculate time-lagged correlations to find lead/lag relationships
    R5_sig = pd.DataFrame(np.sum(conv_matrix_r5_cropped, axis=0))
    Hel_sig = pd.DataFrame(np.sum(conv_matrix_hel_cropped, axis=0))
    ms = 100  # Every 100 millisecond
    fps_2 = 5  # 100 times so in total plus/minus 10 seconds
    rs = [_crosscorr(Hel_sig[0], R5_sig[0], lag) for lag in range(-int(ms * fps_2), int(ms * fps_2 + 1))]
    offset = np.floor(len(rs) / 2) - np.argmax(rs)
    print('Offset center to peak synchrony:', offset)

    # Plot cross-correlation results
    f, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(f'Offset = {-offset} ms\nHel. leads <> R5 leads', fontsize=14)
    ax.plot(rs, color='k', linewidth=0.7)
    ax.axvline(np.ceil(len(rs) / 2), color='b', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set_xlabel('offset [s]', fontsize=14)
    ax.set_ylabel('correlation coefficient', fontsize=14)
    ax.set_xticks(np.linspace(0, len(rs), 5))
    ax.set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5])
    plt.legend(fontsize='medium')
    plt.tight_layout()
    # Uncomment to save the figure
    # plt.savefig(path_utils.path_to_figures + f'CrossCorr_{daytime}_{label_diver_neuron}_{r_num}.svg', format='svg')

    # Compute power spectral density
    # Skip first 1500 time steps to avoid initial transients
    no_overlap = 256 * 50 / 2
    P_SD_r5 = ml.psd(np.sum(conv_matrix_r5[:, 1500:-1], axis=0), Fs=1200, NFFT=256 * 50)
    # Crop out the 0Hz values (focus on frequencies >= 0.1 Hz)
    freq_r5 = P_SD_r5[1][P_SD_r5[1] >= 0.1]
    PSD_r5 = P_SD_r5[0][(np.shape(P_SD_r5[0])[0] - np.shape(freq_r5)[0]):]

    P_SD_hel = ml.psd(np.sum(conv_matrix_hel[:, 1500:-1], axis=0), Fs=1200, NFFT=256 * 50)
    # Crop out the 0Hz values
    freq_hel = P_SD_hel[1][P_SD_hel[1] >= 0.1]
    PSD_hel = P_SD_hel[0][(np.shape(P_SD_hel[0])[0] - np.shape(freq_hel)[0]):]

    # Save simulation results if requested
    if save_results == 'true':
        # Save synaptic currents
        np.save(path_utils.path_to_files + f'I_r5_hel_ecx_{daytime}_{label_diver_neuron}_{r_num}.npy', M_g_from_r5_ex.g_from_r5_ex)
        np.save(path_utils.path_to_files + f'I_r5_hel_inh_{daytime}_{label_diver_neuron}_{r_num}.npy', M_g_from_r5_inh.g_from_r5_inh)
        np.save(path_utils.path_to_files + f'I_hel_r5_{daytime}_{label_diver_neuron}_{r_num}.npy', M_g_from_hel.g_from_hel)
        
        # Save R5 currents
        np.save(path_utils.path_to_files + f'I_r5_exc_{daytime}_{label_diver_neuron}_{r_num}.npy', M_g_syn_ex.g_syn_ex)
        np.save(path_utils.path_to_files + f'I_r5_inh_{daytime}_{label_diver_neuron}_{r_num}.npy', M_g_syn_inh.g_syn_inh)
        
        # Save time information
        np.save(path_utils.path_to_files + f'time_{daytime}_{label_diver_neuron}.npy', M_g_from_r5_inh.t)
        np.save(path_utils.path_to_files + f'runtime_{daytime}_{label_diver_neuron}.npy', runtime)
        
        # Save spike information
        np.save(path_utils.path_to_files + f'spikemon_r5_t_{daytime}_{label_diver_neuron}.npy', spikemon_r5.t)
        np.save(path_utils.path_to_files + f'spikemon_r5_i_{daytime}_{label_diver_neuron}_{r_num}.npy', spikemon_r5.i)
        np.save(path_utils.path_to_files + f'spikemon_hel_t_{daytime}_{label_diver_neuron}.npy', spikemon_hel.t)
        np.save(path_utils.path_to_files + f'spikemon_hel_i_{daytime}_{label_diver_neuron}_{r_num}.npy', spikemon_hel.i)
        
        # Save compound signals and PSDs
        np.save(path_utils.path_to_files + f'compound_r5_{daytime}_{label_diver_neuron}_{r_num}.npy', conv_matrix_r5)
        np.save(path_utils.path_to_files + f'compound_hel_{daytime}_{label_diver_neuron}_{r_num}.npy', conv_matrix_hel)
        np.save(path_utils.path_to_files + f'PSD_r5_{daytime}_{label_diver_neuron}_{r_num}.npy', PSD_r5)
        np.save(path_utils.path_to_files + f'PSD_hel_{daytime}_{label_diver_neuron}_{r_num}.npy', PSD_hel)
        np.save(path_utils.path_to_files + f'frequencies_{daytime}_{label_diver_neuron}.npy', freq_hel)

    return offset, Hel_R5_corr_coef[0], mean_corr_coef_r5, mean_corr_coef_hel, spikemon_dr_neur