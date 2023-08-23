import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], 'lalsuite-extra/data/lalsimulation')

import numpy as np
import argparse
import h5py
import lal
import scipy.linalg as sl
import scipy.signal as sig
import scipy.stats as ss
from collections import OrderedDict
import pandas as pd
from contextlib import closing
import sys
sys.path.append('../')
import utils
from utils import reconstructwf as rwf 
import lalsimulation as lalsim
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt

from helper_functions import whitenData_TD

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.lines import Line2D
from textwrap import wrap

style.use('plotting.mplstyle')


#  Path where all data is stored: 
data_dir = '/Users/smiller/Documents/gw190521-timedomain-release/data_simonas_laptop/' 
                 
    
# ----------------------------------------------------------------------------
# Load strain data

ifos = ['L1']
data_path = data_dir+'GW190521_data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5'
raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos,path=data_path)

psd_path = data_dir+'GW190521_data/glitch_median_PSD_forLI_{}.dat'
pe_path = data_dir+'GW190521_data/GW190521_posterior_samples.h5'
pe_out = utils.get_pe(raw_time_dict, path=pe_path, psd_path=psd_path)
tpeak_geocent, tpeak_dict, _, pe_samples, log_prob, pe_psds, maxP_skypos = pe_out

t0_geocent = 1242442967.405764
tstart = 1242442966.9077148
tend = 1242442967.607715

ra_0 = 6.07546535866838
dec_0 = -0.8000357325337637
psi_0 = 2.443070879119043

t0_dict, ap_0_dict = utils.get_tgps_and_ap_dicts(t0_geocent, ifos, ra_0, dec_0, psi_0)

cond_psds = {}
for ifo, freq_psd in pe_psds.items():
    
    # psd for whitening
    freq, psd = freq_psd.copy().T
    m = freq >= 11
    psd[~m] = 100*max(psd[m]) # set values below 11 Hz to be equal to 100*max(psd)    
    
    cond_psds[ifo] = (freq, psd)
    
    
# ----------------------------------------------------------------------------
# Condition data

ds_factor = 8
f_low = 11

# i0 = index corresponding to peak time
time_dict, data_dict, i0_dict = utils.condition(raw_time_dict, raw_data_dict,
                                                t0_dict, ds_factor, f_low) 

# Decide how much data to analyze
dt = time_dict['L1'][1] - time_dict['L1'][0]
                                      
TPre = (t0_geocent - tstart) + 1.3
TPost = tend - t0_geocent
    
Npre = int(round(TPre / dt))
Npost = int(round(TPost / dt)) + 1  # must add one so that the target time is actually included, even if Tpost = 0,
                                         # otherwise WF placement gets messed up   
Nanalyze = Npre + Npost

rho_dict = OrderedDict() # stores acf 
L_dict = OrderedDict()   # stores L such that cov matrix C = L L^T
for ifo, data in data_dict.items():
    freq, psd = cond_psds[ifo]
    dt = 0.5 / round(freq.max())
    rho = 0.5*np.fft.irfft(psd) / dt # dt comes from numpy fft conventions
    
    # compute covariance matrix  C and its Cholesky decomposition L (~sqrt of C)
    C = sl.toeplitz(rho[:Nanalyze])
    L_dict[ifo] = np.linalg.cholesky(C)
    
    rho_dict[ifo] = rho[:Nanalyze]

# Crop analysis data to specified duration.
for ifo, I0 in i0_dict.items():
    # I0 = sample closest to desired time
    time_dict[ifo] = time_dict[ifo][I0-Npre:I0+Npost]
    data_dict[ifo] = data_dict[ifo][I0-Npre:I0+Npost] 
    
    
# ----------------------------------------------------------------------------
# Load posterior samples   

pathname = '{0}_gw190521_{1}_NRSur7dq4_dec8_flow11_fref11_{2}_TstartTend'
path_template = data_dir + pathname + '.dat'
    
paths = {}

date = '063023'
paths['full'] = path_template.format(date, 'full','0M')

td_samples = {}
for k, p in paths.items():
    try:
        td_samples[k] = np.genfromtxt(p, names=True, dtype=float)
    except:
        print(f'cannot find {p}')
        
        
# ----------------------------------------------------------------------------
# Generate reconstructions from posteriors

# Reference freq = 11 Hz to correspond to LVK analysis
fref = 11

# Look at LIGO Livingston data only
ifo = 'L1'
    
print('Generating reconstructions ... ')

                
# Fetch samples
samples = td_samples['full']
nsamples = len(samples)

# Downsample 
ntraces = 200
indices = np.random.choice(nsamples, ntraces, replace=False)

whitened = []
unwhitened = []

for j in indices:

    # Unpack parameters
    m1, m2 = utils.m1m2_from_mtotq(samples['mtotal'][j], samples['q'][j])
    m1_SI = m1*lal.MSUN_SI
    m2_SI = m2*lal.MSUN_SI
    chi1 = samples['chi1'][j]
    chi2 = samples['chi2'][j]
    tilt1 = samples['tilt1'][j]
    tilt2 = samples['tilt2'][j]
    phi12 = samples['phi12'][j]
    theta_jn = samples['theta_jn'][j]
    phi_jl = samples['phi_jl'][j]
    dist_mpc = samples['dist'][j]
    phi_ref = samples['phase'][j]

    # Translate spin convention
    iota, s1x, s1y, s1z, s2x, s2y, s2z = lalsim.SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2, m1_SI, m2_SI, fref, phi_ref
    )

    tt_dict = tpeak_dict.copy()
    ap_dict = ap_0_dict.copy()

    # Get strain
    hp, hc = rwf.generate_lal_hphc('NRSur7dq4', m1, m2, 
                                   [s1x, s1y, s1z], [s2x, s2y, s2z],
                                   dist_mpc=dist_mpc, dt=dt,
                                   f_low=fref, f_ref=fref,
                                   inclination=iota,
                                   phi_ref=phi_ref, ell_max=None)

    # Time align
    h = rwf.generate_lal_waveform(hplus=hp, hcross=hc,
                                  times=time_dict[ifo], 
                                  triggertime=tt_dict[ifo])

    # Project onto detectors
    Fp, Fc = ap_dict[ifo]
    h_ifo = Fp*h.real - Fc*h.imag

    # Whiten
    w_h_ifo = whitenData_TD(h_ifo, L_dict[ifo])


    # Add to arrays
    unwhitened.append(h_ifo)
    whitened.append(w_h_ifo)

        
# 0M <-> seconds
t0_0M_dict = {}
t0_0M_geo = 1242442967.405764
dt_10M = 0.0127 # 10 M = 12.7 ms 
dt_1M = dt_10M/10.

ra = 6.07546535866838
dec = -0.8000357325337637

# define t_0M in each detector in seconds
for ifo in ['H1', 'L1', 'V1']: 
    
    t_delay = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location, ra, dec, t0_0M_geo)
    t0_0M_dict[ifo] = t0_0M_geo + t_delay
    
reconstruction_times_M = (time_dict['L1'] - t0_0M_dict['L1'])/dt_1M

alph = 0.01

plt.figure()

plt.plot(reconstruction_times_M, np.asarray(unwhitened).T*5e21, color='k', alpha=alph)
plt.plot(reconstruction_times_M, np.asarray(whitened).T, color='r', alpha=alph)

plt.ylim(-4, 4)
plt.xlim(-70, 60)

# Legend
handles = [
    Line2D([], [], color='k', label=r'colored $\times$ 5e21'),
    Line2D([], [], color='r', label='whitened (T = 2 seconds)'),
]
plt.legend(handles=handles)

plt.xlabel(r'$\Delta t [M_\mathrm{f}]$', fontsize=15)
plt.ylabel(r'$h$', fontsize=15)

plt.show()