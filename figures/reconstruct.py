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
from tqdm import tqdm
import lalsimulation as lalsim


# parse args 
p = argparse.ArgumentParser()
p.add_argument('--vary-time', action='store_true')
p.add_argument('--vary-skypos', action='store_true')
p.add_argument('--reload', action='store_true')
args = p.parse_args()

reload = args.reload
varyT = args.vary_time 
varySkyPos = args.vary_skypos

#  Path where all data is stored: 
data_dir = '/Users/smiller/Documents/gw190521-timedomain-release/data_simonas_laptop/' 
                 

# ----------------------------------------------------------------------------
# Load strain data

ifos = ['L1']
data_path = data_dir+'GW190521_data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5' ## TODO-update with final file paths 
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
                                      
TPre = t0_geocent - tstart
TPost = tend - t0_geocent
    
Npre = int(round(TPre / dt))
Npost = int(round(TPost / dt)) + 1  # must add one so that the target time is actually included, even if Tpost = 0,
                                         # otherwise WF placement gets messed up   
Nanalyze = Npre + Npost
Tanalyze = Nanalyze*dt
print('Will analyze {:.3f} s of data at {:.1f} Hz'.format(Tanalyze, 1/dt))

L_dict = OrderedDict()   # stores L such that cov matrix C = L^T L
for ifo, data in data_dict.items():
    freq, psd = cond_psds[ifo]
    dt = 0.5 / round(freq.max())
    rho = 0.5*np.fft.irfft(psd) / dt # dt comes from numpy fft conventions
    
    # compute covariance matrix  C and its Cholesky decomposition L (~sqrt of C)
    C = sl.toeplitz(rho[:Nanalyze])
    L_dict[ifo] = np.linalg.cholesky(C)
        
# Crop analysis data to specified duration.
for ifo, I0 in i0_dict.items():
    # I0 = sample closest to desired time
    time_dict[ifo] = time_dict[ifo][I0-Npre:I0+Npost]
    data_dict[ifo] = data_dict[ifo][I0-Npre:I0+Npost]
    
    
# ----------------------------------------------------------------------------
# Load posterior samples   

pathname = '{0}_gw190521_{1}_NRSur7dq4_dec8_flow11_fref11_{2}_TstartTend'
path_template = data_dir + pathname + '.dat'

if varyT and varySkyPos: 
    path_template = path_template.replace('.dat','_VaryTAndSkyPos.dat')
    date = '062323'
elif varyT: 
    path_template = path_template.replace('.dat','_VaryT_FixedSkyPos.dat')
    date = '062823'
# elif varySkyPos: 
#     path_template = path_template.replace('.dat','_FixedT_VarySkyPos.dat')
else: 
    date = '063023'
    
paths = {}

runs = ['insp', 'rd']
tcutoffs = ['m50M', 'm40M', 'm37.5M', 'm35M', 'm32.5M', 'm30M', 'm27.5M', 'm25M', 'm22.5M', 'm20M', 
                'm17.5M', 'm15M', 'm12.5M', 'm10M', 'm7.5M', 'm5M', 'm2.5M', '0M', '2.5M', '5M', '7.5M', 
                '10M', '12.5M', '15M', '17.5M', '20M', '30M', '40M', '50M']
for run in runs: 
    for tcut in tcutoffs: 
        key = f'{run} {tcut}'
        paths[key] = path_template.format(date,run,tcut)
        
paths['full'] = path_template.format(date, 'full','0M')
paths['prior'] = data_dir+'prior_vary_time_and_skypos.dat'

print('\nLoading PE samples ... ')

td_samples = {}
for k, p in paths.items():
    try:
        td_samples[k] = np.genfromtxt(p, names=True, dtype=float)
    except:
        print(f'cannot find {p}')
        
        
# ----------------------------------------------------------------------------
# Generate reconstructions from posteriors

fref = 11
ifo = 'L1'

# where to save: 
savename = "waveform_reconstructions_L1"
savepath = path_template.replace(pathname, savename).replace('.dat', '.npy')

# load in existing if we want 
if os.path.exists(savepath) and reload:
    reconstructions = np.load(savepath,allow_pickle=True).item()
    keys_to_calculate = [key for key in td_samples.keys() if key not in reconstructions.keys()]
else:
    reconstructions = {}
    reconstructions['time samples'] = time_dict[ifo]
    keys_to_calculate = td_samples.keys()

print('\nGenerating reconstructions ... ')

for k in keys_to_calculate:
            
    print(k)
    samples = td_samples[k]

    indices = np.random.choice(range(len(samples)), 1000)

    whitened = []
    unwhitened = []
    bandpassed = []

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
        
        # Get time and skyposition 
        if varySkyPos or varyT:
            
            ra = samples['ra'][j] if varySkyPos else ra_0
            dec = samples['dec'][j] if varySkyPos else dec_0
            psi = samples['psi'][j] if varySkyPos else psi_0
            tt_geocent = samples['tgps_geocent'][j] if varyT else t0_geocent

            tt_dict, ap_dict = utils.get_tgps_and_ap_dicts(tt_geocent, ifos, ra, dec, psi, verbose=False)
        
        else: 
            tt_dict = tpeak_dict.copy()
            ap_dict = ap_0_dict.copy()

        # Get strain
        hp, hc = rwf.generate_lal_hphc('NRSur7dq4', m1, m2, 
                                       [s1x, s1y, s1z], [s2x, s2y, s2z],
                                       dist_mpc=dist_mpc, dt=dt,
                                       f_low=11, f_ref=fref,
                                       inclination=iota,
                                       phi_ref=phi_ref, ell_max=None)
       
        # Time align
        h = rwf.generate_lal_waveform(hplus=hp, hcross=hc,
                                      times=time_dict[ifo], 
                                      triggertime=tt_dict[ifo])

        # Whiten 
        w_h = np.linalg.solve(L_dict[ifo], h)
        
        # Just bandpass 
        fmin_bp, fmax_bp = 20, 500
        h_bp = rwf.bandpass(h, time_dict[ifo], fmin_bp, fmax_bp)

        # Project onto detectors
        Fp, Fc = ap_dict[ifo]
        h_ifo = Fp*h.real - Fc*h.imag
        w_h_ifo = Fp*w_h.real - Fc*w_h.imag
        h_bp_ifo = Fp*h_bp.real - Fc*h_bp.imag

        unwhitened.append(h_ifo)
        whitened.append(w_h_ifo)
        bandpassed.append(h_bp_ifo)

    reconstructions[k] = {'wh':whitened, 'h':unwhitened, 'bp':bandpassed, 'params':samples[indices]}
    
    # Save results as we go
    np.save(savepath, reconstructions, allow_pickle=True)