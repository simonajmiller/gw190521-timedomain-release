import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], 'lalsuite-extra/data/lalsimulation')

import numpy as np
import argparse
import lal
from collections import OrderedDict
import sys
sys.path.append('../')
import utils
from utils import reconstructwf as rwf 
import lalsimulation as lalsim
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt

from helper_functions import *

#  Path where all data is stored: 
data_dir = '../data/' 
                 
    
# ----------------------------------------------------------------------------
# Load strain data

ifos = ['H1', 'L1', 'V1']
data_path = data_dir+'GW190521_data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5'
raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos,path=data_path)

psd_path = data_dir+'GW190521_data/glitch_median_PSD_{}.dat'
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

# Crop analysis data to specified duration
for ifo, I0 in i0_dict.items():
    # I0 = sample closest to desired time
    time_dict[ifo] = time_dict[ifo][I0-Npre:I0+Npost]
    data_dict[ifo] = data_dict[ifo][I0-Npre:I0+Npost]
    
    
# Calculate ACF
rho_dict = OrderedDict() 
for ifo, data in data_dict.items():
    freq, psd = cond_psds[ifo]
    dt = 0.5 / round(freq.max())
    rho = 0.5*np.fft.irfft(psd) / dt # dt comes from numpy fft conventions
    rho_dict[ifo] = rho
    
    
# ----------------------------------------------------------------------------
# Load posterior samples   

pathname = '{0}_gw190521_{1}_NRSur7dq4_dec8_flow11_fref11_{2}_TstartTend'
path_template = data_dir + pathname + '.dat'

date = '063023'
runs = ['insp', 'rd']
tcutoffs = ['m50M', 'm40M', 'm37.5M', 'm35M', 'm32.5M', 'm30M', 'm27.5M', 'm25M', 'm22.5M', 'm20M', 
                'm17.5M', 'm15M', 'm12.5M', 'm10M', 'm7.5M', 'm5M', 'm2.5M', '0M', '2.5M', '5M', '7.5M', 
                '10M', '12.5M', '15M', '17.5M', '20M', '30M', '40M', '50M']
paths = {}
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
# Generate SNRs from posteriors

# Reference freq = 11 Hz to correspond to LVC analysis
fref = 11

# Where to save: 
savepath = data_dir+'snrs.npy'
print(f'\nWill save SNRs at {savepath}')

# Runs 
runs = [f'{y} {x}' for x in tcutoffs for y in ['insp', 'rd']]
runs.append('full')
runs.append('prior')

# Cycle through detectors
snr_dict = {}
for ifo in ['H1', 'L1', 'V1']:
    print(ifo)
    
    # time stamps in units of M 
    dt_10M = 0.0127 # 10 M = 12.7 ms 
    dt_1M = dt_10M/10.
    times_M = (time_dict[ifo] - t0_dict[ifo])/dt_1M

    snr_dict_ifo = {}

    # Cycle through runs 
    for run in runs: 
        print(run)

        # Start and end time of analysis depend on the run
        if run=='full' or run=='prior': 
            tstart = times_M[0]
            tend = times_M[-1]
        else:

            # Format run into a number
            cut_num = float(((run.split('insp ')[-1]).split('rd ')[-1]).split('m')[-1][:-1])
            cut = -1*cut_num if 'm' in run else cut_num

            # pre-t_cut analyses
            if run[0] == 'i':
                tstart = times_M[0]
                tend = cut
            #post-t_cut analyses
            else: 
                tstart = cut
                tend =  times_M[-1]

        # Get times mask for truncation                
        mask = (times_M > tstart) & (times_M < tend)

        # Fetch samples
        try:
            samples = td_samples[run]
        except: 
            samples = td_samples['prior']
        
        # Generate reconstructions
        h_array = []
        for samp in samples:

            # Unpack parameters
            m1, m2 = m1m2_from_mtotq(samp['mtotal'], samp['q'])
            chi1 = samp['chi1']
            chi2 = samp['chi2']
            tilt1 = samp['tilt1']
            tilt2 = samp['tilt2']
            phi12 = samp['phi12']
            theta_jn = samp['theta_jn']
            phi_jl = samp['phi_jl']
            dist_mpc = samp['dist']
            phi_ref = samp['phase']

            # Translate spin convention
            iota, s1x, s1y, s1z, s2x, s2y, s2z = transform_spins(
                theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2, m1, m2, fref, phi_ref
            )

            # Get strain
            hp, hc = rwf.generate_lal_hphc('NRSur7dq4', m1, m2, 
                                           [s1x, s1y, s1z], [s2x, s2y, s2z],
                                           dist_mpc=dist_mpc, dt=dt,
                                           f_low=fref, f_ref=fref,
                                           inclination=iota,
                                           phi_ref=phi_ref)

            # Time align
            h = rwf.generate_lal_waveform(hplus=hp, hcross=hc,
                                          times=time_dict[ifo], 
                                          triggertime=tpeak_dict[ifo])

            # Project onto detectors
            Fp, Fc = ap_0_dict[ifo]
            h_ifo = Fp*h.real - Fc*h.imag
            
            # Apply time mask and add to array 
            h_array.append(h_ifo[mask])
        

        # Get ACF
        rho = rho_dict[ifo]
        Nanalyze = len(h.T)
        print(run, Nanalyze)

        # Calculate SNR for each reconstruction
        d = data_dict[ifo][mask]
        snrs = np.zeros(len(h))
        for i, s in enumerate(h):
            snrs[i] = calc_mf_SNR(d, s, rho[:Nanalyze]) 

        snr_dict_ifo[run] = snrs

    snr_dict[ifo] = snr_dict_ifo

    print('')

# Save snrs dict so we can use it in Fig. 1
np.save(savepath, snr_dict, allow_pickle=True)


# Print out info 
print(('Run\t L1 SNR').expandtabs(12) + '\tNetwork SNR'.expandtabs(5))
print('-----------------------------------')

for run, snrs in snr_dict['L1'].items(): 
    
    if run in runs_to_plot:
    
        # L1
        med_snr_L1 = np.quantile(np.abs(snrs), 0.5)

        # Network 
        network_snrs = np.asarray([calc_network_mf_SNR([L, H, V]) for L, H, V in zip(snr_dict['L1'][run], 
                                                                                     snr_dict['H1'][run], 
                                                                                     snr_dict['V1'][run])])
        med_network_snr = np.quantile(network_snrs[~np.isnan(network_snrs)], 0.5)

        # print
        run_lbl = run.replace('m', '-').replace('insp', 't <').replace('rd', 't >')

        print(f"{run_lbl:<12} {round(med_snr_L1, 2):<10} {round(med_network_snr, 2)}")
        
        
# And make figure 
plt.figure(figsize=(12, 5))

for run in snr_dict['L1'].keys(): 
    
    if run in runs_to_plot:
    
        snrs = [calc_network_mf_SNR([L, H, V]) for L, H, V in zip(snr_dict['L1'][run], 
                                                                  snr_dict['H1'][run], 
                                                                  snr_dict['V1'][run])]

        lbl = run.replace('m', '-').replace('insp', 't $<$').replace('rd', 't $>$')
        plt.hist(snrs, histtype='step', label=lbl, density=True, bins=np.linspace(0,15,100))

plt.legend(loc='upper left',ncols=2)
plt.xlabel(r'$\mathrm{SNR}_\mathrm{mf}$', fontsize=15)
plt.ylabel(r'$p(\mathrm{SNR}_\mathrm{mf})$', fontsize=15)
plt.xlim(0,15)
plt.show()