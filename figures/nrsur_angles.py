#############################################################################
##
##      Filename: nrsur_Lhat.py
##
##      Author: Vijay Varma, Simona Miller
##
##      Created: 21-05-2023, edited 12-06-23
##
##      Description: plot inclination angle as a function of time
##
#############################################################################

import sys
sys.path.append('../')
import utils

import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], 'lalsuite-extra/data/lalsimulation')

import numpy as np
import matplotlib.pyplot as P
import lal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from lalsimulation.nrfits.NRSur7dq4Remnant import NRSur7dq4Remnant
from gwtools import rotations   # pip install gwtools

def get_unit_vector(v): 
    return v/np.sum(v * v, axis=1)

# Define surrogate
sur = NRSur7dq4Remnant()
sur_dyn = sur._get_surrogate_dynamics

f_ref = 25      # You may want to use -1, but with Vijay's lalsuite branch

#  Path where all data is stored
data_dir = '/Users/smiller/Documents/gw190521-timedomain-release/data_simonas_laptop/' 

# Load waveform reconstructions and their parameters 
reconstruction_dict = np.load(data_dir+"waveform_reconstructions_L1.npy",allow_pickle=True).item()
reconstruction_dict.pop('time samples')

# where to save: 
savepath = data_dir+'angles_vs_time_dict.npy'

# load in existing if we want 
reload = True
if os.path.exists(savepath) and reload:
    # Dict in which to store angles vs. time data
    angles_vs_time_dict = np.load(savepath,allow_pickle=True).item()
else:
    angles_vs_time_dict = {}

# Cycle through the runs
#keys_to_calculate = [key for key in reconstruction_dict.keys() if key not in angles_vs_time_dict.keys()]
keys_to_calculate = ['full']

for key in keys_to_calculate:
       
    # To track progress
    print(key)
            
    # Fetch posterior_samples 
    samples = reconstruction_dict[key]['params']
    
    # To store incl versus time
    incl_vs_t_list = []
    dyn_times_list = []
    
    # Cycle through samples 
    for samp in samples: 
    
        M = samp['mtotal'] # Detector frame total mass
        q = samp['q']
        m1, m2 = utils.m1m2_from_mtotq(M, q)
        m1_SI = m1*lal.MSUN_SI # convert to kg
        m2_SI = m2*lal.MSUN_SI
        chi1 = samp['chi1']
        chi2 = samp['chi2']
        tilt1 = samp['tilt1']
        tilt2 = samp['tilt2']
        phi12 = samp['phi12']
        theta_jn = samp['theta_jn']
        phi_jl = samp['phi_jl']
        dist_mpc = samp['dist']
        phi_ref = samp['phase']

        # Transform spin convention 
        incl, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2, m1_SI, m2_SI, f_ref, phi_ref
        )
        
        # Orbital angular frequency converted to dimensionless units.
        # (Freq. scales as 1/M; this takes out the total mass dependence to get the dim-less omega_ref.
        # lal.MTSUN_SI is a solar mass in seconds.)
        omega_ref = f_ref * np.pi* M * lal.MTSUN_SI if f_ref != -1 else -1

        # Get surrogate dynamics
        q_sur = 1.0/q
        chiA0 = [s1x, s1y, s1z] # A subscript in NR <-> 1 subscript in LIGO
        chiB0 = [s2x, s2y, s2z]
        dyn_times, quat_sur, orbphase_sur, chiA_copr_sur, chiB_copr_sur \
            = sur_dyn(q_sur, chiA0, chiB0, [1,0,0,0], 0, omega_ref, False)

        # Direction of the orbital angular momentum, defined with respect to the
        # source frame at f_ref
        Lhat = rotations.lHat_from_quat(quat_sur).T
        
        # Get total angular momentum vector
        chiA = ???                  # dimensionless spins over time
        chiB = ??? 
        S_1 = chiA * m1**2          # magnitude of spins over time
        S_2 = chiB * m2**2
        Lmag = ???                  # magnitude of orbital angular momentum
        L = Lmag * Lhat
        J = L + S_1 + S_2           # add it all up 
        Jhat = get_unit_vector(J)

        # This is the phi that goes into the Ylms, and phi_ref is defined
        # like this by silly LIGO.
        phi = np.pi/2 - phi_ref
        
        # N_hat = the direction of the line of sight from source to obsever,
        # also defined with respect to the source frame at f_ref
        Nhat = np.array([np.sin(incl)*np.cos(phi), \
                          np.sin(incl)*np.sin(phi), \
                          np.cos(incl)]).T

        # Take inner product of the two unit vector and arccos it to get
        # inclination as a function of time.
        incl_vs_t = np.arccos(np.sum(Lhat * Nhat, axis=1))
        
        # Get angles between J and L as function of time
        thetajl_vs_t = np.arccos(np.sum(Lhat * Jhat, axis=1))
        phijl_vs_t = ???
        
        # Get angles between J and spin vectors as a function of time
        S1_hat = get_unit_vector(S_1)
        theta_J_chi1_vs_t = np.arccos(np.sum(S1_hat * Jhat, axis=1))
        phi_J_chi1_vs_t = ???
        
        S2_hat = get_unit_vector(S_2)
        theta_J_chi2_vs_t = np.arccos(np.sum(S2_hat * Jhat, axis=1))
        phi_J_chi2_vs_t = ???

        # Add to ongoing lists 
        dyn_times_list.append(dyn_times)
        incl_vs_t_list.append(incl_vs_t)
        thetajl_vs_t_list.append(thetajl_vs_t)
        phijl_vs_t_list.append(phijl_vs_t)
        theta_J_chi1_vs_t_list.append(theta_J_chi1_vs_t)
        phi_J_chi1_vs_t_list.append(phi_J_chi1_vs_t)
        theta_J_chi2_vs_t_list.append(theta_J_chi2_vs_t)
        phi_J_chi2_vs_t_list.append(phi_J_chi2_vs_t)
    
    # Add to dict
    angles_vs_time_dict[key] = {
        'time_M':dyn_times_list,
        'incl':incl_vs_t_list, 
        'theta_JL':thetajl_vs_t_list, 
        'phi_JL':phijl_vs_t_list,
        'theta_Jchi1':theta_J_chi1_vs_t_list,
        'phi_Jchi1':phi_J_chi1_vs_t_list,
        'theta_Jchi2':theta_J_chi2_vs_t_list,
        'phi_Jchi2':phi_J_chi2_vs_t_list
    }
        
    # Save results as we go
    np.save(savepath, angles_vs_time_dict, allow_pickle=True)
