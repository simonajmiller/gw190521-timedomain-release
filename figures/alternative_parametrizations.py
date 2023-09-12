import numpy as np 
from pesummary.gw.conversions import precessing_snr
from helper_functions import * 
import sys
import os
import subprocess

# Generalized chi_p
# Eq.(15) of https://arxiv.org/abs/2011.11948
def calculate_generalizedChiP(m1, a1, tilt1, m2, a2, tilt2, phi12): 
    
    q = m2/m1
    omega_tilda = q*(4*q + 3)/(4 + 3*q) 
    
    term1 = a1*np.sin(tilt1)
    term2 = omega_tilda*a2*np.sin(tilt2)
    term3 = 2*omega_tilda*a1*a2*np.sin(tilt1)*np.sin(tilt2)*np.cos(phi12)
    
    gen_chip = np.sqrt(term1**2 + term2**2 + term3**3)
    
    return gen_chip


# Magnitude of chi_perp
# Eq.(9) of https://arxiv.org/abs/2012.02209
def calculate_magnitudeChiPerp(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z): 

    # dimensonless spin vectors
    vec_chi1 = [s1x, s1y, s1z]
    vec_chi2 = [s2x, s2y, s2z]
    
    # add dimension
    vec_S1 = np.array(vec_chi1)*m1*m1
    vec_S2 = np.array(vec_chi2)*m2*m2
    
    # total spin 
    vec_S = vec_S1 + vec_S2 
        
    # mag of in plane (perp) components
    S1_perp = get_mag(vec_S1[0:-1])
    S2_perp = get_mag(vec_S2[0:-1])
    S_perp = get_mag(vec_S[0:-1])
    
    # figured out norm based on conditions
    mask = S1_perp>=S2_perp
    norm1 = m1*m1 + S2_perp
    norm2 = m2*m2 + S1_perp
        
    # calculate mag_ chiperp
    mag_chiperp = np.zeros(len(m1))
    mag_chiperp[mask] = S_perp[mask]/norm1[mask]
    mag_chiperp[~mask] = S_perp[~mask]/norm2[~mask]
    
    return mag_chiperp


# Rho_p
# Eq.(39) of https://arxiv.org/abs/1908.05707
# See: https://git.ligo.org/lscsoft/pesummary/-/blob/master/pesummary/gw/conversions/snr.py#L597
def calculate_rhoP(m1, a1, tilt1, m2, a2, tilt2, phi12, iota, theta_jn, phi_jl, ra, dec, psi, time, distance,
                   phi_ref, f_ref, duration, psd_files, delta_f=1/256.): 
    
    # Get psds
    psd_dict = {k:get_PSD(f, f_ref, delta_f) for k,f in psd_files.items()}
    psd_freqs = psd_dict['H1'].sample_frequencies
    f_max = psd_freqs[-1]
    df = psd_freqs[1] -  psd_freqs[0]
        
    # Format time and skypos into arrays
    nsamps = len(m1)
    ra_array = np.ones(nsamps) * ra 
    dec_array = np.ones(nsamps) * dec 
    time_array = np.ones(nsamps) * time 
    psi_array =  np.ones(nsamps) * psi 
    
    # The angle between the total angular momentum and the total orbital angular momentum
    # (also known as theta_jl) -- see Fig. 1 of https://arxiv.org/abs/1908.05707
    beta = iota - theta_jn 
        
    # Use PEsummary function; note: calculating with IMRPhenomXPHM because NRSur because only freq. domain 
    # waveforms work with this functon 
    rho_p = precessing_snr(
        m1, m2, beta, psi_array, a1, a2, tilt1, tilt2, phi12, theta_jn,
        ra_array, dec_array, time_array, phi_jl, distance, phi_ref, f_low=f_ref, psd=psd_dict, 
        approx="IMRPhenomXPHM",f_final=f_max, f_ref=f_ref, duration=duration, df=df, 
        debug=False
    )
    
    return rho_p


if __name__ == '__main__':
    
    # Load posterior samples 
    data_dir = '/Users/smiller/Documents/gw190521-timedomain-release/data/' 
    template = data_dir+'063023_gw190521_{0}_NRSur7dq4_dec8_flow11_fref11_{1}_TstartTend.dat'

    paths = {
        'full':template.format('full', '0M'),
        'm10M ':template.format('rd', 'm10M'),
        'm40M ':template.format('rd', 'm40M')
    }

    td_samples = {k: np.genfromtxt(p, names=True, dtype=float) for k, p in paths.items()}
    
    # Where to save 
    savepath = data_dir + 'alternate_precession_parametrizations.npy'
    
    # Format into way to plot 
    posteriors = {}

    # Runs 
    runs = paths.keys()
    for run in runs:
        
        print(run)

        # Get samples
        samples =  td_samples[run]
        m1, m2 = m1m2_from_mtotq(samples['mtotal'], samples['q'])
        a1 = samples['chi1']
        a2 = samples['chi2']
        tilt1 = samples['tilt1']
        tilt2 = samples['tilt2']
        phi12 = samples['phi12']
        theta_jn = samples['theta_jn']
        phi_jl = samples['phi_jl']
        phi_ref = samples['phase']
        distance = samples['dist']

        # Transform spin convention
        f_ref = 11
        incl, s1x, s1y, s1z, s2x, s2y, s2z = transform_spins(theta_jn, phi_jl, tilt1, tilt2, 
                                                               phi12, a1, a2, m1, m2, f_ref, phi_ref)

        # Other info needed for calculating the other spin parameterizations
        duration = 0.7    
        t0_0M_geo = 1242442967.405764
        ra = 6.07546535866838
        dec = -0.8000357325337637
        psi = 2.443070879119043
        psd_filenames = {k:data_dir+f'GW190521_data/glitch_median_PSD_forLI_{k}.dat' for k in ['H1', 'L1', 'V1']}


        # Traditional chi_p 
        chip = chi_precessing(m1, a1, tilt1, m2, a2, tilt2)

        # Generalized chi_p
        gen_chip = calculate_generalizedChiP(m1, a1, tilt1, m2, a2, tilt2, phi12)

        # Mag. of chi_perp 
        mag_chiperp = calculate_magnitudeChiPerp(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)

        # Rho_p 
        rho_p = calculate_rhoP(m1, a1, tilt1, m2, a2, tilt2, phi12, incl, theta_jn, phi_jl, ra, dec, psi, t0_0M_geo,
                        distance, phi_ref, f_ref, duration, psd_filenames)
        
        # Add to dict
        posteriors[run] = {
            'chip':chip,
            'gen chip':gen_chip, 
            'mag chiperp':mag_chiperp, 
            'rho_p':rho_p
        }
    
        # Save results as we go
        np.save(savepath, posteriors, allow_pickle=True)
    