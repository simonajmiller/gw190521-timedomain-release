## TODO: add better documentation

import numpy as np
from scipy.stats import gaussian_kde
import lal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
import pycbc.psd

def load_posterior_samples(pe_output_dir, date, tcutoffs, verbose=False): 
    
    path_template = pe_output_dir+'{0}_gw190521_{1}_NRSur7dq4_dec8_flow11_fref11_{2}_TstartTend.dat'

    runs = ['insp', 'rd']

    # samples from the various time cuts
    paths = {}
    for run in runs: 
        for tcut in tcutoffs: 
            key = f'{run} {tcut}'
            paths[key] = path_template.format(date,run,tcut)

    # samples from full duration (no time cut) 
    paths['full'] = path_template.format(date,'full','0M') 

    # prior samples
    paths['prior'] = pe_output_dir+'prior_vary_time_and_skypos.dat'

    td_samples = {}
    for k, p in paths.items():
        try:
            td_samples[k] = np.genfromtxt(p, names=True, dtype=float)
        except:
            if verbose:
                print(f'could not find {p}')
                
    return td_samples, tcutoffs


# Function to component masses 
def m1m2_from_mtotq(mtot, q):
    m1 = mtot / (1 + q)
    m2 = mtot - m1
    return m1, m2


# Function to calculate chi_p 
def chi_precessing(m1, a1, tilt1, m2, a2, tilt2):
    q_inv = m1/m2
    A1 = 2. + (3.*q_inv/2.)
    A2 = 2. + 3./(2.*q_inv)
    S1_perp = a1*np.sin(tilt1)*m1*m1
    S2_perp = a2*np.sin(tilt2)*m2*m2
    Sp = np.maximum(A1*S2_perp, A2*S1_perp)
    chi_p = Sp/(A2*m1*m1)
    return chi_p


# Function to calculate chi_eff 
def chi_effective(m1, a1, tilt1, m2, a2, tilt2):
    chieff = (m1*a1*np.cos(tilt1) + m2*a2*np.cos(tilt2))/(m1+m2)
    return chieff


# reflected KDE
def reflected_kde(samples, lower_bound, upper_bound, npoints=500, bw=None): 
    
    if isinstance(npoints, int):
        grid = np.linspace(lower_bound, upper_bound, npoints)
    else:
        grid = npoints
    
    kde_on_grid = gaussian_kde(samples, bw_method=bw)(grid) + \
                  gaussian_kde(2*lower_bound-samples, bw_method=bw)(grid) + \
                  gaussian_kde(2*upper_bound-samples, bw_method=bw)(grid) 
    
    return grid, kde_on_grid

# define function to whiten data 
def whitenData(h_td, psd, freqs):
    
    # Get segment length and sampling rate 
    dt = 0.5 / round(freqs.max())
    df = freqs[1] - freqs[0]
    seglen = 1 / df
    sampling_rate = 1 / dt 
    N = int(seglen * sampling_rate) - 1
    
    # Into fourier domain
    h_fd = np.fft.rfft(h_td, n=N) / N
    
    # Divide out ASD 
    wh_fd = h_fd/np.sqrt(psd * seglen / 4)
    
    # Back into time domain
    wh_td = 0.5*np.fft.irfft(wh_fd) / dt
    wh_td = wh_td[:len(h_td)]
    
    return wh_td

# define function to whiten data in the time domain 
def whitenData_TD(h_td, L): 
    w_h_td = np.linalg.solve(L, h_td)
    return w_h_td

# Function to get the magnitude of a vector v
def get_mag(v): 
    v_squared = [x*x for x in v]
    mag_v = np.sqrt(sum(v_squared))
    return mag_v

# Transform initial spin conditions with LAL
def transform_spins(theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phi_ref):

    # Transform spins 
    m1_SI = m1*lal.MSUN_SI   
    m2_SI = m2*lal.MSUN_SI
    
    nsamps = len(m1)
    incl = np.zeros(nsamps)
    s1x = np.zeros(nsamps)
    s1y = np.zeros(nsamps)
    s1z = np.zeros(nsamps)
    s2x = np.zeros(nsamps)
    s2y = np.zeros(nsamps)
    s2z = np.zeros(nsamps)
    
    for i in range(nsamps): 
        
        incl[i], s1x[i], s1y[i], s1z[i], s2x[i], s2y[i], s2z[i] = SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn[i], phi_jl[i], tilt1[i], tilt2[i], phi12[i], a1[i], a2[i], 
            m1_SI[i], m2_SI[i], f_ref, phi_ref[i]
        )
        
    return incl, s1x, s1y, s1z, s2x, s2y, s2z


# Function to load in psds
def get_PSD(filename, f_low, delta_f): 
    
    # The PSD will be interpolated to the requested frequency spacing
    length = int(1024 / delta_f)
    psd = pycbc.psd.from_txt(filename, length, delta_f, f_low, is_asd_file=False)
    return psd