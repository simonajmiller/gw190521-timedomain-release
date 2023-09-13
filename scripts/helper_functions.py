## TODO: add better documentation

import numpy as np
from scipy.stats import gaussian_kde
import lal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
import pycbc.psd
from pesummary.gw.conversions import precessing_snr

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