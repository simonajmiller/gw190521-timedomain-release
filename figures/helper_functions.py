## TODO: add documentation

import numpy as np
from scipy.stats import gaussian_kde

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
    paths['full'] = path_template.format('050323','full','0M') # TODO update with date

    # prior samples
    paths['prior'] = pe_output_dir+'gw190521_sample_prior.dat'

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


# reflected KDE
def reflected_kde(samples, lower_bound, upper_bound, npoints=500): 
    
    grid = np.linspace(lower_bound, upper_bound, npoints)
    
    kde_on_grid = gaussian_kde(samples)(grid) + \
                  gaussian_kde(2*lower_bound-samples)(grid) + \
                  gaussian_kde(2*upper_bound-samples)(grid) 
    
    return grid, kde_on_grid
    