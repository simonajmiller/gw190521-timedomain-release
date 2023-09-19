from pylab import *
import lal
import h5py
from . import reconstructwf as rwf
import scipy.signal as sig


"""
Functions to input and condition GW190521 data
"""

def load_raw_data(path='../data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5',
                  ifos=('H1', 'L1', 'V1'), verbose=True):
    
    """
    Load in raw interferometer timeseries strain data
    
    Parameters
    ----------
    path : string (optional)
        file path to location of raw timeseries strain data for GW190521; the {} 
        are where the names of the interferometers go
    ifos : tuple of strings (optional)
        which interometers to load data from (some combination of 'H1', 'L1',
        and 'V1')
    verbose : boolean (optional)
        whether or not to print out information as the data is loaded
    
    Returns
    -------
    raw_time_dict : dictionary
        time stamps for the data from each ifo 
    raw_data_dict : dictionary
        the data from each ifo 
    """
    
    raw_time_dict = {}
    raw_data_dict = {}
    
    for ifo in ifos:
        # for real data downloaded from gwosc...
        with h5py.File(path.format(ifo[0], ifo), 'r') as f:
            strain = array(f['strain/Strain'])
            T0 = f['meta/GPSstart'][()]
            ts = T0 + arange(len(strain))*f['meta/Duration'][()]/len(strain)
        
        raw_time_dict[ifo] = ts
        raw_data_dict[ifo] = strain
    
        fsamp = 1.0/(ts[1]-ts[0])
        if verbose:
            print("Raw %s data sampled at %.1f Hz" % (ifo, fsamp))
            
    return raw_time_dict, raw_data_dict


def get_pe(raw_time_dict, path='../data/GW190521_posterior_samples.h5', 
           psd_path=None, verbose=True):
    
    """
    Load in parameter estimation (pe) samples from LVC GW190521 analysis, and calculate
    the peak strain time at geocenter and each detector, the detector antenna patterns, 
    the psds, and the maximum posterior sky position    
    
    Parameters
    ----------
    raw_time_dict : dictionary
        output from load_raw_data function above
    path : string (optional)
        file path for pe samples
    psd_path : string (optional)
        if power spectral density (psd) in a different file than the pe samples, provide
        the file path here
    verbose : boolean (optional)
        whether or not to print out information as the data is loaded
    
    Returns
    -------
    tpeak_geocent : float
        the peak strain time at geocenter 
    tpeak_dict : dictionary
        the peak strain time at each interferometer  
    ap_dict : dictionary
        the antenna patterns at peak strain time for each interferometer  
    pe_samples : dictionary
        parameter estimation samples released by the LVC
    log_prob : `numpy.array`
        log posterior probabilities corresponding to each pe sample
    pe_psds : dictionary
        the power spectral densities for each interferometer in the format 
        (frequencies, psd values)
    maxP_skypos : dictionary
        the right ascension, declination, and polarization angle for the maximum 
        posterior sample
    """
    
    # Interferometer names 
    ifos = list(raw_time_dict.keys())
    
    # Load in posterior samples
    with h5py.File(path, 'r') as f:
        pe_samples = f['NRSur7dq4']['posterior_samples'][()]
    
    # Load in PSDs
    pe_psds = {}
    if psd_path is None: # use same file as posteriors 
        with h5py.File(path, 'r') as f:
            for ifo in ifos:
                pe_psds[ifo] = f['NRSur7dq4']['psds'][ifo][()]
    else: # use different, provided file
        for ifo in ifos: 
            pe_psds[ifo] = genfromtxt(psd_path.format(ifo), dtype=float)
            
    # Find sample where posterior is maximized
    log_prob = pe_samples['log_likelihood'] + pe_samples['log_prior']
    imax = argmax(log_prob)
    
    # Sky position for the max. posterior sample
    ra = pe_samples['ra'][imax]   # right ascension
    dec = pe_samples['dec'][imax] # declination
    psi = pe_samples['psi'][imax] # polarization angle
    maxP_skypos = {'ra':ra, 'dec':dec, 'psi':psi}
    
    # Set truncation time
    amporder = 1
    flow = 11
    fstart = flow * 2./(amporder+2)
    peak_times = rwf.get_peak_times(parameters=pe_samples[imax], times=raw_time_dict[ifos[0]], 
                                    f_ref=11, flow=flow, lal_amporder=1)
    
    # Get peak time of the signal in LIGO Hanford
    tpeak_H = peak_times['H1']
    dt_H = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix['H1'].location,
                                        ra, dec, lal.LIGOTimeGPS(tpeak_H))
    
    # Translate to geocenter time
    tpeak_geocent = tpeak_H - dt_H
    
    # Get peak time and antenna pattern for all ifos
    tpeak_dict, ap_dict = get_tgps_and_ap_dicts(tpeak_geocent, ifos, ra, 
                                               dec, psi, verbose=verbose)
    
    return tpeak_geocent, tpeak_dict, ap_dict, pe_samples, log_prob, pe_psds, maxP_skypos


def get_tgps_and_ap_dicts(tgps_geocent, ifos, ra, dec, psi, verbose=True):
    
    """
    Get the time and antenna pattern at each detector at the given geocenter time and 
    sky position 
    
    Parameters
    ----------
    tgps_geocent : float
        geocenter time
    ifos : tuple of strings (optional)
        which interometers to load data from (some combination of 'H1', 'L1',
        and 'V1')
    ra : float
        right ascension
    dec : float
        declination
    psi : float
        polarization angle
    verbose : boolean (optional)
        whether or not to print out information calculated
    
    Returns
    -------
    tgps_dict : dictionary
        time at each detector at the given geocenter time and sky position 
    ap_dict : dictionary
        antenna pattern for each interferometer at the given geocenter time and sky 
        position 
    """
    
    tgps_dict = {}
    ap_dict = {}
    
    # Greenwich mean sidereal time 
    gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(tgps_geocent))
    
    # Cycle through interferometers
    for ifo in ifos:
        
        # Calculate time delay between geocenter and this ifo 
        dt_ifo = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location,
                                              ra, dec, lal.LIGOTimeGPS(tgps_geocent))
        tgps_dict[ifo] = tgps_geocent + dt_ifo
        
        # Calculate antenna pattern 
        ap_dict[ifo] = lal.ComputeDetAMResponse(lal.cached_detector_by_prefix[ifo].response,
                                                ra, dec, psi, gmst)
        if verbose:
            print(ifo, tgps_dict[ifo], ap_dict[ifo])
            
    return tgps_dict, ap_dict
    

def condition(raw_time_dict, raw_data_dict, t0_dict, ds_factor=16, f_low=11,
              scipy_decimate=True, verbose=True):
    
    """
    Filter and downsample the data, and locate target sample corresponding
    to the times in t0_dict
    
    Parameters
    ----------
    raw_time_dict : dictionary
        time stamps for the raw strain data from each ifo (output from load_raw_data 
        function above)
    raw_data_dict : dictionary
        the raw strain data data from each ifo (output from load_raw_data function 
        above)
    t0_dict : dictionary
        time at each interferometer find the sample index of
    ds_factor : float (optional)
        downsampling factor for the data; defaults to 16 which takes ~16kHz data to 
        1024 Hz data
    f_low : float (optional)
        frequency for the highpass filter
    scipy_decimate : boolean (optional)
        whether or not to use the scipy decimate function for downsampling, defaults
        to True 
    verbose : boolean (optional)
        whether or not to print out information calculated
        
    Returns
    -------
    time_dict : dictionary
        time stamps for the conditioned strain data from each ifo 
    data_dict : dictionary
        the conditioned strain data from each ifo 
    i0_dict : dictionary
        indices corresponding to the time values in t0_dict
    """
    
    ifos = list(raw_time_dict.keys())
    data_dict = {}
    time_dict = {}
    i0_dict = {}
    
    # Cycle through interferometers
    for ifo in ifos:
        
        # Find the nearest sample in H to the designated cutoff time t0
        i = np.argmin(np.abs(raw_time_dict[ifo] - t0_dict[ifo]))
        ir = i % ds_factor
        print('Rolling {:s} by {:d} samples'.format(ifo, ir))
        raw_data = roll(raw_data_dict[ifo], -ir)
        raw_time = roll(raw_time_dict[ifo], -ir)
        
        # Filter
        if f_low:
            fny = 0.5/(raw_time[1]-raw_time[0])
            b, a = sig.butter(4, f_low/fny, btype='highpass', output='ba')
            data = sig.filtfilt(b, a, raw_data)
        else:
            data = raw_data.copy()
        
        # Decimate
        if ds_factor > 1:
            if scipy_decimate:
                data = sig.decimate(data, ds_factor, zero_phase=True)
            else:
                data = data[::ds_factor]
            time = raw_time[::ds_factor]
        else: 
            time = raw_time
        
        # Subtract mean and store
        data_dict[ifo] = data - mean(data)
        time_dict[ifo] = time
        
        # Locate target sample
        i0_dict[ifo] = np.argmin(np.abs(time - t0_dict[ifo]))
        if verbose:
            print('tgps_{:s} = {:.6f}'.format(ifo, t0_dict[ifo]))
            print('t0_{:s} - tgps_{:s} is {:.2e} s\n'.format(ifo, ifo, time[i0_dict[ifo]]-t0_dict[ifo]))
            
    return time_dict, data_dict, i0_dict

