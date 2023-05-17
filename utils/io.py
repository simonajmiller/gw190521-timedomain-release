from pylab import *
import h5py
import lal
from . import reconstructwf as rwf
import scipy.linalg as sl
import scipy.signal as sig
import scipy.stats as ss

def load_raw_data(path='../data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5',
                  ifos=('H1', 'L1', 'V1'), verbose=True):
    raw_time_dict = {}
    raw_data_dict = {}
    
    for ifo in ifos:
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

def get_pe(raw_time_dict, path='/home/simona.miller/gw190521-td/data/GW190521_posterior_samples.h5', 
           psd_path=None, verbose=True, return_likelihood=False):
    
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
    
    ra = pe_samples['ra'][imax]
    dec = pe_samples['dec'][imax]
    psi = pe_samples['psi'][imax]
    
    maxP_skypos = {'ra':ra, 'dec':dec, 'psi':psi}
    
    # Set truncation time
    amporder = 1
    flow = 11
    fstart = flow * 2./(amporder+2)
    peak_times = rwf.get_peak_times(parameters=pe_samples[imax],
                                    times=raw_time_dict[ifos[0]], f_ref=11,
                                    flow=flow, lal_amporder=1)
    
    # Peak time = coalescence time 
    tpeak_H = peak_times['H1']
    dt_H = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix['H1'].location,
                                        ra, dec, lal.LIGOTimeGPS(tpeak_H))
    
    tpeak_geocent = tpeak_H - dt_H
    
    tpeak_dict, ap_dict = get_tgps_and_ap_dicts(tpeak_geocent, ifos, ra, 
                                               dec, psi, verbose=verbose)
    if return_likelihood:
        
        logL = pe_samples['log_likelihood']
        imax_L = argmax(logL)
        maxL_skypos = {'ra':pe_samples['ra'][imax_L], 'dec':pe_samples['dec'][imax_L], 
                   'psi':pe_samples['psi'][imax_L]}
        
        _, ap_dict_L = get_tgps_and_ap_dicts(tpeak_geocent, ifos, maxL_skypos['ra'], 
                                               maxL_skypos['dec'], maxL_skypos['psi'])
        
        return tpeak_geocent, tpeak_dict, ap_dict, pe_samples, log_prob, pe_psds, maxP_skypos, logL, maxL_skypos, ap_dict_L
    
    else:
        return tpeak_geocent, tpeak_dict, ap_dict, pe_samples, log_prob, pe_psds, maxP_skypos


def get_tgps_and_ap_dicts(tgps_geocent, ifos, ra, dec, psi, verbose=True):
    
    gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(tgps_geocent))
    
    tgps_dict = {}
    ap_dict = {}
    for ifo in ifos:
        dt_ifo = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location,
                                              ra, dec, lal.LIGOTimeGPS(tgps_geocent))
        tgps_dict[ifo] = tgps_geocent + dt_ifo
        ap_dict[ifo] = lal.ComputeDetAMResponse(lal.cached_detector_by_prefix[ifo].response,
                                                ra, dec, psi, gmst)
        if verbose:
            print(ifo, tgps_dict[ifo], ap_dict[ifo])
            
    return tgps_dict, ap_dict
    

def condition(raw_time_dict, raw_data_dict, t0_dict, ds_factor=16, f_low=11,
              scipy_decimate=True, verbose=True):
    
    ifos = list(raw_time_dict.keys())
    data_dict = {}
    time_dict = {}
    i0_dict = {}
    
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

