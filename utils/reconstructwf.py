import numpy as np
import lal
import lalsimulation
import scipy.signal
import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], 'lalsuite-extra/data/lalsimulation')
import sys
import scipy.signal as sig

sys.path.append('../scripts')
from helper_functions import transform_spins

lalsim = lalsimulation


"""
Functions to generate waveform reconstructions
"""

def get_peak_times(*args, **kwargs):
    
    """
    Get the time at which the Hanford strain is loudest (peak time) at geocenter and 
    each inferometer 
    """
    
    # Unpack inputs
    p = kwargs.pop('parameters')
    times = kwargs.pop('times')
    ifos = kwargs.pop('ifos', ['H1', 'L1', 'V1'])
    approx = kwargs.pop('approx', 'NRSur7dq4')

    # Get delta t and length of timeseries
    delta_t = times[1] - times[0]
    tlen = len(times)
    
    # Frequency parameters
    fp = {k: kwargs[k] if k in kwargs else p[k] for k in ['f_ref', 'flow', 'lal_amporder']}
    f_start = fp['flow']*2/(fp['lal_amporder'] + 2.)
    
    # Change spin convention
    iota, s1x, s1y, s1z, s2x, s2y, s2z = transform_spins(p['theta_jn'], p['phi_jl'], p['tilt_1'], p['tilt_2'],
                                          p['phi_12'], p['a_1'], p['a_2'], p['mass_1'], p['mass_2'],
                                          fp['f_ref'], p['phase'])
    chi1 = [s1x, s1y, s1z]
    chi2 = [s2x, s2y, s2z]
        
    # Get time-domain strain
    h_td = generate_lal_waveform(approx, p['mass_1'], p['mass_2'], chi1, chi2, dist_mpc=p['luminosity_distance'],
                                  dt=delta_t, f_low=f_start, f_ref=fp['f_ref'], inclination=iota,
                                  phi_ref=p['phase'], ell_max=None, times=times, triggertime=p['geocent_time'])
    # FFT
    hp_td = h_td.real
    hc_td = -h_td.imag
    fft_norm = delta_t
    hp_fd = np.fft.rfft(hp_td) * fft_norm
    hc_fd = np.fft.rfft(hc_td) * fft_norm
    frequencies = np.fft.rfftfreq(tlen) / fft_norm
    
    # get peak time
    tp_geo_loc = np.argmax(np.abs(h_td))
    tp_geo = times[tp_geo_loc]
    
    geo_gps_time = lal.LIGOTimeGPS(p['geocent_time'])
    gmst = lal.GreenwichMeanSiderealTime(geo_gps_time)

    # Cycle through ifos
    tp_dict = {'geo': tp_geo}
    for ifo in ifos:
        
        detector = lal.cached_detector_by_prefix[ifo]
        
        # get antenna patterns
        Fp, Fc = lal.ComputeDetAMResponse(detector.response, p['ra'], p['dec'], p['psi'], gmst)
        
        # get time delay and align waveform
        # assume reference time corresponds to envelope peak
        timedelay = lal.TimeDelayFromEarthCenter(detector.location,  p['ra'], p['dec'], geo_gps_time)

        fancy_timedelay = lal.LIGOTimeGPS(timedelay)
        timeshift = fancy_timedelay.gpsSeconds + 1e-9*fancy_timedelay.gpsNanoSeconds

        timeshift_vector = np.exp(-2.*1j*np.pi*timeshift*frequencies)
    
        tp_dict[ifo] = tp_geo + timedelay
    
    return tp_dict


def generate_lal_hphc(approximant_key, m1_msun, m2_msun, chi1, chi2, dist_mpc=1,
                      dt=None, f_low=20, f_ref=11, inclination=0, phi_ref=0., epoch=0): 
    
    """
    Generate the plus and cross polarizations for given waveform parameters and approximant
    """

    approximant = lalsim.SimInspiralGetApproximantFromString(approximant_key)

    m1_kg = m1_msun*lal.MSUN_SI
    m2_kg = m2_msun*lal.MSUN_SI
    
    distance = dist_mpc*1e6*lal.PC_SI

    hp, hc = lalsim.SimInspiralChooseTDWaveform(m1_kg, m2_kg,
                                                chi1[0], chi1[1], chi1[2],
                                                chi2[0], chi2[1], chi2[2],
                                                distance, inclination,
                                                phi_ref, 0., 0., 0.,
                                                dt, f_low, f_ref,
                                                param_dict,
                                                approximant)
    return hp, hc


def generate_lal_waveform(*args, **kwargs):
    
    """
    Generate a waveform in a detector with time of coalescence 'triggertime' for given parameters 
    and approximant; if 'hplus' and 'hcross' not passed, uses the generate_lal_hphc function above to 
    generate a waveform. Then aligns places the waveform at the desired time. 
    """
    
    times = kwargs.pop('times')
    triggertime_geo = kwargs.pop('triggertime')
    
    bufLength = len(times)
    delta_t = times[1] - times[0]
    tStart = times[0]
    tEnd = tStart + delta_t * bufLength

    hplus = kwargs.pop('hplus', None)
    hcross = kwargs.pop('hcross', None)
    if (hplus is None) or (hcross is None):
        hplus, hcross = generate_lal_hphc(*args, **kwargs)
    
    # align waveform, based on LALInferenceTemplate
    # https://git.ligo.org/lscsoft/lalsuite/blob/master/lalinference/lib/LALInferenceTemplate.c#L1124

    # The nearest sample in model buffer to the desired time of coalescence (t_c)
    tcSample = round((triggertime_geo - tStart)/delta_t)

    # The actual coalescence time that corresponds to the buffer sample on which the waveform's 
    # t_c lands, i.e. the nearest time in the buffer
    injTc = tStart + tcSample*delta_t

    # The sample at which the waveform reaches t_c
    hplus_epoch = hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds*1E-9
    waveTcSample = round(-hplus_epoch/delta_t)

    # 1 + (number of samples post - t_c in waveform)
    wavePostTc = hplus.data.length - waveTcSample

    # Start and end indices for the buffer
    bufStartIndex = int(tcSample - waveTcSample if tcSample >= waveTcSample else 0)
    bufEndIndex = int(tcSample + wavePostTc if tcSample + wavePostTc <= bufLength else bufLength)
    
    # Length of buffer
    bufWaveLength = bufEndIndex - bufStartIndex
    
    # Start indec for the waveform 
    waveStartIndex = int(0 if tcSample >= waveTcSample else waveTcSample - tcSample)
    
    # Window if we want
    if kwargs.get('window', True) and tcSample >= waveTcSample:
        # smoothly turn on waveform
        window = scipy.signal.tukey(bufWaveLength)
        window[int(0.5*bufWaveLength):] = 1.
    else:
        window = 1
        
    h_td = np.zeros(bufLength, dtype=complex)
    
    # Place waveform 
    h_td[bufStartIndex:bufEndIndex] = window*hplus.data.data[waveStartIndex:waveStartIndex+bufWaveLength] -\
                                      1j*window*hcross.data.data[waveStartIndex:waveStartIndex+bufWaveLength]
    return h_td
