import numpy as np
import lal
import lalsimulation
import scipy.signal
import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], 'lalsuite-extra/data/lalsimulation')
import sys

# utilities from https://git.ligo.org/waveforms/reviews/nrsur7dq4/blob/master/utils.py
# and from Carl
lalsim = lalsimulation

def change_spin_convention(theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phi_orb=0.):
    iota, S1x, S1y, S1z, S2x, S2y, S2z = lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                                             theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2,
                                             m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_ref, phi_orb)
    spin1 = [S1x, S1y, S1z]
    spin2 = [S2x, S2y, S2z]
    return spin1, spin2, iota

def set_single_mode(params, l, m):
    """ Sets modes in params dict.
        Only adds (l,m) and (l,-m) modes.
    """
    # First, create the 'empty' mode array
    ma = lalsimulation.SimInspiralCreateModeArray()
    # add (l,m) and (l,-m) modes
    lalsimulation.SimInspiralModeArrayActivateMode(ma, l, m)
    lalsimulation.SimInspiralModeArrayActivateMode(ma, l, -m)    
    # then insert the ModeArray into the LALDict params
    lalsimulation.SimInspiralWaveformParamsInsertModeArray(params, ma)
    return params

def generate_lal_hphc(approximant_key, m1_msun, m2_msun, chi1, chi2, dist_mpc=1,
                      dt=None, f_low=20, f_ref=11, inclination=0, phi_ref=0.,  #3/8/22: changed fref=20 -> fref=11
                      ell_max=None, single_mode=None, epoch=0): 

    approximant = lalsim.SimInspiralGetApproximantFromString(approximant_key)

    m1_kg = m1_msun*lal.MSUN_SI
    m2_kg = m2_msun*lal.MSUN_SI
    
    distance = dist_mpc*1e6*lal.PC_SI

    if single_mode is not None and ell_max is not None:
        raise Exception("Specify only one of single_mode or ell_max")

    param_dict = lal.CreateDict()

    if ell_max is not None:
        # If ell_max, load all modes with ell <= ell_max
        ma = lalsim.SimInspiralCreateModeArray()
        for ell in range(2, ell_max+1):
            lalsim.SimInspiralModeArrayActivateAllModesAtL(ma, ell)
        lalsim.SimInspiralWaveformParamsInsertModeArray(param_dict, ma)
    elif single_mode is not None:
        # If a single_mode is given, load only that mode (l,m) and (l,-m)
        param_dict = set_single_mode(param_dict, single_mode[0], single_mode[1])

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

    # /* The nearest sample in model buffer to the desired tc. */
    tcSample = round((triggertime_geo - tStart)/delta_t)

    # /* The actual coalescence time that corresponds to the buffer
    #    sample on which the waveform's tC lands. */
    # i.e. the nearest time in the buffer
    injTc = tStart + tcSample*delta_t

    # /* The sample at which the waveform reaches tc. */
    hplus_epoch = hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds*1E-9
    waveTcSample = round(-hplus_epoch/delta_t)

    # /* 1 + (number of samples post-tc in waveform) */
    wavePostTc = hplus.data.length - waveTcSample

    # bufStartIndex = (tcSample >= waveTcSample ? tcSample - waveTcSample : 0);
    bufStartIndex = int(tcSample - waveTcSample if tcSample >= waveTcSample else 0)
    # size_t bufEndIndex = (wavePostTc + tcSample <= bufLength ? wavePostTc + tcSample : bufLength);
    bufEndIndex = int(tcSample + wavePostTc if tcSample + wavePostTc <= bufLength else bufLength)
    bufWaveLength = bufEndIndex - bufStartIndex
    waveStartIndex = int(0 if tcSample >= waveTcSample else waveTcSample - tcSample)
    
    if kwargs.get('window', True) and tcSample >= waveTcSample:
        # smoothly turn on waveform
        window = scipy.signal.tukey(bufWaveLength)
        window[int(0.5*bufWaveLength):] = 1.
    else:
        window = 1
    h_td = np.zeros(bufLength, dtype=complex)
    h_td[bufStartIndex:bufEndIndex] = window*hplus.data.data[waveStartIndex:waveStartIndex+bufWaveLength] -\
                                      1j*window*hcross.data.data[waveStartIndex:waveStartIndex+bufWaveLength]
    return h_td

def project_fd(hp_fd, hc_fd, frequencies, Fp=None, Fc=None, time_delay=None, ifo=None):
    
    if ifo is not None:
        geo_gps_time = lal.LIGOTimeGPS(triggertime_geo)
        gmst = lal.GreenwichMeanSiderealTime(geo_gps_time)

        detector = lal.cached_detector_by_prefix[ifo]
        # get antenna patterns
        Fp, Fc = lal.ComputeDetAMResponse(detector.response, p['ra'], p['dec'], p['psi'], gmst)
        # get time delay and align waveform
        # assume reference time corresponds to envelope peak
        time_delay = lal.TimeDelayFromEarthCenter(detector.location,  p['ra'], p['dec'], geo_gps_time)
    
    fancy_timedelay = lal.LIGOTimeGPS(time_delay)
    timeshift = fancy_timedelay.gpsSeconds + 1e-9*fancy_timedelay.gpsNanoSeconds
    
    timeshift_vector = np.exp(-2.*1j*np.pi*timeshift*frequencies)
    
    h_fd = (Fp*hp_fd + Fc*hc_fd)*timeshift_vector
    return h_fd

def project_td(h_td, times, **kwargs):
    hp_td = h_td.real
    hc_td = -h_td.imag
    
    fft_norm = times[1] - times[0]
    hp_fd = np.fft.rfft(hp_td) * fft_norm
    hc_fd = np.fft.rfft(hc_td) * fft_norm
    frequencies = np.fft.rfftfreq(len(times)) / fft_norm

    h_fd = project_fd(hp_fd, hc_fd, frequencies, **kwargs)
    return np.fft.irfft(h_fd) / fft_norm

def get_peak_times(*args, **kwargs):
    p = kwargs.pop('parameters')
    times = kwargs.pop('times')
    ifos = kwargs.pop('ifos', ['H1', 'L1', 'V1'])
    approx = kwargs.pop('approx', 'NRSur7dq4')
    
    delta_t = times[1] - times[0]
    tlen = len(times)
    
    fp = {k: kwargs[k] if k in kwargs else p[k] for k in ['f_ref', 'flow', 'lal_amporder']}

    chi1, chi2, iota = change_spin_convention(p['theta_jn'], p['phi_jl'], p['tilt_1'], p['tilt_2'],
                                          p['phi_12'], p['a_1'], p['a_2'], p['mass_1'], p['mass_2'],
                                          fp['f_ref'], p['phase'])
    
    f_start = fp['flow']*2/(fp['lal_amporder'] + 2.)
    # get strain
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

def get_fd_waveforms(*args, **kwargs):
    p = kwargs.pop('parameters')
    times = kwargs.pop('times')
    ifos = kwargs.pop('ifos', ['H1', 'L1', 'V1'])
    approx = kwargs.pop('approx', 'NRSur7dq4')
    
    delta_t = times[1] - times[0]
    tlen = len(times)
    
    fp = {k: kwargs[k] if k in kwargs else p[k] for k in ['f_ref', 'flow', 'lal_amporder']}

    chi1, chi2, iota = change_spin_convention(p['theta_jn'], p['phi_jl'], p['tilt1'], p['tilt2'],
                                          p['phi12'], p['a1'], p['a2'], p['m1'], p['m2'],
                                          fp['f_ref'], p['phase'])
    
    f_start = fp['flow']*2/(fp['lal_amporder'] + 2.)
    # get strain
    h_td = generate_lal_waveform(approx, p['m1'], p['m2'], chi1, chi2, dist_mpc=p['dist'],
                                  dt=delta_t, f_low=f_start, f_ref=fp['f_ref'], inclination=iota,
                                  phi_ref=p['phase'], ell_max=None, times=times, triggertime=p['time'])
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
    
    geo_gps_time = lal.LIGOTimeGPS(p['time'])
    gmst = lal.GreenwichMeanSiderealTime(geo_gps_time)

    h_fd_dict = {}
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
        h_fd_dict[ifo] = (Fp*hp_fd + Fc*hc_fd)*timeshift_vector
    
    return h_fd_dict
