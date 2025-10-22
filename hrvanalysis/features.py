# third-party
import numpy as np
import pandas as pd
import biosppy as bp
import scipy
from PyEMD import EMD
from PyEMD.compact import filt6, pade6


def get_hrv_features(segments, ts, fs, feature_list, feature_names, fbands=None):
    ''' Compute HRV features for each segment. Returns formatted DataFrame.

    Parameters
    ---------- 
    segments : array-like, shape (#segments, #points)
        Filtered ECG segments. Segments should have at least 60s duration.
    ts : array-like, shape (#segments, )
        Startime of corresponding segment within acquisition file (in seconds).
    fs : int
        Sampling frequency.
    feature_list : list of str
        List of HRV features to compute for each segment. Possible values are:
        'rr_mean', 'sdnn', 'rmssd', 'pnn50', 'lf_pwr', 'lf_hf', 'hf_pwr'.
    feature_names : list of str
        List of names corresponding to each feature in feature_list, to be used as column 'name' in the output DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ["start_timedelta", "name", "value"], where each entry corresponds to a single feature extracted from a single segment. Lenght is equal to #segments * #HRV features.
    '''

    df = pd.DataFrame(columns=['start_timedelta', 'name', 'value'])
    hrv_features = {}
    fs = float(fs)

    for i in range(segments.shape[0]):
        (segment_peaks,) = bp.signals.ecg.hamilton_segmenter(
            segments[i, :], sampling_rate=fs)

        (rpeaks,) = bp.signals.ecg.correct_rpeaks(
            segments[i, :], segment_peaks, sampling_rate=fs)

        rri = _compute_and_process_rri(rpeaks, fs)

        # compute duration
        duration = np.sum(rri) / 1000.  # seconds

        # Compute time-domain HRV features
        hrv_td = bp.signals.hrv.hrv_timedomain(rri=rri,
                                               duration=duration,
                                               detrend_rri=False,
                                               show=False)
        out = dict(hrv_td)

        hrv_fd = hrv_frequencydomain(rri=rri, sampling_rate=1/0.193)
        out.update(hrv_fd)

        for key, n_key in zip(feature_list, feature_names):
            hrv_features[n_key] = out.pop(key)

        for feat in hrv_features.keys():
            df.loc[len(df)] = [pd.to_timedelta(
                ts[i], unit='s').round('s'), feat, hrv_features[feat]]

    return df


def _compute_and_process_rri(rpeaks, sampling_rate):
    ''' Compute and process RRI from R-peaks. Adapted from BioSPPy.'''
    # compute RRIs
    rpeaks = np.array(rpeaks, dtype=float)
    rri = bp.signals.hrv.compute_rri(rpeaks=rpeaks, sampling_rate=sampling_rate,
                                     filter_rri=False)

    # cubic spline interpolation to uniform time grid
    t_rri = np.cumsum(rri) / 1000.  # to seconds
    t_rri = t_rri - t_rri[0]
    cs = scipy.interpolate.CubicSpline(t_rri, rri)

    resample_period = 0.193  # seconds
    t_uniform = np.arange(0, t_rri[-1], resample_period)

    # Evaluate the spline at uniform times (resampled RRI signal)
    rri = cs(t_uniform)
    return rri


def hrv_frequencydomain(rri, sampling_rate):
    ''' Compute HRV frequency-domain features using EMD and Hilbert-Huang Transform.

    Returns
    -------
    LF_HF_ratio : array-like
        Time series of LF/HF ratio computed from the RRI signal.
    '''
    out = bp.utils.ReturnTuple((), ())

    # convert RRI to seconds
    rri = np.array(rri) / 1000.0

    # Step 1: Decompose RRI signal into IMFs using EMD
    emd = EMD()
    imfs = emd(rri)

    LF_power, HF_power = 0, 0

    # print(f"Number of IMFs: {imfs.shape[0]}")

    # Step 2: Select IMFs corresponding to LF (0.038–0.15 Hz) and HF (0.15–0.6 Hz)
    # A simple approach: compute mean instantaneous frequency for each IMF and select
    LF_range = (0.038, 0.15)
    HF_range = (0.15, 0.6)

    t = np.linspace(0, len(rri) / sampling_rate, len(rri))
    inst_freq = _calc_inst_freq(imfs, t)

    for i, imf in enumerate(imfs):
        analytic_signal = scipy.signal.hilbert(imf)

        inst_freq_imf = inst_freq[i, :]

        # print(
        #     f'min freq: {inst_freq_imf.min()} | max freq: {inst_freq_imf.max()}')

        LF_idx = np.where((LF_range[0] <= inst_freq_imf)
                          & (inst_freq_imf <= LF_range[1]))
        HF_idx = np.where((HF_range[0] <= inst_freq_imf)
                          & (inst_freq_imf <= HF_range[1]))

        LF_power += np.sum(np.abs(analytic_signal[LF_idx])**2)
        HF_power += np.sum(np.abs(analytic_signal[HF_idx])**2)

    LF_HF_ratio = LF_power / HF_power if HF_power != 0 else np.nan

    # print(
    #     f'LF Power: {LF_power}, HF Power: {HF_power}, LF/HF Ratio: {LF_HF_ratio}')
    return {'lf_pwr': LF_power*1000., 'hf_pwr': HF_power*1000., 'lf_hf': LF_HF_ratio}


def _calc_inst_freq(imfs, t, order=False, alpha=None):
    """Extracts instantaneous frequency through the Hilbert Transform. From PyEMD."""
    inst_phase = _calc_inst_phase(imfs, alpha=alpha)
    if order is False:
        inst_freqs = np.diff(inst_phase) / (2 * np.pi * (t[1] - t[0]))
        inst_freqs = np.concatenate(
            (inst_freqs, inst_freqs[:, -1].reshape(inst_freqs[:, -1].shape[0], 1)), axis=1)
    else:
        inst_freqs = [pade6(row, t[1] - t[0]) / (2.0 * np.pi)
                      for row in inst_phase]
    if alpha is None:
        return np.array(inst_freqs)
    else:
        # Filter freqs
        return np.array([filt6(row, alpha) for row in inst_freqs])


def _calc_inst_phase(imfs, alpha):
    """Extract analytical signal through the Hilbert Transform."""
    analytic_signal = scipy.signal.hilbert(
        imfs)  # Apply Hilbert transform to each row
    if alpha is not None:
        assert -0.5 < alpha < 0.5, "`alpha` must be in between -0.5 and 0.5"
        real_part = np.array([filt6(row.real, alpha)
                             for row in analytic_signal])
        imag_part = np.array([filt6(row.imag, alpha)
                             for row in analytic_signal])
        analytic_signal = real_part + 1j * imag_part
    # Compute angle between img and real
    phase = np.unwrap(np.angle(analytic_signal))
    if alpha is not None:
        phase = np.array([filt6(row, alpha) for row in phase])  # Filter phase
    return phase
