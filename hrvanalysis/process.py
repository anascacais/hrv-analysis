# third-party
import numpy as np
import pandas as pd
import biosppy as bp
import scipy
from scipy.optimize import linear_sum_assignment

from hrvanalysis.visualization import plot_channels


def get_filtered_ecg_df(signal, fs, ftype='FIR', band='bandpass', frequency=[0.67, 45], check_if_inverted=True, use_display=False):
    ''' Applies digital filtering to signal (1D array). The default uses a FIR bandpass filter with order 1.5*fs, and cutoff frequency range of [0.67, 45]. Adapted from BioSPPy. 

    Parameters
    ---------- 
    sinal : array, shape (#points,)
        ECG channel to filter.
    fs : float
        Sampling frequency.
    ftype : str
        Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').
            * Notch filter ('notch').
    band : str
        Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').
    order : int
        Order of the filter.
    frequency : int, float, list, array
        Cutoff frequencies; format depends on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.

    Returns 
    -------
    ecg_df : pd.DataFrame
        DataFrame with 3 columns: "time" (in seconds), "Bipolar ECG" and "Filtered ECG". 
    '''

    signal = np.array(signal)
    ts = np.linspace(0, len(signal)/fs, len(signal))

    # filter in order to remove movement and electrical noise
    order = int(1.5 * fs)
    filtered, _, _ = bp.tools.filter_signal(
        signal=np.reshape(signal, (-1,)),
        ftype=ftype,
        band='lowpass',
        order=order,
        frequency=45,
        sampling_rate=fs,
    )

    # polynomial trend removal
    signal = scipy.signal.detrend(signal, type='linear')

    # bandpass filter to isolate the periodicities of interest
    order = int(1.5 * fs)
    filtered, _, _ = bp.tools.filter_signal(
        signal=np.reshape(signal, (-1,)),
        ftype=ftype,
        band=band,
        order=order,
        frequency=frequency,
        sampling_rate=fs,
    )

    if check_if_inverted:
        (segment_peaks,) = bp.signals.ecg.hamilton_segmenter(
            filtered, sampling_rate=fs)
        (rpeaks,) = bp.signals.ecg.correct_rpeaks(
            filtered, segment_peaks, sampling_rate=fs)

        (segment_peaks,) = bp.signals.ecg.hamilton_segmenter(
            (-1)*filtered, sampling_rate=fs)
        (rpeaks_inv,) = bp.signals.ecg.correct_rpeaks(
            (-1)*filtered, segment_peaks, sampling_rate=fs)

        # Hungarian algorithm
        cost = np.abs(np.subtract.outer(rpeaks, rpeaks_inv))
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = [(rpeaks[i], rpeaks_inv[j]) for i, j in zip(row_ind, col_ind)]
        negative_diff = sum(d < 0 for d in [i - j for i, j in pairs])

        if negative_diff < len(pairs) * 0.5:
            print(
                f'\nINVERTED ECG DETECTED.')

            plot_channels(pd.DataFrame({'time': ts, 'original': filtered, 'inverted': (-1)*filtered}), channels=[
                "original", "inverted"], datetime_as_index=True, use_display=use_display)

            resp = input('\nShould we proceed with signal inversion? (y/n): ')
            if resp.lower() == 'y':
                filtered = (-1) * filtered
                print('Signal inverted.\n')

    ecg_df = pd.DataFrame(
        {'time': ts, 'Bipolar ECG': np.reshape(signal, (-1,))})
    ecg_df['Filtered ECG'] = filtered - np.mean(filtered)

    return ecg_df


def get_rpeaks(signal, fs):
    ''' ECG R-peak segmentation algorithm. Adapted from BioSPPy. Follows the approach by Hamilton [Hami02]_.

    Parameters
    ---------- 
    signal : array-like, shape (#points,)
        Filtered ECG signal to extract R peaks from.
    fs : int
        Sampling frequency.

    Returns
    -------
    rpeaks : array-like
        R-peak location indices.

    References
    ----------
    .. [Hami02] P.S. Hamilton, "Open Source ECG Analysis Software Documentation", E.P.Limited, 2002
    '''
    # segmentation
    (rpeaks,) = bp.signals.ecg.hamilton_segmenter(
        signal=signal, sampling_rate=fs)

    # correct R-peak locations
    (rpeaks,) = bp.signals.ecg.correct_rpeaks(
        signal=signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.05
    )
    hr = fs * (60.0 / np.diff(rpeaks))
    return np.mean(hr)


def get_peaks_onsets(array):
    ''' Get extrema from signal. Each row is treated independently.

    Parameters
    ---------- 
    array : np.ndarray, shape (#segments, #points)
        The input array, where each row is an independent segment.

    Returns
    -------    
    onsets_indx : np.ndarray
        Indices of the onset points (minima) in the signal. Output of np.argwhere().
    peaks_indx : np.ndarray
        Indices of the peak points (maxima) in the signal. Output of np.argwhere().
    '''
    onsets_indx = _find_extrema_indx(array, mode='min')
    peaks_indx = _find_extrema_indx(array, mode='max')
    onsets_indx, peaks_indx = _remove_unwanted_extrema(onsets_indx, peaks_indx)
    return onsets_indx, peaks_indx


def correct_rpeaks(signal, peaks, fs, amp_tol=0.15):
    """
    Corrects the detected R-peaks in an ECG signal based on amplitude tolerance.

    Parameters
    ---------- 
    signal : np.ndarray, shape (#points,)
        The ECG signal.
    peaks : array-like
        Indices of the detected R-peaks in the signal.
    fs : int 
        Sampling frequency of the signal.
    amp_tol : float, defaults to 0.15
        Amplitude tolerance for peak correction.

     Returns
    -------  
    corrected_peaks_indx : numpy.ndarray, shape (#peaks,)
        Indices of the corrected R-peaks.
    """
    bpm40_npeaks = int((len(signal)/fs * 40) / 60)

    if len(peaks) < bpm40_npeaks:
        corrected_peaks_indx = peaks
    else:
        bpm40_thr = np.partition(
            signal[peaks], - bpm40_npeaks)[-bpm40_npeaks] * (1-amp_tol)
        corrected_peaks_indx = peaks[signal[peaks] >= bpm40_thr]
    return corrected_peaks_indx


# import plotly.graph_objects as go
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=signal))
# fig.add_trace(go.Scatter(x=peaks, y=signal[peaks], mode='markers'))
# fig.add_trace(go.Scatter(x=corrected_peaks_indx, y=signal[corrected_peaks_indx], mode='markers'))
# fig.show()


def _find_extrema_indx(array=None, mode="both"):
    """Locate local extrema points in a signal, returning an array of the indices of the extrema, shape (N, array.ndim), where N is the number of extrema. Adapted from BioSSPy. Based on Fermat's Theorem."""

    if mode not in ["max", "min", "both"]:
        raise ValueError("Unknwon mode %r." % mode)

    aux = np.diff(np.sign(np.diff(array, axis=1)), axis=1)

    if mode == "both":
        aux = np.abs(aux)
        inflection_points = aux > 0
    elif mode == "max":
        inflection_points = aux < 0
    elif mode == "min":
        inflection_points = aux > 0

    extrema = np.zeros_like(array, dtype=bool)
    extrema[:, 1:-1] = inflection_points[:, :]
    extrema = np.argwhere(extrema)

    if len(extrema) == 0:
        raise RuntimeError('No extrema found in any of the samples.')

    return extrema


def _remove_unwanted_extrema(onsets_indx, peaks_indx):
    """Internal method that received a set of extrema indices (corresponding to an array with shape (#samples, #points in sample)) and removes the first peaks if they are not preceeded by an onset and removes the last onsets if they are not followed by a peak."""

    if len(onsets_indx[:, 0]) == 0 or len(peaks_indx[:, 0]) == 0:
        raise RuntimeError('No extrema found in any of the samples.')

    # remove first peak if before onset and last onset if after peak
    onsets_indx_by_sample, peaks_indx_by_sample = [], []
    for sample_id in np.unique(np.append(onsets_indx[:, 0], peaks_indx[:, 0])):
        try:
            remove_first_peak = peaks_indx[peaks_indx[:, 0] ==
                                           sample_id][0][1] < onsets_indx[onsets_indx[:, 0] == sample_id][0][1]
            remove_last_onset = onsets_indx[onsets_indx[:, 0] ==
                                            sample_id][-1][1] > peaks_indx[peaks_indx[:, 0] == sample_id][-1][1]
        except IndexError:
            continue
        peaks_indx_by_sample += [peaks_indx[peaks_indx[:, 0]
                                            == sample_id][remove_first_peak:]]
        onsets_indx_by_sample += [onsets_indx[onsets_indx[:, 0] == sample_id][:(
            len(onsets_indx[onsets_indx[:, 0] == sample_id]) - remove_last_onset)]]

    onsets_indx = np.concatenate(onsets_indx_by_sample)
    peaks_indx = np.concatenate(peaks_indx_by_sample)

    return onsets_indx, peaks_indx
