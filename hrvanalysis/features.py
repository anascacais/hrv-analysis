# third-party
import numpy as np
import pandas as pd
import biosppy as bp


def get_hrv_features(segments, ts, fs, amp_tol=0.15):
    ''' Compute HRV features for each segment. Returns formatted DataFrame.

    Parameters
    ---------- 
    segments : array-like, shape (#segments, #points)
        Filtered ECG segments. Segments should have at least 60s duration.
    ts : array-like, shape (#segments, )
        Startime of corresponding segment within acquisition file (in seconds).
    fs : int
        Sampling frequency.
    amp_tol : float, defaults to 0.15
        Amplitude tolerance for peak correction.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ["timedelta", "name", "value"], where each entry corresponds to a single feature extracted from a single segment. Lenght is equal to #segments * #HRV features.
    '''

    df = pd.DataFrame(columns=['timedelta', 'name', 'value'])

    for i in range(segments.shape[0]):
        (segment_peaks,) = bp.signals.ecg.hamilton_segmenter(
            segments[i, :], sampling_rate=fs)

        (rpeaks,) = bp.signals.ecg.correct_rpeaks(
            segments[i, :], segment_peaks, sampling_rate=fs)

        hrv_features = bp.signals.hrv.hrv(
            rpeaks=rpeaks, sampling_rate=fs, parameters='auto', outliers=None, show=False)

        hrv_features = dict(hrv_features)

        for key, n_key in zip(['rr_mean', 'sdnn', 'rmssd', 'pnn50', 'lf_pwr', 'lf_hf', 'hf_pwr'], ['rr_mean (ms)', 'sdnn (ms)', 'rmssd (ms)', 'pnn50 (%)', 'lf_pwr (ms^2)', 'lf_hf (n.u.)', 'hf_pwr (ms^2)']):
            hrv_features[n_key] = hrv_features.pop(key)

        for feat in hrv_features.keys():
            df.loc[len(df)] = [pd.to_timedelta(
                ts[i], unit='s'), feat, hrv_features[feat]]

    return df


# def get_twa_features(segments, fs, amp_tol=0.15):
#     df = pd.DataFrame(columns=['name', 'value'])

#     _, peaks_indx = get_peaks_onsets(segments)

#     for i in range(segments.shape[0]):
#         segment_peaks = peaks_indx[peaks_indx[:, 0] == i][:, 1]

#         # correct R-peak locations using the the amplitude of the top peaks (corresponding to 40 BPM) as reference and giving a 15% tolerance
#         rpeaks = correct_rpeaks(
#             segments[i, :], segment_peaks, fs, amp_tol=amp_tol)

#         rr = np.diff(rpeaks) / fs
#         t_wave_onsets = np.rint((((rpeaks / fs) * 1000 + 40 + 1.33 * (
#             np.append(rr, rr[-1]) * 1000)**(1/2)) / 1000) * fs).astype(np.int64)
#         t_wave_offsets = t_wave_onsets + \
#             np.rint((160 / 1000) * fs).astype(np.int64)
#         t_wave_offsets = t_wave_offsets[t_wave_offsets < len(segments[i, :])]
#         pass

#     return df
