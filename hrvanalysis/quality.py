import biosppy as bp
import pandas as pd
import numpy as np
import warnings


def extract_sqi(signal, indexes, fs, win_duration=5, binary=True):
    ''' Extract signal quality indexes from ECG signal. SNR is computed according [Rahman2022]_.

    Parameters
    ---------- 
    signal : array-like, shape (#points,)
        Filtered ECG signal to extract R peaks from.
    indexes : list<str>
        Names of features to extract. Should be chosen from: "kurtosis", "skewness", "SNR", "peak_amplitude", "ratio_flat_ECG", and "HR".
    fs : int
        Sampling frequency.
    win_duration : int, defaults to 60 
        Duration of segment to analyse (in seconds).

    Returns
    -------
    sqi : np.ndarray, shape (#points, #indexes), dtype=bool
        Array with quality indexes. Values are either 0 (bad quality) or 1 (good quality), according to the specified SQI. An index is provided for each point in the signal, despite being computed for segments of duration "win_duration". 

    References
    ----------
    .. [Rahman2022] Rahman Saifur, Karmakar Chandan, Natgunanathan Iynkaran, Yearwood John and Palaniswami Marimuthu 2022. Robustness of electrocardiogram signal quality indicesJ. R. Soc. Interface.1920220012 http://doi.org/10.1098/rsif.2022.0012
    '''

    signal = np.array(signal)
    sqi = np.zeros((len(indexes), len(signal)))

    # Segment signal into segments with duration "win_duration" and no overlap
    win_samples = int(fs * win_duration)
    split_indx = range(win_samples, len(signal), win_samples)
    segments = np.split(signal, split_indx)

    # Handle last segment if not divisible
    last_segment = np.reshape(segments[-1], (1, -1))
    segments = np.vstack(segments[:-1])

    peak_metrics = _compute_peak_metrics(segments, fs)
    peak_metrics_last = _compute_peak_metrics(last_segment, fs)

    for i, index in enumerate(indexes):
        if index == 'peak_amplitude':
            new_sqi = peak_metrics[:, 0].copy()
            new_sqi_last = peak_metrics_last[:, 0].copy()
        elif index == 'HR':
            new_sqi = peak_metrics[:, 1].copy()
            new_sqi_last = peak_metrics_last[:, 1].copy()
        elif index == 'template_corr':
            new_sqi = peak_metrics[:, 2].copy()
            new_sqi_last = peak_metrics_last[:, 2].copy()
        else:
            raise ValueError(
                f'{index} is not a valid quality index. Choose between "peak_amplitude", "template_corr", and "HR".')

        if binary:
            new_sqi = quantify_quality(new_sqi, index)
            new_sqi_last = quantify_quality(new_sqi_last, index)

        sqi[i, :] = np.hstack([np.repeat(new_sqi, win_samples), np.repeat(
            new_sqi_last, last_segment.shape[1])])

    return sqi.T


def quantify_quality(quality, metric):
    ''' Applies logic to quantify each metric value corresponding to each segment as either 0 (bad quality) or 1 (good quality).

    Parameters
    ---------- 
    quality : array-like, shape (#segments,)
        Array with quality indice for each segment.
    metric : str
        Name of quality metric. Can be "kurtosis", "skewness", "SNR", "peak_amplitude", "ratio_flat_ECG", and "HR"..

    Returns
    -------
    sqi : np.ndarray, shape (#segments,), dtype=bool
        Array with quality indexes for each segment. Values are either 0 (bad quality) or 1 (good quality), according to the specified SQI.
    '''
    quality = np.array(quality)
    sqi = np.zeros((len(quality),), dtype=bool)

    if metric == 'HR':
        sqi[np.where(np.logical_and((quality >= 40), quality <= 200))] = 1
    elif metric == 'peak_amplitude':
        sqi[np.where(quality <= 10)] = 1
    elif metric == 'template_corr':
        sqi[np.where(quality > 0.7)] = 1
    else:
        raise ValueError(
            f'{metric} is not a valid quality index. Choose between "peak_amplitude", "template_corr", and "HR".')

    return sqi


def apply_quality_logic(sqi, fs, win_duration=5, nb_blocks_smooth=5):
    ''' Applies fuzzy logic on quality indexes. Result is the average SQI score. Smooth sample-level SQI array using 25s windows composed of 5 constant 5s blocks.


    Parameters
    ---------- 
    sqi : np.ndarray, shape (#points, #indexes)
        SQI, where values are either 0 (bad quality) or 1 (good quality).

    Returns
    -------
    np.ndarray, shape (#points,)
        Result of logic applied. Values are either 0 (bad quality) or 1 (good quality). An index is provided for each point in the signal.
    '''
    avg_sqi = np.sum(sqi, axis=1) / sqi.shape[1]

    # Collapse into one SQI value per "win_duration"-second block (since all samples in block are equal)
    win_samples = int(fs * win_duration)
    block_sqi = avg_sqi[::win_samples]

    blocks = np.split(block_sqi, range(
        nb_blocks_smooth, len(block_sqi), nb_blocks_smooth))
    blocks_last = blocks[-1]
    blocks = np.vstack(blocks[:-1])

    blocks_smooth = blocks.copy()

    blocks_surrounding = np.delete(blocks, int((nb_blocks_smooth-1)/2), axis=1)
    mask = np.all(blocks_surrounding == blocks_surrounding[:, [0]], axis=1)
    blocks_smooth[mask] = blocks[mask, 0][:, np.newaxis]

    # Expand back to sample-level
    smoothed_sqi = np.hstack(
        [np.repeat(blocks_smooth, win_samples), np.repeat(blocks_last, win_samples)])
    if len(smoothed_sqi) > len(avg_sqi):
        smoothed_sqi = smoothed_sqi[:-(len(smoothed_sqi)-len(avg_sqi))]
    return smoothed_sqi


def get_quality_summary_prev(sqi, fs):
    segments = (sqi != sqi.shift()).cumsum()
    result = sqi.groupby(segments).agg(Value=('first'), Duration=('size'))

    total_duration = result['Duration'].sum() / fs
    good_sqi_total_duration = result[result['Value']
                                     == 1]['Duration'].sum() / fs
    good_sqi_median_duration = np.median(
        result[result['Value'] == 1]['Duration']) / fs
    medium_sqi_median_duration = np.median(
        result[np.logical_or(result['Value'] == 1/3, result['Value'] == 0.5, result['Value'] == 2/3)]['Duration']) / fs
    bad_sqi_median_duration = np.median(
        result[result['Value'] == 0]['Duration']) / fs

    return pd.to_timedelta([
        total_duration,
        good_sqi_total_duration,
        good_sqi_median_duration,
        medium_sqi_median_duration,
        bad_sqi_median_duration
    ], unit='s')


def get_quality_summary(sqi, fs):
    segments = (sqi != sqi.shift()).cumsum()
    result = sqi.groupby(segments).agg(Value=('first'), Duration=('size'))

    total_duration = result['Duration'].sum() / fs
    good_sqi_total_duration = result[result['Value']
                                     == 1]['Duration'].sum() / fs
    good_sqi_median_duration = np.median(
        result[result['Value'] == 1]['Duration']) / fs
    medium_sqi_median_duration = np.median(
        result[np.logical_or(result['Value'] == 1/3, result['Value'] == 0.5, result['Value'] == 2/3)]['Duration']) / fs
    bad_sqi_median_duration = np.median(
        result[result['Value'] == 0]['Duration']) / fs

    return pd.to_timedelta([
        total_duration,
        good_sqi_total_duration,
        good_sqi_median_duration,
        medium_sqi_median_duration,
        bad_sqi_median_duration
    ], unit='s')


def _compute_hr(rpeaks, fs):
    return np.mean(fs * (60.0 / np.diff(rpeaks)))


def _compute_peak_metrics(signal, fs):
    '''Internal method that computes the mean amplitude of the R peaks, the ratio of flat ECG (second dimension), and the mean HR (third dimension). Output shape (#segments, 3)'''
    rpeaks_mean_amp, mean_hr, template_corr = [], [], []
    for i in range(signal.shape[0]):
        # Compute HR metrics
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in scalar divide")
            (segment_peaks,) = bp.signals.ecg.hamilton_segmenter(
                signal[i, :], sampling_rate=fs)
            (rpeaks,) = bp.signals.ecg.correct_rpeaks(
                signal[i, :], segment_peaks, sampling_rate=fs)

        if len(rpeaks) == 0:
            print(
                f'No R-peaks detected in this segment. Returning NaN for peak metrics.')
            rpeaks_mean_amp += [np.nan]
            mean_hr += [np.nan]
            template_corr += [np.nan]

        else:
            templates, _ = bp.signals.ecg.extract_heartbeats(
                signal=signal[i, :], rpeaks=rpeaks, sampling_rate=fs, before=0.2, after=0.4)
            corr_points = np.corrcoef(templates)

            rpeaks_mean_amp += [np.mean(signal[i, rpeaks])]
            mean_hr += [_compute_hr(rpeaks, fs)]
            template_corr += [np.mean(corr_points)]

        # COLOR_PALETTE = ['#4179A0', '#A0415D', '#44546A',
        #                  '#44AA97', '#FFC000', '#0F3970', '#873C26']
        # from plotly import graph_objects as go
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #     x=pd.to_timedelta(
        #         np.arange(0, len(signal[i, :])/fs, 1/fs), unit='s') + pd.Timestamp(2024, 1, 1),
        #     y=signal[i, :],
        #     line=dict(width=3, color=COLOR_PALETTE[0])
        # ))
        # fig.add_trace(go.Scatter(
        #     x=(pd.to_timedelta(np.arange(0, len(signal[i, :])/fs, 1/fs), unit='s')+pd.Timestamp(
        #         2024, 1, 1))[rpeaks],
        #     y=signal[i, rpeaks],
        #     mode='markers',
        #     marker=dict(color=COLOR_PALETTE[2])
        # ))
        # fig.show()

    rpeaks_mean_amp = np.array(rpeaks_mean_amp)
    mean_hr = np.array(mean_hr)
    mean_template_corr = np.array(template_corr)

    return np.column_stack((rpeaks_mean_amp, mean_hr, mean_template_corr))


def filter_hq_segments(signal, sqi, fs, win_duration):
    ''' Filters only high-quality sections of a signal and segments it into segments with duration "win_duration". Non-contiguous sections are not put together into the same segment. Incomplete segments are removed.

    Parameters
    ---------- 
    signal : array-like, shape (#points,)
        Filtered ECG signal
    sqi : array-like, shape (#points,)
        Array with quality indexes. Values are either 0 (bad quality) or 1 (good quality), according to the specified SQI.
    win_duration : int
        Duration of segments into which the signal will be split (in seconds).

    Returns
    -------
    segments : np.ndarray, shape=(#segments, #points), dtype=float
        High quality ECG segments.
    ts : np.ndarray, shape=(#segments, ), dtype=float
        Startime of corresponding segment within acquisition file (in seconds).
    '''

    signal = np.array(signal)

    signal_hq_df = pd.Series(np.array(signal))[sqi == 1]
    hq_segments = np.split(np.array(signal_hq_df.values),
                           np.where(signal_hq_df.index.diff() != 1)[0][1:])
    hq_ts = np.split(np.linspace(0, len(signal) / fs, len(signal))[sqi == 1],
                     np.where(signal_hq_df.index.diff() != 1)[0][1:])

    segments, ts = [], []
    win_samples = int(fs * win_duration)

    for hq_segment, hq_ts_singular in zip(hq_segments, hq_ts):
        # Segment signal into segments with duration "win_duration" and no overlap
        split_indx = range(win_samples, len(hq_segment), win_samples)
        temp_segments = np.split(hq_segment, split_indx)
        temp_ts = np.split(hq_ts_singular, split_indx)

        if len(temp_segments[-1]) != win_samples:
            temp_segments = temp_segments[:-1]
            temp_ts = temp_ts[:-1]

        segments += temp_segments
        ts += temp_ts

    if len(segments) == 0:
        return np.array([]), np.array([])

    segments = np.vstack(segments)
    ts = np.vstack(ts)

    return segments, ts[:, 0]
