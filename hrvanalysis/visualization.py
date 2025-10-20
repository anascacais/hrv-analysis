# third-party
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
import pandas as pd
import numpy as np
from IPython.display import display, clear_output


COLOR_PALETTE = ['#4179A0', '#A0415D', '#44546A',
                 '#44AA97', '#FFC000', '#0F3970', '#873C26']


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])


def plot_channels(data, channels, datetime_as_index=True, use_display=False):
    ''' Plots the signals corresponding to the channel namels in "channels".

    Parameters
    ----------
    data : mne.io.Raw or pd.DataFrame
        Instance of mne.io.Raw or dataframe containing the columns "time" and the ones in "channels".
    channels : list
        List containing the names of the channels to plot. 
    datetime_as_index : bool, defaults to True
        Whether to use datetime as x-axis or not, in which case a timedelta (in seconds) will be used.
    segment = int, optional
        Duration of segment (in seconds) to use. A random start index will be used.
    Returns
    -------
    None
    '''
    if isinstance(data, pd.DataFrame):
        data_df = data.copy()
    else:
        data_df = data.copy().to_data_frame(picks=channels)

    fig = go.Figure()

    # Plot data channels
    for i, ch in enumerate(channels):
        if datetime_as_index:
            axis_label = 'Time (H:M:S)'
            t = pd.to_timedelta(data_df.time, unit='s') + \
                pd.Timestamp(2024, 1, 1)
        else:
            axis_label = 'Time (s)'
            t = data_df.time

        fig.add_trace(go.Scatter(
            x=t, y=data_df[ch],
            line=dict(width=3, color=COLOR_PALETTE[i]),
            name=ch
        ))

    # Config plot layout
    fig.update_yaxes(
        gridcolor='lightgrey',
        title='Amplitude (mV)'
    )
    fig.update_xaxes(
        gridcolor='lightgrey',
        title=axis_label,
        tickformat="%H:%M:%S",
    )
    fig.update_layout(
        title='ECG data',
        showlegend=True,
        plot_bgcolor='white',
    )
    if use_display:
        fig.show(mode='inline')
        display(fig)
    else:
        fig.show(mode='inline')


def plot_quality(data_df, channel, quality_channel, datetime_as_index=True):
    ''' Plots the signals corresponding to the channel namels in "channels".

    Parameters
    ----------
    data_df : pd.DataFrame
        Dataframe containing the columns "time" and "channel".
    channel : str
        Name of column with signal to plot.
    quality_channels : list
        List containing the names of the channels with the qualty indexes.
    datetime_as_index : bool, defaults to True
        Whether to use datetime as x-axis or not, in which case a timedelta (in seconds) will be used.

    Returns
    -------
    None
    '''
    quality2color = {0: COLOR_PALETTE[1],
                     1/3: COLOR_PALETTE[4], 0.5: COLOR_PALETTE[4], 2/3: COLOR_PALETTE[0], 1: COLOR_PALETTE[3]}

    fig = go.Figure()

    if datetime_as_index:
        axis_label = 'Time (H:M:S)'
        t = pd.to_timedelta(data_df.time, unit='s') + \
            pd.Timestamp(2024, 1, 1)
    else:
        axis_label = 'Time (s)'
        t = data_df.time

    # Plot quality
    signal = data_df[channel]
    edges = data_df[quality_channel][data_df[quality_channel].diff() != 0]

    if len(edges) == 1:
        fig.add_trace(go.Scatter(
            x=t[0:len(t)],
            y=signal.values,
            mode='lines',
            line=dict(
                width=3, color=quality2color[data_df[quality_channel].values[0]]),
            name=quality_channel,
            legendgroup=quality_channel,
            showlegend=True,
        ))

    for j in range(len(edges) - 1):  # Loop through all pairs of edges
        fig.add_trace(go.Scatter(
            x=t[edges.index[j]: edges.index[j + 1]],
            y=signal.values[edges.index[j]: edges.index[j + 1]],
            mode='lines',
            line=dict(
                width=3, color=quality2color[edges.values[j]]),
            name=quality_channel,
            legendgroup=quality_channel,
            showlegend=(j == 0),
        ))

    fig.add_trace(go.Scatter(
        x=t[edges.index[-1]:len(t)],
        y=signal.values[edges.index[-1]:len(t)],
        mode='lines',
        line=dict(
            width=3, color=quality2color[edges.values[-1]]),
        name=quality_channel,
        legendgroup=quality_channel,
        showlegend=(j == 0),
    ))

    # Config plot layout
    fig.update_yaxes(
        gridcolor='lightgrey',
        title='Amplitude (mV)'
    )
    fig.update_xaxes(
        gridcolor='lightgrey',
        title=axis_label,
        tickformat="%H:%M:%S",
    )
    fig.update_layout(
        title='ECG SQI',
        showlegend=True,
        plot_bgcolor='white',
    )
    fig.show(mode='inline')


def plot_quality_subplots(data_df, channel, quality_channels, joint_quality_channel, datetime_as_index=True):

    fig = make_subplots(rows=len(quality_channels)+1,
                        cols=1, subplot_titles=quality_channels+[joint_quality_channel])

    if datetime_as_index:
        axis_label = 'Time (H:M:S)'
        t = pd.to_timedelta(data_df.time, unit='s') + \
            pd.Timestamp(2024, 1, 1)
    else:
        axis_label = 'Time (s)'
        t = data_df.time

    for i, chn in enumerate(quality_channels+[joint_quality_channel]):
        # Plot quality
        signal = data_df[channel]
        edges = data_df[chn][data_df[chn].diff() != 0]

        for j in range(len(edges) - 1):  # Loop through all pairs of edges
            fig.add_trace(go.Scatter(
                x=t[edges.index[j]: edges.index[j + 1]],
                y=signal.values[edges.index[j]: edges.index[j + 1]],
                mode='lines',
                line=dict(
                    width=3, color=_assign_color2quality(edges.values[j])),
                name=chn,
                legendgroup=chn,
                showlegend=(j == 0),
            ), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            x=t[edges.index[-1]:len(t)],
            y=signal.values[edges.index[-1]:len(t)],
            mode='lines',
            line=dict(
                width=3, color=_assign_color2quality(data_df[chn].values[-1])),
            name=chn,
            legendgroup=chn,
            showlegend=True,
        ), row=i+1, col=1)

    # Config plot layout
    fig.update_yaxes(
        gridcolor='lightgrey',
        title='Amplitude (mV)'
    )
    fig.update_xaxes(
        gridcolor='lightgrey',
        title=axis_label,
        tickformat="%H:%M:%S",
    )
    fig.update_layout(
        title='ECG SQI',
        showlegend=False,
        plot_bgcolor='white',
        height=800,
        width=1000,
    )
    fig.show(mode='inline')


def _assign_color2quality(sqi):
    ''''''
    if sqi <= 1/3:
        return COLOR_PALETTE[1]
    elif sqi <= 2/3:
        return COLOR_PALETTE[4]
    else:
        return COLOR_PALETTE[3]


def plot_quality_prev(data_df, channel, quality_channels, datetime_as_index=True):
    ''' Plots the signals corresponding to the channel namels in "channels".

    Parameters
    ----------
    data_df : pd.DataFrame
        Dataframe containing the columns "time" and "channel".
    channel : str
        Name of column with signal to plot.
    quality_channels : list
        List containing the names of the channels with the qualty indexes.
    datetime_as_index : bool, defaults to True
        Whether to use datetime as x-axis or not, in which case a timedelta (in seconds) will be used.

    Returns
    -------
    None
    '''

    fig = FigureResampler(go.Figure())

    if datetime_as_index:
        axis_label = 'Time (H:M:S)'
        t = pd.to_timedelta(data_df.time, unit='s') + \
            pd.Timestamp(2024, 1, 1)
    else:
        axis_label = 'Time (s)'
        t = data_df.time

    fig.add_trace(go.Scatter(
        x=t, y=data_df[channel],
        line=dict(width=3, color=COLOR_PALETTE[2]),
        name=channel
    ))

    signal = data_df[channel].values
    mean = np.mean(signal)
    step = np.std(signal)*0.5

    # Plot quality
    for i, ch in enumerate(quality_channels):
        edges = data_df[ch][data_df[ch].diff() != 0]

        if len(edges) == 1:
            fig.add_trace(go.Scatter(
                x=[t[0], t[len(t)-1]],
                y=[mean+(-1+i)*step, mean+(-1+i)*step],
                fill='toself',
                mode='lines',
                line=dict(width=10, color=COLOR_PALETTE[3:5]
                          [int(not data_df[ch].values[0])]),
                name=ch,
                legendgroup=ch,
                showlegend=True,
            ))

        for j in range(len(edges) - 1):  # Loop through all pairs of edges
            fig.add_trace(go.Scatter(
                x=[t[edges.index[j]], t[edges.index[j + 1]]],
                y=[mean+(-1+i)*step, mean+(-1+i)*step],
                fill='toself',
                mode='lines',
                line=dict(
                    width=10, color=COLOR_PALETTE[3:5][int(not edges.values[j])]),
                name=ch,
                legendgroup=ch,
                showlegend=(j == 0),
            ))

    # Config plot layout
    fig.update_yaxes(
        gridcolor='lightgrey',
        title='Amplitude (mV)'
    )
    fig.update_xaxes(
        gridcolor='lightgrey',
        title=axis_label,
        tickformat="%H:%M:%S",
    )
    fig.update_layout(
        title='ECG SSQI',
        showlegend=True,
        plot_bgcolor='white',
    )
    fig.show(mode='inline')
