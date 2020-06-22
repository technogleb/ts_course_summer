import matplotlib.pyplot as plt
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout
from plotly.offline import download_plotlyjs
py.init_notebook_mode(connected=True)


def plot_ts(ts):
    """Plots stand-alone time-series"""
    ts.dropna(inplace=True)
    trace = Scatter(x=list(ts.index),
                    y=ts.values, name='time-series')
    data = [trace]
    layout = Layout(title=ts.name)
    fig = dict(data=data, layout=layout)
    py.iplot(fig)


def plot_multiple_ts(*args):
    """
    Plots multiple time-series on one plot.
    arguments: any time-series
    """
    data = [Scatter(x=list(ts.index), y=ts.values, name=ts.name)
            for ts in args]
    fig = dict(data=data)
    py.iplot(fig)


def plot_decomposition(res, show_season=False):
    """
    Plots output from stl decomposition (_detect_ts or _detect_feature),
    which is [anoms, ts, trend, season].
    """
    anoms, ts, trend, season = (res[i] for i in range(4))
    trace_ts = Scatter(x=list(ts.index),
                       y=ts.values, name='time_series')
    trace_tr = Scatter(x=list(trend.index),
                       y=trend.values, name='estimate')
    trace_se = Scatter(x=list(season.index),
                       y=season.values, name='season')
    trace_an = Scatter(x=list(anoms.index),
                       y=anoms.values,
                       mode='markers',
                       name='anomalies',
                       marker={'color': 'red'})
    data = [trace_ts, trace_tr, trace_an, trace_se]
    if not show_season:
        data.pop()
    layout = Layout(title=res[1].name)
    fig = dict(data=data, layout=layout)
    py.iplot(fig)
