# coding: utf-8
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram, detrend


def detect_ts(ts, significance_level=0.15, k=3,
              include_seasonality=False, period=None):
    """This function gets anomalies from stand-alone time-Series.

    Parameters
    -----------
    ts : pandas.Series with pd.Datetime index
        Univariate time_series.
    significance_level : float
        Percentage deviation. Below that level all
        anomalies will be ignored, even if detected.
    k : int
        Multiply coeficient of standart deviation used, defaults to 3.
    include_seasonality : bool, default=False
        If True, season component is extracted from time-series.
    period : int, default=None
        Number of points, used to define both smoothing window and seasonal
        period.

    Returns
    -----------
    (anomalies, time_series, smoothed_trend, seasonality) : tuple
    """

    period = period if period else get_season_period(ts)
    if not period:
        raise ValueError(
            'Couldn\'t automatically define season period'
            'and no period provided.'
        )

    k, b = np.polyfit(range(len(ts)), ts.values, 1)
    trend = pd.Series(k*np.array(range(len(ts))) + b, index=ts.index)
    ts_detrended = ts - trend
    season = ts_detrended.rolling(period).median()
    resid = ts - trend - season

    threshold = k * resid.std()
    indexes = np.where(abs(resid) > threshold)[0]
    anomalies = ts[indexes]

    width = max(ts) - min(ts)
    is_significant = np.where(
        (abs((anomalies - trend).dropna()) / width) > significance_level)[0]
    anomalies = anomalies[is_significant]

    return trend, season, resid


def get_season_period(ts):
    ts = pd.Series(detrend(ts), ts.index)
    f, Pxx = periodogram(ts)
    Pxx = list(map(lambda x: x.real, Pxx))
    ziped = list(zip(f, Pxx))
    ziped.sort(key=lambda x: x[1])
    highest_freqs = [x[0] for x in ziped[-100:]]
    season_periods = [round(1/(x+0.001)) for x in highest_freqs]
    for period in reversed(season_periods):
        if 4 < period < 100:
            return int(period)
