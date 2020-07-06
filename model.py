from typing import Dict, Callable
import pandas as pd
import numpy as np
import isodate
from copy import deepcopy

from monthdelta import monthdelta


def get_timedelta_from_granularity(granularity: str):
    datetime_interval = isodate.parse_duration(granularity)
    if isinstance(datetime_interval, isodate.duration.Duration):
        years, months = datetime_interval.years, datetime_interval.months
        total_months = int(years * 12 + months)
        datetime_interval = monthdelta(months=total_months)
    return datetime_interval


class TimeSeriesPredictor:
    def __init__(
            self,
            granularity: str,
            num_lags: int,
            Model,
            mappers: Dict[str, Callable] = {},
            *args, **kwargs
    ):

        self.granularity = granularity
        self.num_lags = num_lags
        self.model = Model(*args, **kwargs)
        self.mappers = mappers
        self.fitted = False

    def transform_into_matrix(self, ts: pd.Series) -> pd.DataFrame:
        """
        Transforms time series into lags matrix to allow
        applying supervised learning algorithms

        Parameters
        ------------
        ts
            Time series to transform

        Returns
        --------
        lags_matrix
            Dataframe with transformed values
        """

        ts_values = ts.values
        data = {}
        for i in range(self.num_lags + 1):
            data[f'lag_{self.num_lags - i}'] = np.roll(ts_values, -i)

        lags_matrix = pd.DataFrame(data)[:-self.num_lags]
        lags_matrix.index = ts.index[self.num_lags:]

        return lags_matrix

    def enrich(
            self,
            lags_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds external features to time series

        Parameters
        ------------
        lags_matrix
            Pandas dataframe with transformed time-series values
        mappers
            Dictionary of functions to map each timestamp of lags matrix.
            Each function should take timestamp as the only positional parameter
            and return value of your additional feature for that timestamp
        """

        mappers = self.mappers
        for name, mapper in mappers.items():
            feature = pd.Series(lags_matrix.index.map(mapper), lags_matrix.index, name=name)
            lags_matrix[name] = feature

        return lags_matrix

    def fit(self, ts: pd.Series, *args, **kwargs):
        lag_matrix = self.transform_into_matrix(ts)
        feature_matrix = self.enrich(lag_matrix)

        X, y = feature_matrix.drop('lag_0', axis=1), feature_matrix['lag_0']
        self.model.fit(X, y, *args, **kwargs)
        self.fitted = True

    def predict_next(self, ts_lags, n_steps=1):
        predict = {}

        ts = deepcopy(ts_lags)
        for _ in range(n_steps):
            next_row = self.generate_next_row(ts)
            next_timestamp = next_row.index[-1]
            value = self.model.predict(next_row)[0]
            predict[next_timestamp] = value
            ts[next_timestamp] = value
        return pd.Series(predict)

    def predict_batch(self, ts_batch: pd.Series):

        pass

    def generate_next_row(self, ts):
        """
        Takes time-series as an input and returns next row, that is fed to the fitted model,
        when predicting next value.

        Parameters
        ----------
        ts : pd.Series(values, timestamps)
            Time-series to detect on

        Returns
        ---------
        feature_matrix : pd.DataFrame
            Pandas dataframe, which contains feature lags of
            shape(1, num_lags+len(external_feautres))
        """

        delta = get_timedelta_from_granularity(self.granularity)
        next_timestamp = pd.to_datetime(ts.index[-1]) + delta
        lag_dict = {'lag_{}'.format(i): [ts[-i]] for i in range(1, self.num_lags + 1)}
        df = pd.DataFrame.from_dict(lag_dict)
        df.index = [next_timestamp]
        df = self.enrich(df)

        return df

    def set_params(self, **parameters):
        """
        Delayed
        """
        self_params = {'num_lags'}
        for parameter, value in parameters.items():
            if parameter in self_params:
                setattr(self,  parameter, value)
            else:
                setattr(self.model, parameter, value)
        pass

    def get_params(self):
        """
        Delayed
        """
        params = {
            'num_lags': self.num_lags
        }
        params.update(self.model.get_params())
        return params
