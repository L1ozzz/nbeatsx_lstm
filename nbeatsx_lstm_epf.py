__all__ = ['EPF']
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
if not os.path.exists('./results/'):
    os.makedirs('./results/')

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


@dataclass
class EPFInfo:
    groups: Tuple[str] = ('Local',)


class EPF:


    @staticmethod
    def normalize_data(df, save_path='scaler.joblib'):
        scaler = StandardScaler()
        numerical_cols = ['GustDir', 'GustSpd', 'WindRun', 'Rain', 'Tmean', 'Tmax', 'Tmin', 'Tgmin', 'VapPress', 'ET10', 'Rad', 'SoilM']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        dump(scaler, save_path)  # 保存scaler对象
        return df, scaler

    @staticmethod
    def load_scaler(path='scaler.joblib'):
        return load(path)

    @staticmethod
    def remove_outliers(df):
        # 使用Z-score方法标识和删除异常值
        numerical_cols = ['GustDir', 'GustSpd', 'WindRun', 'Rain', 'Tmean', 'Tmax', 'Tmin', 'Tgmin', 'VapPress', 'ET10', 'Rad','SoilM']
        df[numerical_cols] = df[numerical_cols].apply(lambda x: x[(np.abs(stats.zscore(x)) < 3)])
        return df.dropna()

    @staticmethod
    def load(directory: str, filename: str ) -> Tuple[pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]]:
        """
        Loads local EPF data.

        Parameters
        ----------
        directory: str
            Directory where data is located.

        Returns
        -------
        Tuple of DataFrames (Y, X, S)
        """
        file = os.path.join(directory,filename)
        #file = "data/updated_processed_weather_data.csv" # 假设数据文件名为 local_data.csv
        '''
        df = pd.read_csv(file)
        # 假设你的数据列名如图所示，进行相应处理
        df.columns = ['ds', 'GustDir', 'GustSpd', 'WindRun', 'Rain', 'Tmean', 'Tmax', 'Tmin', 'Tgmin', 'VapPress',
                      'ET10', 'Rad', 'SoilM', 'Season']
        df = EPF.remove_outliers(df)
        df, scaler = EPF.normalize_data(df)  # 接收返回的两个值
        df['unique_id'] = 'Local'
        df['ds'] = pd.to_datetime(df['ds'])
        df['week_day'] = df['ds'].dt.dayofweek

        dummies = pd.get_dummies(df['week_day'], prefix='day')
        df = pd.concat([df, dummies], axis=1)

        dummies_cols = [col for col in df if col.startswith('day')]

        Y = df.filter(items=['unique_id', 'ds', 'SoilM']).rename(columns={'SoilM': 'y'})
        X = df.filter(
            items=['unique_id', 'ds', 'GustDir', 'GustSpd', 'WindRun', 'Rain', 'Tmean', 'Tmax', 'Tmin', 'Tgmin',
                   'VapPress', 'ET10', 'Rad', 'week_day'] + dummies_cols)

        # 静态数据集只包含 'unique_id' 和 'Season'，并进行适当的处理
        S = df[['unique_id', 'Season']].drop_duplicates().reset_index(drop=True)
        season_dummies = pd.get_dummies(S['Season'], prefix='season')
        S = pd.concat([S[['unique_id']], season_dummies], axis=1)

        return Y, X, S
        '''

        df = pd.read_csv(file)
        # 处理异常值和标准化

        # 假设你的数据列名如图所示，进行相应处理
        df.columns = ['ds', 'GustDir', 'GustSpd', 'WindRun', 'Rain', 'Tmean', 'Tmax', 'Tmin', 'Tgmin', 'VapPress',
                      'ET10', 'Rad', 'SoilM', 'Season']
        #df = EPF.remove_outliers(df)
        #df = EPF.normalize_data(df)
        #maxdf = np.max(df)
        #mindf = np.min(df)
        #df = (df - mindf)/(maxdf-mindf)
        df['unique_id'] = 'Local'
        df['ds'] = pd.to_datetime(df['ds'])
        df['week_day'] = df['ds'].dt.dayofweek


        dummies = pd.get_dummies(df['Season'], prefix='season')
        df = pd.concat([df, dummies], axis=1)
        #print(df.head)

        dummies_cols = [col for col in df if col.startswith('season')]
        #print(dummies_cols)

        Y = df.filter(items=['unique_id', 'ds', 'SoilM']).rename(columns={'SoilM': 'y'})
        X = df.filter(
            items=['unique_id', 'ds', 'GustDir', 'GustSpd', 'WindRun', 'Rain', 'Tmean', 'Tmax', 'Tmin', 'Tgmin',
                   'VapPress', 'ET10', 'Rad','week_day'] + dummies_cols)

        # 静态数据集只包含 'unique_id' 和 'Season'，并进行适当的处理
        S = df.filter(items=['unique_id'])
        #S = df[['unique_id', 'Season']].drop_duplicates().reset_index(drop=True)
        #season_dummies = pd.get_dummies(S['Season'], prefix='season')
        #S = pd.concat([S[['unique_id']], season_dummies], axis=1)

        #print(Y.head())
        #print(X.head())
        #print(S.head())

        return Y, X, S


    @staticmethod
    def load_groups(directory: str, groups: List[str] = None) -> Tuple[pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]]:
        """
        Loads panel of EPF data according to groups.
        Here groups are not used as we have local single dataset.

        Parameters
        ----------
        directory: str
            Directory where data is located.
        groups: List[str]
            Not used, present for compatibility.

        Returns
        -------
        Tuple of DataFrames (Y, X, S)
        """
        return EPF.load(directory)

    @staticmethod
    def download(directory: str) -> None:
        """Downloads EPF Dataset. Not needed for local data."""
        pass  # 不需要下载
