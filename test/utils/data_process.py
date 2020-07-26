import pandas as pd
import numpy as np
from datetime import timedelta
from utils import data_preparation as prep

import warnings
warnings.filterwarnings('ignore')


"""
Function to prepare data for models:
parameters:
    measurement: measurement df (see data_preparation.py)
    forecast: forecast df (see data_preparation.py)
    past_n_steps: past # of hours of data up to present day
    pred_period: prediction period # of hours
    param: prediction parameter, e.g. 'speed'
output:
    x_df, y_df: df format x and y
    x, y: array format (no datetime columns)"""

def prepare_x_y(measurement, forecast, past_n_steps, pred_period, param='speed'):

    # concatenate past_n_steps data
    df1=get_past_n_steps(measurement, past_n_steps)

    # add forecast data
    x_df = join_forecast(df1, forecast, pred_period)

    #add smoothed day, hour features
    x_df = smooth_day_hour(x_df)

    # define y accordingly
    index = x_df['forecast_time']
    df2 = measurement[param]
    y_df = pd.merge(df2, index, left_on = 'datetime', right_on='forecast_time' )

    #fillna: use last measurements
    x_df.fillna(method='ffill', inplace=True)
    y_df.fillna(method='ffill', inplace=True)

    #dropna
    x_df.dropna(axis=1, inplace=True)
    y_df.dropna(axis=1, inplace=True)

    #select intersection of the forecast times:
    index2= y_df['forecast_time']
    x_df = pd.merge(x_df, index2, left_on = 'forecast_time', right_on='forecast_time')

    #set index:
    x_df.reset_index(inplace=True, drop=True)

    #drop timestamps for model training
    x, y = drop_timestamps(x_df, y_df)
    return x_df, y_df, x, y


"""
Below are the sub functions
"""
# data_merge, data, forecast = prepare_data_with_forecast(data, keep_only_last=False)
def get_past_n_steps(df, steps_in):
    #rename column to most remote data
    df_out = df.copy().add_suffix('_t-'+str(steps_in))
    #t-i remote data
    for i in range(1, steps_in+1):
        df_temp = df.copy().add_suffix('_t-'+str(steps_in-i)) #rename column
        df_temp= df_temp.shift(periods=-i, axis=0) #shift down i row
        df_out=df_out.join(df_temp, how = 'inner')#join
    #shift index to present time (+steps_in)
    df_out['present_time']=df_out.index.to_series()+timedelta(hours=steps_in)
    df_out.set_index(pd.DatetimeIndex(df_out['present_time']), inplace=True)
    return df_out

def join_forecast(df, forecast, predict):
    #crop out under 7 hour forecasts due to lag
    forecast = forecast.loc[forecast['f_period']>=7]
    #crop out forecast if forecast period is less than prediction period
    forecast = forecast.loc[forecast['f_period']>=predict]
    forecast = prep.keep_last_forecast(forecast)
    forecast = forecast.add_suffix('_forecast')

    #calculate forecast_time
    df['forecast_time'] = df['present_time']+ timedelta(hours=predict)

    df_out = pd.merge(df, forecast, how = 'left', left_on = 'forecast_time', right_on ='f_date')
    df_out.fillna(value=0, inplace=True) #fill missing forecasts as 0
    return df_out

def smooth_day_hour(df):
    df['cos_day'] = np.cos(2 * np.pi * df['present_time'].dt.day / 365)
    df['cos_hour'] =  np.cos(2 * np.pi * df['present_time'].dt.hour / 24)
    return df

#change x,y to array like
def drop_timestamps(x_df, y_df):
    #drop timestamp columns
    x= x_df.drop(['present_time','forecast_time'], axis=1)
    y= y_df.drop(['forecast_time'], axis=1)
    return x, y
