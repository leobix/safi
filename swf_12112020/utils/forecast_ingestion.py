######################
#### Dependencies ####
######################
import sys
sys.path.append('../')

import os 
import numpy as np
import pandas as pd
import datetime
import pickle


from utils import data_preparation
from utils import data_preparation as prep


def get_one_forecast(main_dir,DAY,HOUR):
    forecast_dir_path = main_dir + DAY + '_' + HOUR + '/'
    # Init output
    data = pd.DataFrame()
    # Init f_period
    period_root = 0
    # Read 2 meteo file
    for dir_day in os.listdir(forecast_dir_path):
        if len(dir_day) == 10:
            # load file
            data_per_day = pd.read_csv(forecast_dir_path + dir_day + '/meteo.txt',delimiter=";")
            data_per_day['date'] = dir_day.replace('_','-')
            data_per_day['f_period'] = period_root + data_per_day['heure']
            # append to output
            data = data.append(data_per_day)
            # add 1 day to fperiod
            period_root += 24

    # forecast date
    data['f_date'] = data['date'] + ' ' 
    data['f_date'] += data['heure'].map(lambda x : str(x).zfill(2))
    data['f_date'] += ':00:00'
    # present date
    data['p_date'] = DAY.replace('_','-') + ' ' + HOUR + ':00:00'

    # rename columns 
    data = data.rename(columns={'vitesse' : 'speed', 
                                'temperature' : 'temp', 
                                'rayonnement' : 'rad',
                                'direction' : 'wind_dir'})

    # compute cos and sin
    data = prep.smooth_wind_dir(data)

    # select columns
    data = data[['p_date','f_date','f_period','speed','temp','rad','precip','cos_wind_dir','sin_wind_dir']]
    return data


def get_forecast(main_dir,
                 limit_date=pd.to_datetime(datetime.datetime.now())):
    
    start_date=pd.to_datetime(limit_date - datetime.timedelta(days=60))
    sub_dir = os.listdir(main_dir)
    # Concat files
    forecast = pd.DataFrame()
    for file_path in sub_dir:
        DAY = file_path[0:10]
        HOUR = file_path[11:13]
        file_path_date = pd.to_datetime(DAY.replace('_','-') + ' ' + HOUR + ':00:00')
        file_path_date += datetime.timedelta(hours=8)
        if (start_date <= file_path_date) & (file_path_date <= limit_date):
            forecast = forecast.append(get_one_forecast(main_dir,DAY,HOUR))
    forecast = forecast.reset_index(drop=True)

    # Change to dt 
    forecast['f_date']= pd.to_datetime(forecast['f_date'],format='%Y-%m-%d %H:%M:%S')
    forecast['p_date']= pd.to_datetime(forecast['p_date'],format='%Y-%m-%d %H:%M:%S')
    # Crop out <=7 hours 
    # in fact we need to crop 8 hours 
    forecast = forecast.loc[forecast['f_period']>=7]

    # Keep last forecast
    forecast= data_preparation.keep_last_forecast(forecast)
    forecast.reset_index(inplace=True)
    
    #read pickle format columns 
    forecast_cols = pickle.load(open('../utils/forecast_cols.pkl', 'rb'))
    assert((forecast_cols == forecast.columns).any())

    return forecast