######################
#### Dependencies ####
######################
import os
import datetime
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from utils.util_functions import *
from subprocess import Popen, PIPE

main_dir = '//SA-MODAT-MTO-PR/Data-Safi/'
#main_dir = '../Sa-modat-mpo-pr/Data-Safi/'

p = Popen("last_file_gp2.bat", shell=True, stdout=PIPE,cwd='../utils/')
stdout, stderr = p.communicate()
file_path = stdout.decode('utf-8').rstrip()

data = pd.read_csv(main_dir + file_path,low_memory=False,
               delimiter='\t',quotechar='"',decimal=',').dropna().tail(24*60 *120)

# rename columns
data = data.rename(columns={'Unnamed: 0' : 'datetime',
                            'Speed@1m': 'speed', 
                            'Dir': 'wind_dir',
                            'AirTemp' : 'temp',
                            "Rad'n" : 'radiation',
                            'Rain@1m' : 'precip',
                            'Speed@5m': 'speed', 
                            'Rain@5m' : 'precip'})



# convert str to float
for col in ['wind_dir','speed','temp','precip']:
    data[col] = data[col].map(comma_to_float)

# replace #-INF by 0
data.loc[data['radiation'] == '#-INF', 'radiation'] = 0
data.loc[data['radiation'] == '#+INF', 'radiation'] = 0
# select columns
data = data[['datetime','speed','wind_dir', 'temp', 'radiation', 'precip']]

#######################################################
##### TEMPORARY DUE TO WEATHER STATION MAINTENANCE ####
#######################################################
data['temp'] = 22
#######################################################

measurement = data.reset_index(drop=True)

# Date format
measurement['datetime'] = pd.to_datetime(measurement['datetime'],format='%d/%m/%Y %H:%M:%S')

# Skip incomplete hours
measurement['Id_hour'] = measurement['datetime'].map(lambda x : str(x)[0:13])
measurement = measurement.merge(measurement.groupby(['Id_hour'])['datetime'].count().reset_index() \
                                .rename(columns={'datetime':'Id_hour_count'}),
                                how='left')
measurement = measurement.loc[measurement['Id_hour_count'] >= 40,].reset_index(drop=True)

# Drop na
measurement = measurement.set_index('datetime') \
              [['speed','temp', 'radiation', 'precip','wind_dir']] \
              .dropna(axis=0, how='all')

# Smooth wind direction 
measurement['cos_wind_dir'] = np.cos(2 * np.pi * measurement['wind_dir'] / 360)
measurement['sin_wind_dir'] = np.sin(2 * np.pi * measurement['wind_dir'] / 360)


# Init output measurement data
measurement_out = pd.DataFrame()
# Speed weighted hourly mean for sin & cos
measurement_out['cos_wind_dir'] = (measurement['cos_wind_dir'] * measurement['speed']).resample('H', label='right').sum() \
                                                   / measurement['speed'].resample('H', label='right').sum()
# Speed weighted hourly mean for sin & cos
measurement_out['sin_wind_dir'] = (measurement['sin_wind_dir'] * measurement['speed']).resample('H', label='right').sum() \
                                                   / measurement['speed'].resample('H', label='right').sum()


# Hourly mean for speed, temperature, radiation and precipitation
for col in ['speed','temp','radiation','precip']:
    measurement_out[col] = measurement[col].map(float).resample('1H', label='right').mean()

# Add caterogical features
measurement_out['season'] = measurement_out.index.month.map(get_season) # ordinal not categorical for linear models

measurement_out = measurement_out.reset_index()
# Select columns
measurement_out = measurement_out[['datetime','speed','cos_wind_dir','sin_wind_dir','temp','radiation','precip','season']]

# Build date Index and fill na
Idx_Measurement = pd.DataFrame(pd.date_range(measurement_out.datetime[0],
                                             measurement_out.datetime.iloc[-1],
                                             freq='H'),
                                             columns=['datetime'])

measurement_out = Idx_Measurement.merge(measurement_out,how='left').fillna(method='ffill')

# Save file
measurement_out.to_csv('../data/processed/last_measurement.csv',index=False)

