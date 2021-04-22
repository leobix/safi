######################
#### Dependencies ####
######################
def get_season(month):
    if month in [12,1,2]:
        return 1
    if month in [3,4,5]:
        return 2
    if month in [6,7,8]:
        return 3
    if month in [9,10,11]:
        return 4

def smooth_wind_dir(df):
    df['cos_wind_dir'] = np.cos(2 * np.pi * df['wind_dir'] / 360)
    df['sin_wind_dir'] = np.sin(2 * np.pi * df['wind_dir'] / 360)
    #print('smooth wind direction')
    df.drop(columns=['wind_dir'], inplace=True)
    return df

import os
import sys
import datetime
import pandas as pd
import numpy as np

##############################
#### Init path and months ####
##############################
main_dir = '../data/raw/measurements/'
sys_date = sys.argv[1]
sys_year = sys_date[-2:]
sys_date = datetime.datetime.strptime(sys_date,'%d/%m/%Y')
sys_month = str(sys_date.month).zfill(2)
previous_month = str((sys_date - datetime.timedelta(sys_date.day)).month).zfill(2)

######################################
#### Gather 2 last months of data ####
######################################
data = pd.DataFrame()
for file_path in os.listdir(main_dir):
    if file_path[-9:] in [y + '-' + sys_year + '.csv' \
                          for y in (sys_month,previous_month)]:
        try:
            data = data.append(pd.read_csv(main_dir + file_path))
        except:
            print('failed : ', file_path)
            
data.sort_values(by=['datetime','mtime'], inplace=True)
data.drop_duplicates(subset = 'datetime', keep = 'last', inplace=True)

##########################
#### Process and save ####
##########################
measurement = data.reset_index(drop=True)

# Date format
measurement['datetime'] = pd.to_datetime(measurement['datetime'],format='%Y-%m-%d %H:%M:%S')

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
measurement = smooth_wind_dir(measurement)

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
    measurement_out[col] = measurement[col].resample('1H', label='right').mean()

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
