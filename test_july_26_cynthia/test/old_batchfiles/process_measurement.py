def get_season(month):
    if month in [12,1,2]:
        return 1
    if month in [3,4,5]:
        return 2
    if month in [6,7,8]:
        return 3
    if month in [9,10,11]:
        return 4

def get_am(hour):
    if hour in range(0,12):
        return 1
    else:
        return 0

import pandas as pd
from datetime import timedelta

# Load data 
measurement = pd.read_csv('../data/last_measurement.csv')
measurement['datetime'] = pd.to_datetime(measurement['datetime'],format='%Y-%m-%d %H:%M:%S')
measurement['datetime'] = measurement['datetime'] + timedelta(hours=1)
# Drop na
measurement = measurement.set_index('datetime') \
              [['speed','wind_dir','temp', 'radiation', 'precip','cos_wind_dir','sin_wind_dir']] \
              .dropna(axis=0, how='all')

# Convert to numeric:
c_list = ['wind_dir','speed','temp','radiation','precip']
for c in c_list:
    measurement[c]= pd.to_numeric(measurement[c], errors= 'coerce')

# init output measurement data
measurement_out = pd.DataFrame()
# speed weighted hourly mean for sin & cos
measurement_out['cos_wind_dir'] = (measurement['cos_wind_dir'] * measurement['speed']).resample('H').sum() \
                                                   / measurement['speed'].resample('H').sum()
measurement_out['sin_wind_dir'] = (measurement['sin_wind_dir'] * measurement['speed']).resample('H').sum() \
                                                   / measurement['speed'].resample('H').sum()

# hourly mean for speed, temperature, radiation and precipitation
for col in ['speed','temp','radiation','precip','wind_dir']:
    measurement_out[col] = measurement[col].resample('1H').mean()
    
# add caterogical features
measurement_out['season'] = measurement_out.index.month.map(get_season) # ordinal not categorical for linear models
measurement_out['am'] = measurement_out.index.hour.map(get_am)
measurement_out.drop(columns=['am'],inplace=True)
measurement_out.to_csv('../data/processed_measurement.csv')



