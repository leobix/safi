######################
#### Dependencies ####
######################
def get_forecast_data(DAY,HOUR):
    # Define folder
    forecast_dir_path = main_dir + DAY + '_' + HOUR + '_optimise_previ/'
    # Init output
    data = pd.DataFrame()
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
    # compute cos and sin
    data['cos_wind_dir'] = np.cos(2 * np.pi * data['direction'] / 360)
    data['sin_wind_dir'] = np.sin(2 * np.pi * data['direction'] / 360)
    # rename columns 
    data = data.rename(columns={'vitesse' : 'speed', 
                                'temperature' : 'temp', 
                                'rayonnement' : 'rad',
                                'direction' : 'angle'})
    
    data = data[['p_date','f_date','f_period',
                 'speed','angle',
                 'temp','rad',
                 'cos_wind_dir','sin_wind_dir']]
    
    return data.reset_index(drop=True)        

import os 
import pandas as pd
import numpy as np
import datetime

################################
#### Define path and months ####
################################
main_dir = '//Sa-modat-cs-pr/plumair/SAFI/Sorties/2020/'
sub_dir = [x for x in os.listdir(main_dir) if (x[-8:] == 'se_previ') and (x[5:7] in ('06','07','08','09'))]

##############################
#### Read and concat data ####
##############################
fetched_data = pd.DataFrame()
for file_path in sub_dir:
    data = get_forecast_data(file_path[0:10],file_path[11:13])
    data['f_day'] = file_path[0:10]
    data['f_hour'] = file_path[11:13]
    fetched_data = fetched_data.append(data)
fetched_data = fetched_data.reset_index(drop=True)

###################
#### Save data ####
###################
file_suffix = str(datetime.datetime.today())[0:19]
file_suffix = file_suffix.replace('-','_')
file_suffix = file_suffix.replace(':','_')
file_suffix = file_suffix.replace(' ','__')
fetched_data.to_csv('../data/forecast/last_forecast__' + file_suffix + '.csv',index=False)
fetched_data.to_csv('../data/last_forecast.csv',index=False)