######################
#### Dependencies ####
######################
def comma_to_float(x):
    try:
        return float(x.replace(',','.'))
    except:
        return np.nan
    
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
    
def DAY_format(DAY):
    return ('20' + DAY[-2:] + '-' + DAY[3:5] + '-' + DAY[0:2])
    
    
def get_measurement(main_dir,file_path):
    data = pd.read_csv(main_dir + file_path,low_memory=False,
                       delimiter='\t',quotechar='"',decimal=',').dropna()
    data = data.loc[1:,:]
    data = data.rename(columns={'Unnamed: 0' : 'datetime',
                                'Speed@1m': 'speed', 
                                'Dir': 'wind_dir',
                                'AirTemp' : 'temp',
                                "Rad'n" : 'radiation',
                                'Rain@1m' : 'precip'})

    data['datetime'] = pd.to_datetime(data['datetime'],format= '%d/%m/%Y %H:%M:%S')
    data['day'] = data['datetime'].map(lambda x : str(x)[0:10])
    
    data = data.loc[data['day'] == DAY_format(DAY)]

    data['wind_dir'] = data['wind_dir'].map(comma_to_float)
    data['speed'] = data['speed'].map(comma_to_float)

    data['cos_wind_dir'] = np.cos(2 * np.pi * data['wind_dir'] / 360)
    data['sin_wind_dir'] = np.sin(2 * np.pi * data['wind_dir'] / 360)

    # add caterogical features
    data['season'] = data['datetime'].map(lambda x : get_season(x.month)) # ordinal not categorical for linear models
    data['am'] = data['datetime'].map(lambda x : get_am(x.hour))
    data = data[['datetime','speed', 'wind_dir','cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation', 'precip','season', 'am']]
    return data

import os
import sys
import pandas as pd
import numpy as np
import datetime

################################
#### Define path and months ####
################################
#sys_date = sys.argv[1]
#sys_year = sys_date.split('/')[2][-2:]
#sys_month = sys_date.split('/')[1]
#sys_day = sys_date.split('/')[0]
#DAY = sys_day + '-' + sys_month + '-' + sys_year

DAY = '08-07-20'

if (os.path.exists('../data/measurement/' + DAY) == False):
    os.mkdir('../data/measurement/' + DAY)

main_dir = '//Sa-modat-cs-pr/plumair/SAFI/Entrees/2020/'

##############################
#### Read and concat data ####
##############################
all_data = pd.DataFrame()
for file_path in os.listdir(main_dir):
    if (file_path[0:12] == 'GP2 ' + DAY):
        data = get_measurement(main_dir,file_path)
        all_data = all_data.append(data)
        
all_data = all_data.drop_duplicates().reset_index(drop=True)

###################
#### Save data ####
###################
file_suffix = str(datetime.datetime.today())[0:19]
file_suffix = file_suffix.replace('-','_')
file_suffix = file_suffix.replace(':','_')
file_suffix = file_suffix.replace(' ','__')
all_data.to_csv('../data/measurement/' + DAY + '/measurement__' + file_suffix + '.csv',index=False)

####################################################
#### Concat mezsurement data and save last file ####
####################################################
main_dir = '../data/measurement/'
all_data = pd.DataFrame()
for sub_dir in os.listdir(main_dir):
    for file_path in os.listdir(main_dir + sub_dir +'/'):
        try:
            data = pd.read_csv(main_dir + sub_dir + '/' + file_path)
            all_data = all_data.append(data)
            del data
        except:
            pass
all_data = all_data.drop_duplicates().reset_index(drop=True)
all_data.to_csv('../data/last_measurement.csv',index=False)