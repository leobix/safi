######################
#### Dependencies ####
######################
import os
import pandas as pd
import numpy as np
import datetime 

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

def get_one_measurement(main_dir,file_path,DAY):
    data = pd.read_csv(main_dir + file_path,low_memory=False,
                   delimiter='\t',quotechar='"',decimal=',').dropna()
    # drop first row (contains unit)
    data = data.loc[1:,:]
    # rename columns
    data = data.rename(columns={'Unnamed: 0' : 'datetime',
                                'Speed@1m': 'speed', 
                                'Dir': 'wind_dir',
                                'AirTemp' : 'temp',
                                "Rad'n" : 'radiation',
                                'Rain@1m' : 'precip',
                                'Speed@5m': 'speed', 
                                'Rain@5m' : 'precip'})
    # convert date from gmt to gmt+1
    data['datetime'] = pd.to_datetime(data['datetime'],format= '%d/%m/%Y %H:%M:%S') + datetime.timedelta(hours=1)
    
    # creation and modification time
    data['ctime'] = datetime.datetime.fromtimestamp(os.path.getctime(main_dir+ file_path)).strftime(format ='%Y-%m-%d %H:%M:%S')
    data['mtime'] = datetime.datetime.fromtimestamp(os.path.getmtime(main_dir+ file_path)).strftime(format ='%Y-%m-%d %H:%M:%S')

    # files can contain also old data so we filter on day to avoid duplicates and slow processing
    data['day'] = data['datetime'].map(lambda x : str(x)[0:10])
    data = data.loc[data['day'] == DAY_format(DAY)]

    # convert str to float
    for col in ['wind_dir','speed','temp','precip']:
        data[col] = data[col].map(comma_to_float)

    # replace #-INF by 0
    data.loc[data['radiation'] == '#-INF', 'radiation'] = 0
    data.loc[data['radiation'] == '#+INF', 'radiation'] = 0
    # select columns
    data = data[['datetime','speed','wind_dir', 'temp', 'radiation', 'precip','ctime','mtime']]
    return data.reset_index(drop=True)

def get_one_day_measurement(DAY1,DAY2,main_dir):
    data_per_day = pd.DataFrame()
    for file_path in os.listdir(main_dir):
        if (file_path[4:12] == DAY1):
            data_per_day = data_per_day.append(get_one_measurement(main_dir,file_path,DAY1))
        if (file_path[4:12] == DAY2):
            data_per_day = data_per_day.append(get_one_measurement(main_dir,file_path,DAY2))
    return data_per_day.drop_duplicates().reset_index(drop=True)