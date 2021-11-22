import os
import numpy as np
import pandas as pd

def get_one_forecast(DAY,HOUR,main_dir='../data/raw/forecasts/'):
    forecast_dir_path = main_dir + DAY + '_' + HOUR + '/'
    # Init output
    data = pd.DataFrame()
    # Read 2 meteo file
    for dir_day in os.listdir(forecast_dir_path):
        if len(dir_day) == 10:
            # load file
            data_per_day = pd.read_csv(forecast_dir_path + dir_day + '/meteo.txt',delimiter=";")
            data_per_day['date'] = dir_day.replace('_','-')
            # append to output
            data = data.append(data_per_day)
            # add 1 day to fperiod

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
    data['cos_wind_dir'] = np.cos(2 * np.pi * data['wind_dir'] / 360)
    data['sin_wind_dir'] = np.sin(2 * np.pi * data['wind_dir'] / 360)
    
    # Add scenario forecast legacy
    data = data.rename(columns={'scenario': 'scenario_legacy'})

    # select columns
    data = data[['p_date','f_date','speed','temp','rad','precip','cos_wind_dir','sin_wind_dir','wind_dir','scenario_legacy']]
    return data

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
      

def get_f_period(p_date,f_date):
    d = pd.to_datetime(p_date) - pd.to_datetime(f_date)
    return d.days * 24  + d.seconds // 3600

def get_angle_in_degree(cos, sin):
    #check if cos within reasonable range: 
    if (cos>=-1) & (cos <=1): 
        angle = 360 * np.arccos(cos) / (2*np.pi)
        if sin < 0:
            angle = 360 - angle
    #check if sin within reasonable range:       
    elif (sin>=-1) & (sin <=1):
        angle = 360 * np.arcsin(sin) / (2*np.pi)
        if cos < 0:
            angle = 180 - angle
        if angle < 0:
            angle += 360
    else:
        angle=0 
        #print('cos and sin out of range, returned 0')
    #because we care about the reverse angle for the scenarios
    return angle #(angle + 180) % 360

def is_favorable(cos, sin):
    # angle is between NO and E
    angle = get_angle_in_degree(cos, sin)
    # NO
    if angle > 303.75:
        return True
    # under E
    elif angle < 101.25:
        return True
    return False

def is_South(cos, sin):
    angle = get_angle_in_degree(cos, sin)
    # Direction S, SSO, SSE
    if 146.25 <= angle <= 213.75:
        return True
    return False


def is_S1(speed, cos, sin):
    if speed >= 4:
        return True
    elif speed > 1 and is_favorable(cos, sin):
        return True
    return False


def is_S2(speed, cos, sin):
    if 2 <= speed < 4 and not is_favorable(cos, sin):
        return True
    elif 0.5 <= speed <= 1 and is_favorable(cos, sin):
        return True
    return False


def is_S2b(speed, cos, sin):
    if 1 <= speed <= 2 and not is_favorable(cos, sin) and not is_South(cos, sin):
        return True
    return False


def is_S3(speed, cos, sin):
    if 0.5 <= speed < 1 and not is_favorable(cos, sin) and not is_South(cos, sin):
        return True
    elif speed < 0.5 and is_favorable(cos, sin):
        return True
    return False


def is_S3b(speed, cos, sin):
    if speed < 2 and is_South(cos, sin):
        return True
    return False


def is_S4(speed, cos, sin):
    if speed < 0.5 and not is_favorable(cos, sin):
        return True
    return False

def get_str_scenario(speed, cos, sin):
    if is_S1(speed, cos, sin):
        return 'S1'
    elif is_S2(speed, cos, sin):
        return 'S2'
    elif is_S2b(speed, cos, sin):
        return 'S2b'
    elif is_S3(speed, cos, sin):
        return 'S3'
    elif is_S3b(speed, cos, sin):
        return 'S3b'
    elif is_S4(speed, cos, sin):
        return 'S4'
    return ''
    
def get_int_scenario(speed, cos, sin):
    if is_S1(speed, cos, sin):
        return 1
    elif is_S2(speed, cos, sin):
        return 2
    elif is_S2b(speed, cos, sin):
        return 3
    elif is_S3(speed, cos, sin):
        return 4
    elif is_S3b(speed, cos, sin):
        return 5
    elif is_S4(speed, cos, sin):
        return 6
    return np.nan

def scenario_int_to_str(sc):
    if sc == 1:
        return 'S1'
    elif sc == 2:
        return 'S2'
    elif sc == 3:
        return 'S2b'
    elif sc == 4:
        return 'S3'
    elif sc == 5:
        return 'S3b'
    elif sc == 6:
        return 'S4'
    return ''
    
    
def get_str_binary(x):
    result = ''
    if x in ('S1','S2','S2b'):
        result = 'Safe'
    if x in ('S3','S3b','S4'):
        result = 'Dangerous'
    return result

def binary_int_to_str(x):
    result = ''
    if x == 0:
        result = 'Safe'
    if x == 1:
        result = 'Dangerous'
    return result

     

