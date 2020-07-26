### Dependencies

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
    
def prepare_x_test(measurement, forecast, past_n_steps, pred_period):
    
    #concat past n steps from measurement 
    df = measurement.set_index('datetime')
    df.fillna(method='ffill')
    df=proc.get_past_n_steps(df, past_n_steps)

    #calculate forecast_time
    df['forecast_time'] = df['present_time']+timedelta(hours=pred_period)

    #join forecast according to forecast time 
    forecast = forecast.set_index('f_date') 
    forecast = forecast.add_suffix('_forecast')
    df = pd.merge(df, forecast, how = 'left', left_on = 'forecast_time', right_on ='f_date')
    #add cos day
    df = proc.smooth_day_hour(df)
    #fill missing forecasts as 0
    df.fillna(value=0, inplace=True) 
    #keep first row 
    df = df.head(1)
    #drop timestamp columns
    df_out = df.drop(['present_time','forecast_time'], axis=1)
    return df_out

# test_df = prepare_x_test(measurement, forecast, past_steps, predict )

    
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import pickle

from utils import utils_scenario as utils, data_preparation as prep, data_process as proc

### Load data 
# Load forecast data
# Load last 48 hours measurement data

# forecast data
forecast = pd.read_csv('./data/last_forecast.csv')
forecast = forecast.sort_values(by=['p_date','f_date'])
forecast['p_date'] = pd.to_datetime(forecast['p_date'],format='%Y-%m-%d %H:%M:%S')
forecast['f_date'] = pd.to_datetime(forecast['f_date'],format='%Y-%m-%d %H:%M:%S')
forecast.drop(columns=['f_day','f_hour','p_date','f_period'],inplace=True)
forecast = forecast.loc[len(forecast)-48:,].reset_index(drop=True)

# measurement data
all_measurement = pd.read_csv('./data/processed_measurement.csv')
all_measurement['datetime'] = pd.to_datetime(all_measurement['datetime'],format='%Y-%m-%d %H:%M:%S')
all_measurement = all_measurement.sort_values(by='datetime')
measurement = all_measurement.loc[len(all_measurement)-48:,].reset_index(drop=True)
measurement.drop(columns='wind_dir',inplace=True)

date_to_predict = measurement.datetime.max()

### Model
# Prepare data 
# Make prediction 
# Join with next 48h measurement 
# Save input and result 

result  = pd.DataFrame(columns=['past_n_steps','pred_period','speed', 'cos_wind_dir','sin_wind_dir']) 
pred_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
past_n_steps = 48


for pred in pred_list: 
    #prepare data to be the same format as training data 
    x_test = prepare_x_test(measurement, forecast, past_n_steps, pred)
    x_test= np.array(x_test) #change to array 

    #read 3 models for speed, cos_wind, sin_wind
    xgb1= pickle.load(open('trained_models_04072020//speed_t_'+str(pred), 'rb'))
    xgb2 = pickle.load(open('trained_models_04072020//cos_wind_dir_t_'+str(pred), 'rb'))
    xgb3 = pickle.load(open('trained_models_04072020//sin_wind_dir_t_'+str(pred), 'rb'))

    #predict 
    speed = xgb1.predict(x_test)[0]
    cos_wind = xgb2.predict(x_test)[0]
    sin_wind = xgb3.predict(x_test)[0]

    #record accuracy
    result = result.append({'past_n_steps': str(past_n_steps),
                            'pred_period': str(pred),
                            'speed':round(speed,3),
                            'cos_wind_dir':cos_wind,
                            'sin_wind_dir':sin_wind}, ignore_index=True)    

#convert cos and sin to wind_dir:
result['wind_dir'] = result.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']),axis = 1)
result['datetime'] = date_to_predict + result['pred_period'].map(lambda x : timedelta(hours=int(x)))

result.to_csv('./data/last_result.csv',index=False)