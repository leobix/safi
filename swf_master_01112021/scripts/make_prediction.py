####################
### Dependencies ###
####################

import os 
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from utils.util_functions import *
from datetime import timedelta

#################
### Load Data ###
#################

# Measurements
measurement_out = pd.read_csv('../data/processed/last_measurement.csv')
measurement_out['datetime'] = measurement_out['datetime'].map(lambda x : pd.to_datetime(x)) 
# Forecasts
forecast = pd.read_csv('../data/processed/last_forecast.csv')
forecast['f_date'] = forecast['f_date'].map(lambda x : pd.to_datetime(x))
forecast['p_date'] = forecast['p_date'].map(lambda x : pd.to_datetime(x))
forecast['file_creation_date'] = forecast['file_creation_date'].map(lambda x : pd.to_datetime(x))

#######################
### Data Processing ###
#######################

### Data Merge ###

# Save a copy of measurements to score results
Y_real = measurement_out.copy()

# 49 lag of measurements horizontal stack 
df_out = Y_real.add_suffix('_t-0')
for i in range(1, 49):
    df_temp = Y_real.copy().add_suffix('_t-'+str(i))
    df_out = pd.concat([df_out,df_temp.shift(i)],axis=1)
df_out = df_out.dropna(how='any')
#display(df_out.head(1))

# join measurements & forecast
df_joined = df_out.copy()
df_joined = df_joined.merge(forecast.add_suffix('_forecast'),
                 how='left',
                 left_on = 'datetime_t-0',
                 right_on='f_date_forecast')

# filter forecast files created after prediction time (same as crop out f_period > 7)
df_joined = df_joined.loc[df_joined['datetime_t-0'] >= df_joined['file_creation_date_forecast'],]


# Compute f_period
df_joined['f_period'] = df_joined[['datetime_t-0','p_date_forecast']] \
                         .apply(lambda row : get_f_period(row['datetime_t-0'],row['p_date_forecast']),axis=1)

# assert that file_creation_date_forecast is doing the job
assert((df_joined.f_period > 7).any()) 

# keep last forecast
df_joined = df_joined.groupby('datetime_t-0')['f_period'].min().reset_index() \
             .merge(df_joined,how='left')
    
# compute cos day and hour 
df_joined['cos_day'] = np.cos(2 * np.pi * df_joined['datetime_t-0'].dt.day / 365)
df_joined['cos_hour'] =  np.cos(2 * np.pi * df_joined['datetime_t-0'].dt.hour / 24)
#display(df_joined.head(1))


##############################
### New model adjustements ###
##############################

# Compute needed columns for updated models
df_joined['scenario_forecast'] = df_joined.apply(lambda row : get_int_scenario(row['speed_forecast'],
                                             row['cos_wind_dir_forecast'],
                                             row['sin_wind_dir_forecast']),
                  axis=1)

df_joined['dangerous_forecast'] = (df_joined['scenario_forecast'] > 3 ).map(int)

df_joined = df_joined.rename(columns={'f_period':'f_period_forecast'})

### Keep last row for predictions 
df_joined = df_joined.dropna()
df_joined = df_joined.tail(1).reset_index(drop=True)

###################################
### Make regression predictions ###
###################################

# Load needed columns for all models 
columns_names = list(pd.read_csv('../models_09072021/column_names.csv')['0'])

# Loop lists
model_names = ['xgb', 'dt','mlp','rf']
#model_names = ['dt']
features = ['speed','cos_wind_dir','sin_wind_dir']
pred_periods = ['1','2','3']



# Init regressions results
df_result = pd.DataFrame([df_joined['datetime_t-0'][0],
                          df_joined['datetime_t-0'][0],
                          df_joined['datetime_t-0'][0]],columns=['present_time'])
df_result['datetime'] = [df_joined['datetime_t-0'][0] + timedelta(hours=int(pred_period)) for pred_period in (1,2,3)]

forecast_for_results = forecast[['f_date','p_date','speed','cos_wind_dir','sin_wind_dir']].add_prefix('numtech_').copy()
# Compute f_period
forecast_for_results['f_period'] = forecast_for_results.apply(lambda row : get_f_period(row['numtech_f_date'],row['numtech_p_date']),axis=1)

df_result = df_result.merge(forecast_for_results,
                how='left',
                left_on='datetime',
                right_on='numtech_f_date')

df_result = df_result.loc[df_result.groupby("datetime")["f_period"].idxmin()]

df_result.drop(columns={'numtech_f_date','numtech_p_date','f_period'},inplace=True)


# Predict & save
models = dict()
for model_name in model_names:
    for feature in features:
        column_results = []
        for pred_period in pred_periods:
            x = '_'.join([model_name,feature,pred_period])
            # Load model
            models[x] = pickle.load(open('../models_09072021/trained_models/' + x + '.pkl','rb'))
            # Predict
            column_results += [models[x].predict(df_joined[columns_names])[0]]
        df_result[model_name + '_' + feature] = column_results 

# Compute wind dir, scenario and dangerous
for model_name in model_names + ['numtech']:
    df_result[model_name + '_wind_dir']= df_result.apply(
                                                lambda row : get_angle_in_degree(row[model_name + '_cos_wind_dir'],
                                                                                 row[model_name + '_sin_wind_dir']),
                                                axis=1
                                            )
    df_result[model_name + '_scenario'] = df_result.apply(
                                                lambda row : get_str_scenario(row[model_name + '_speed'],
                                                                              row[model_name + '_cos_wind_dir'],
                                                                              row[model_name + '_sin_wind_dir']),
                                                axis=1
                                            )
    df_result[model_name + '_binary'] = df_result[model_name + '_scenario'].map(get_str_binary)
    
df_result.to_csv('../data/processed/last_reg_results.csv',index=False)








