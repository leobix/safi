######################
#### Dependencies ####
######################
def make_prediction(measurement,forecast):
    # Loop over pred period and save predictions
    result = pd.DataFrame(columns={})
    for i in range(1,49):
        x_test = proc.prepare_x_test(measurement, forecast, 48, int(i))
        result_raw= dict()
        result_raw['f_period'] = i
        result_raw['forecast_time'] = x_test['forecast_time'].iloc[0]
        result_raw['present_time'] = x_test['present_time'].iloc[0]

        for feature in ['speed','cos_wind_dir','sin_wind_dir']:
            x_to_predict = x_test[xgb_models[feature + str(i)].get_booster().feature_names].copy()
            result_raw[feature] = xgb_models[feature + str(i)].predict(x_to_predict)[0]

        result = result.append(result_raw, ignore_index=True)

    result['wind_dir'] =result.apply(lambda row : proc.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']),axis=1)
    return result

def add_measurements_and_forecast(prediction, measurement,forecast):
    result = prediction.rename(columns={'forecast_time':'datetime', 
                                        'speed': 'pred_speed',
                                        'wind_dir' : 'pred_wind_dir'})
    
    # Compare to numtech forecast
    forecast['wind_dir'] =forecast.apply(lambda row : proc.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']),axis=1)
    result = result.merge(forecast[['f_date','speed','wind_dir']] \
                          .rename(columns={'f_date' : 'datetime', 
                                           'speed' : 'numtech_speed',
                                           'wind_dir' : 'numtech_wind_dir'}),
                          how='left')
    return result


import sys
import pickle
import pandas as pd
sys.path.append('../')
from utils import data_process as proc,forecast_ingestion

#####################
#### Load models ####
#####################
xgb_models = dict()
for i in range(1,49):
    for feature in ['speed','cos_wind_dir','sin_wind_dir']:
        xgb_models[feature + str(i)] = pickle.load(open('../trained_models_26072020/' + feature + '_t_' + str(i), 'rb'))


#################################
#### Predict and save result ####
#################################
# measurement
all_measurement = pd.read_csv('../data/processed/last_measurement.csv')
all_measurement['datetime']= pd.to_datetime(all_measurement['datetime'],format='%Y-%m-%d %H:%M:%S')
all_measurement['wind_dir'] =all_measurement.apply(lambda row : proc.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']),axis=1)

# forecast
forecast = pd.read_csv('../data/processed/last_forecast.csv')
forecast['f_date']= pd.to_datetime(forecast['f_date'],format='%Y-%m-%d %H:%M:%S')

# Make prediction and save results
prediction = make_prediction(all_measurement,forecast)

# Compare results to measurements and forecast 
result = add_measurements_and_forecast(prediction,all_measurement,forecast)

# log result
result.to_csv('../data/results/result_' + str(result.present_time[0])[0:13] + '.csv',index=False)
# save for ui
result.to_csv('../data/processed/last_result.csv',index=False)