import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#import local functions
import data_process as proc
import data_preperation as prep
import utils_scenario as utils

def run_xgb(steps_in, steps_out):
    #Parameter list:
    param_list =['speed','cos_wind_dir','sin_wind_dir']

    predict = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})
    true = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})
    baseline = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})

    for param in param_list:
        x_df, y_df, x, y = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, param)
        X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, shuffle = False)
        xg = XGBRegressor(max_depth = 5)
        xg.fit(X_train, y_train)
        y_baseline = x_df
        y_hat = xg.predict(X_test)

        predict[param] = pd.Series(y_hat)
        true[param] = pd.Series(y_test.flatten())
        baseline[param] = x_df[param+'_forecast'][-len(y_hat):]

    #reset index
    baseline.reset_index(inplace=True)
    return predict, true, baseline

def scenario_accuracy(predict, true, baseline):
    pred = utils.get_all_scenarios(predict['speed'], predict['cos_wind_dir'],predict['sin_wind_dir'], b_scenarios=True)
    true = utils.get_all_scenarios(true['speed'], true['cos_wind_dir'],true['sin_wind_dir'], b_scenarios=True)
    base = utils.get_all_scenarios(baseline['speed'], baseline['cos_wind_dir'],baseline['sin_wind_dir'], b_scenarios=True)

    #calculate prediction accuracies
    pred_score = metrics.accuracy_score(pred, true).round(3)
    base_score = metrics.accuracy_score(base, true).round(3)

    return  pred_score, base_score

def binary_accuracy(predict, true, baseline):
    pred = utils.get_all_dangerous_scenarios(predict['speed'], predict['cos_wind_dir'],predict['sin_wind_dir'])
    true = utils.get_all_dangerous_scenarios(true['speed'], true['cos_wind_dir'],true['sin_wind_dir'])
    base = utils.get_all_dangerous_scenarios(baseline['speed'], baseline['cos_wind_dir'],baseline['sin_wind_dir'])

    #calculate prediction accuracies
    pred_score = metrics.accuracy_score(pred, true).round(3)
    base_score = metrics.accuracy_score(base, true).round(3)
    #calculate auc
    pred_auc = metrics.roc_auc_score(pred, true).round(3)
    base_auc = metrics.roc_auc_score(base, true).round(3)
    return  pred_score, base_score, pred_auc, base_auc





if __name__ == "__main__":
    print("Executing as main program")
    print("Value of __name__ is: ", __name__)

    #get data
    measurement=prep.prepare_measurement()
    forecast = prep.prepare_forecast()
    #keep useful columns
    measurement= measurement[['speed', 'cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation', 'precip','d_speed_max','d_temp_max', 'season', 'am']]

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={})
    pred_speed=pd.DataFrame(columns={})
    pred_angle=pd.DataFrame(columns={})

    #prediction steps
    t_list=[1,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]
    steps_in=48


    for t in t_list:
        #run model
        predict, true, base = run_xgb(steps_in, steps_out=t)

        #calculate accuracy & auc
        pred_scenario, base_scenario  = scenario_accuracy(predict, true, base)
        pred_bin_accu, base_bin_accu, pred_bin_auc, base_bin_auc= binary_accuracy(predict, true, base)

        #record accuracy
        accuracy = accuracy.append({'past_n_steps': str(steps_in),
                                          'pred_n_steps': str(t),
                                          'xgb_scenario_accu': pred_scenario,
                                          'base_scenario_accu': base_scenario,
                                          'xbg_binary_accu':pred_bin_accu,
                                          'base_binary_accu':base_bin_accu,
                                            'xbg_binary_auc':pred_bin_auc,
                                          'base_binary_auc':base_bin_auc}, ignore_index=True)
        #record predicted speed
        pred_speed = pd.concat([pred_speed, predict['speed'].rename('speed_t+'+str(t))], axis=1)
        #record predicted angle
        temp = pd.concat([predict['cos_wind_dir'], predict['sin_wind_dir']], axis=1)
        temp['angle'] = temp.apply(lambda row : utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']), axis = 1)
        pred_angle = pd.concat([pred_angle, temp['angle'].rename('angle_t+'+str(t))], axis=1)



    #output results df
    accuracy.to_csv('results/xgboost_accuracy.csv', index=False)
    pred_angle.to_csv('results/xgboost_pred_angle.csv', index=False)
    pred_speed.to_csv('results/xgboost_pred_speed.csv', index=False)