import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#import functions
import data_process as proc
import data_preperation as prep
from utils_scenario import *



def run_regression(steps_in, steps_out):
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
    pred = get_all_scenarios(predict['speed'], predict['cos_wind_dir'],predict['sin_wind_dir'], b_scenarios=True)
    true = get_all_scenarios(true['speed'], true['cos_wind_dir'],true['sin_wind_dir'], b_scenarios=True)
    base = get_all_scenarios(baseline['speed'], baseline['cos_wind_dir'],baseline['sin_wind_dir'], b_scenarios=True)

    #calculate prediction accuracies
    pred_score = accuracy_score(pred, true).round(3)
    base_score = accuracy_score(base, true).round(3)

    return  pred_score, base_score

def binary_accuracy(predict, true, baseline):
    pred = get_all_dangerous_scenarios(predict['speed'], predict['cos_wind_dir'],predict['sin_wind_dir'])
    true = get_all_dangerous_scenarios(true['speed'], true['cos_wind_dir'],true['sin_wind_dir'])
    base = get_all_dangerous_scenarios(baseline['speed'], baseline['cos_wind_dir'],baseline['sin_wind_dir'])

    #calculate prediction accuracies
    pred_score = accuracy_score(pred, true).round(3)
    base_score = accuracy_score(base, true).round(3)
    return  pred_score, base_score



if __name__ == "__main__":
    print("Executing as main program")
    print("Value of __name__ is: ", __name__)

    #get data
    measurement=prep.prepare_measurement()
    forecast = prep.prepare_forecast()

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={'past_n_steps','pred_n_steps','pred_scenario','pred_binary','base_scenario','base_binary'})
    # pred_speed=pd.DataFrame(columns={})
    # pred_cos=pd.DataFrame(columns={})
    # pred_sin=pd.DataFrame(columns={})

    #prediction steps
    t_list=[1,3,6,9,12,15,18,24,30,36,42,48]

    for t in t_list:
        print(t)
        #run model
        predict, true, base = run_regression(steps_in=48, steps_out=t)

        #calculate accuracy
        pred_scenario, base_scenario  = scenario_accuracy(predict, true, base)
        pred_binary, base_binary  = binary_accuracy(predict, true, base)

        #record accuracy
        accuracy = accuracy.append({'past_n_steps': 48,
                                          'pred_n_steps': t,
                                          'pred_scenario': pred_scenario,
                                          'base_scenario': base_scenario,
                                          'pred_binary':pred_binary,
                                          'base_binary':base_binary}, ignore_index=True)
        #record prediction
        # pred_speed = pd.concat([pred_speed, predict['speed'].add_suffix('_t+'+str(t))])
        # pred_cos = pd.concat([pred_cos, predict['cos_wind_dir'].add_suffix('_t+'+str(t))])
        # pred_sin = pd.concat([pred_sin, predict['sin_wind_dir'].add_suffix('_t+'+str(t))])

    #output results df
    accuracy.to_csv('results/xgboost_accuracy.csv', index=False)
