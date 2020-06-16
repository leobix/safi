import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

#import local functions
from utils import utils_scenario as utils, data_preparation as prep, data_process as proc

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=48,
                            help="number of in time steps")

parser.add_argument("--max-depth", type=int, default=5,
                            help="maximum depth of XGB")

parser.add_argument("--n_estimators", type=int, default=100,
                            help="number of estimators of XGB")

parser.add_argument("--t_list", type=list, default=[1,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48],
                            help="list of prediction time steps")

def run_xgb(steps_in, steps_out, max_depth = 5, n_estimators = 100):
    #Parameter list:
    param_list =['speed','cos_wind_dir','sin_wind_dir']

    predict = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})
    true = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})
    baseline = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})

    for param in param_list:
        x_df, y_df, x, y = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, param)
        X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, shuffle = False)
        xg = XGBRegressor(max_depth = max_depth, n_estimators = n_estimators)
        xg.fit(X_train, y_train)
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

def get_mae(predict, true, baseline):
    speed = metrics.mean_absolute_error(predict['speed'], true['speed'])
    speed_base=metrics.mean_absolute_error(baseline['speed'], true['speed'])
    angle = metrics.mean_absolute_error(predict['angle'], true['angle'])
    angle_base=metrics.mean_absolute_error(baseline['angle'], true['angle'])
    return speed, speed_base, angle, angle_base 
    


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    #get data
    measurement=prep.prepare_measurement()
    forecast = prep.prepare_forecast()
    #keep useful columns
    measurement= measurement[['speed', 'cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation', 'precip','season', 'am']]

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={})
    pred_speed=pd.DataFrame(columns={})
    pred_angle=pd.DataFrame(columns={})

    #prediction steps
    t_list= args.t_list #[1,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]

    steps_in = args.steps_in

    for t in t_list:
        #run model
        predict, true, base = run_xgb(steps_in, steps_out=t, max_depth = args.max_depth, n_estimators = args.n_estimators)

        #calculate angles from sin and cosine  
        predict['angle'] = predict.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']),axis = 1)
        true['angle'] = true.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']), axis = 1)
        base['angle'] = base.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']), axis = 1)

        #calculate mae for regression 
        mae_speed, mae_speed_base, mae_angle, mae_angle_base = get_mae(predict, true, base) 
        #calculate accuracy & auc for scenario prediction 
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
                                          'base_binary_auc':base_bin_auc,
                                            'xbg_speed_mae': mae_speed,
                                            'base_speed_mae': mae_speed_base,
                                            'xgb_angle_mae': mae_angle,
                                            'base_angle_mae': mae_angle_base}, ignore_index=True)
        #record predicted speed
        pred_speed = pd.concat([pred_speed, predict['speed'].rename('speed_t+'+str(t))], axis=1)
        #record predicted angle
        pred_angle = pd.concat([pred_angle, predict['angle'].rename('angle_t+'+str(t))], axis=1)

    #output results df
    accuracy.to_csv('results/xgboost_accuracy_in_' + str(args.steps_in) + '_depth_' + str(args.max_depth) + '_estim_' + str(args.n_estimators) + '.csv', index=False)
    pred_angle.to_csv('results/xgboost_pred_angle_in_' + str(args.steps_in) + '_depth_' + str(args.max_depth) + '_estim_' + str(args.n_estimators) + '.csv', index=False)
    pred_speed.to_csv('results/xgboost_pred_speed_in_' + str(args.steps_in) + '_depth_' + str(args.max_depth) + '_estim_' + str(args.n_estimators) + '.csv', index=False)