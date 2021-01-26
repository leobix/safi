import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold

import warnings
warnings.filterwarnings('ignore')

#import local functions
from utils import utils_scenario as utils, data_preparation as prep, data_process as proc

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=48,
                            help="number of in time steps")

# parser.add_argument("--max-depth", type=int, default=5,
#                             help="maximum depth of XGB")
#
# parser.add_argument("--lr", type=float, default=0.3,
#                             help="lr of XGB")
#
# parser.add_argument("--subsample", type=float, default=1.,
#                             help="subsample in XGB")
#
# parser.add_argument("--min_child_weight", type=int, default=1,
#                             help="min_child_weight in XGB")
#
# parser.add_argument("--colsample_bytree", type=float, default=1,
#                             help="colsample_bytree in XGB")
#
# parser.add_argument("--n_estimators", type=int, default=100,
#                             help="number of estimators of XGB")

parser.add_argument("--t_list", type=int, nargs="+", default=[1,2,3,4,5,6],  #9,12,15,18,21,24,27,30,33,36,39,42,45,48],
                            help="list of prediction time steps")

def run_xgb(steps_in, steps_out):

    #Parameter list:
    param_list=['speed','cos_wind_dir','sin_wind_dir','scenario','dangerous']

    predict = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir','scenario','dangerous'})
    true = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir','scenario','dangerous'})
    baseline = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir','scenario','dangerous'})

    for param in param_list:
        x_df, y_df, x, y = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, param)
        x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, shuffle = False)

        # @Leonard: I added new parameters here for scenario and dangerous, call classification algo
        #gridsearch
        if param in ['speed','cos_wind_dir','sin_wind_dir']:
            xgb_model = XGBRegressor()
            splitter = KFold(n_splits=5, shuffle=True)
            print(param)
        if param in ['scenario', 'dangerous']:
            xgb_model = XGBClassifier()
            splitter = StratifiedKFold(n_splits=5, shuffle = True)

        grid = GridSearchCV(xgb_model,
                            param_grid = grid_params,
                            cv = splitter.split(x_train, y_train))
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_


        y_hat = best_model.predict(X_test)

        predict[param] = pd.Series(y_hat)
        #print(np.array(y_test).reshape(-1))
        true[param] = pd.Series(np.array(y_test).reshape(-1))#y_test.flatten())
        baseline[param] = x_df[param+'_forecast'][-len(y_hat):]

    #reset index
    baseline.reset_index(inplace=True)
    return predict, true, baseline

def scenario_accuracy_indirect(predict, true, baseline):
    pred = utils.get_all_scenarios(predict['speed'], predict['cos_wind_dir'],predict['sin_wind_dir'], b_scenarios=True)
    true = utils.get_all_scenarios(true['speed'], true['cos_wind_dir'],true['sin_wind_dir'], b_scenarios=True)
    base = utils.get_all_scenarios(baseline['speed'], baseline['cos_wind_dir'],baseline['sin_wind_dir'], b_scenarios=True)

    #calculate prediction accuracies
    pred_score = metrics.accuracy_score(pred, true).round(3)
    base_score = metrics.accuracy_score(base, true).round(3)

    return  pred_score, base_score

def binary_accuracy_indirect(predict, true, baseline):
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

def get_mae_indirect(predict, true, baseline):
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
    measurement= measurement[['speed', 'cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation', 'precip','season','scenario','dangerous']]

    #set up empty dataframes
    accuracy = pd.DataFrame(columns={})
    pred_speed=pd.DataFrame(columns={})
    pred_angle=pd.DataFrame(columns={})

    #prediction steps
    t_list= args.t_list #[1,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]

    steps_in = args.steps_in

    #parameter search space
    grid_params = {
         'max_depth':[4,5,6],
         'min_child_weight':[4,5,6],
         'gamma': [0, 0.05],
         'learning_rate': [0.1,0.2],
         'n_estimators': [100, 150]}

    for t in t_list:
        #run model
        predict, true, base = run_xgb(steps_in, steps_out=t)

        #calculate angles from sin and cosine
        predict['angle'] = predict.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']),axis = 1)
        true['angle'] = true.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']), axis = 1)
        base['angle'] = base.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']), axis = 1)

        # #calculate mae for regression
        mae_speed, mae_speed_base, mae_angle, mae_angle_base = get_mae(predict, true, base)
        #calculate accuracy & auc for scenario prediction
        # pred_scenario, base_scenario  = scenario_accuracy_indirect(predict, true, base)
        # pred_bin_accu, base_bin_accu, pred_bin_auc, base_bin_auc= binary_accuracy_indirect(predict, true, base)

        #accuracy for direct prediction
        pred_scenario= metrics.accuracy_score(predict['scenario'], true['scenario']).round(3)
        base_scenario = metrics.accuracy_score(base['scenario'], true['scenario']).round(3)
        pred_dangerous= metrics.roc_auc_score(predict['dangerous'], true['dangerous']).round(3)
        base_dangerous= metrics.roc_auc_score(base['dangerous'], true['dangerous']).round(3)

        #record accuracy
        accuracy = accuracy.append({'past_n_steps': str(steps_in),
                                          'pred_n_steps': str(t),
                                          'xgb_scenario_accu': pred_scenario,
                                          'benchmark_scenario_accu': base_scenario,
                                          'xbg_binary_rocauc':pred_dangerous,
                                          'benchmark_binary_rocauc':base_dangerous,
                                          # 'xbg_binary_auc':pred_bin_auc,
                                          # 'base_binary_auc':base_bin_auc,
                                          'xbg_speed_mae': mae_speed,
                                          'benchmark_speed_mae': mae_speed_base,
                                          'xgb_angle_mae': mae_angle,
                                          'benchmark_angle_mae': mae_angle_base}, ignore_index=True)
        #record predicted speed
        pred_speed = pd.concat([pred_speed, predict['speed'].rename('speed_t+'+str(t))], axis=1)
        #record predicted angle
        pred_angle = pd.concat([pred_angle, predict['angle'].rename('angle_t+'+str(t))], axis=1)

    #output results df
    accuracy.to_csv('results/xgboost_accuracy_in_' + str(args.steps_in) + '.csv', index=False)
    # pred_angle.to_csv('results/xgboost_pred_angle_in_' + str(args.steps_in) + '_depth_' + str(args.max_depth) + '_estim_' + str(args.n_estimators) + '.csv', index=False)
    # pred_speed.to_csv('results/xgboost_pred_speed_in_' + str(args.steps_in) + '_depth_' + str(args.max_depth) + '_estim_' + str(args.n_estimators) + '.csv', index=False)
