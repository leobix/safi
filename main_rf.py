import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from imblearn.over_sampling import SMOTE


import warnings
warnings.filterwarnings('ignore')

#import local functions
from utils import utils_scenario as utils, data_preparation as prep, data_process as proc

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=48,
                            help="number of in time steps")
parser.add_argument("--t_list", type=int, nargs="+", default=[1,2,3], #4,5,6
                            help="list of prediction time steps")

def run_rf(steps_in, steps_out):
    #flag message
    print('running random forrest for steps_out=', steps_out)
    #Parameter list:
    param_list=['scenario','dangerous','speed','cos_wind_dir','sin_wind_dir'] #['scenario','dangerous'] #

    predict_test = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir','scenario','dangerous','dangerous_proba'})
    predict_train = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir','scenario','dangerous','dangerous_proba'})

    for param in param_list:
        x_df, y_df, x, y = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, param)
        x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, shuffle = False)

        #gridsearch
        if param in ['speed','cos_wind_dir','sin_wind_dir']:
            rf_model = RandomForestRegressor()
            splitter = KFold(n_splits=4, shuffle=True)
            score = 'neg_mean_absolute_error' #MAE

        if param in ['scenario', 'dangerous']:
            rf_model = RandomForestClassifier()
            splitter = StratifiedKFold(n_splits=4, shuffle = True)
            score = 'accuracy'
            # SMOTE for binary classification
            if (param == 'dangerous'):
                sm = SMOTE(sampling_strategy = 0.6, random_state=0)
                x_train, y_train = sm.fit_resample(x_train, y_train)
                score = 'roc_auc'

        grid = GridSearchCV(rf_model,
                            param_grid = grid_params, scoring = score,
                            cv = splitter.split(x_train, y_train))
        grid.fit(x_train, y_train)

        print('gridsearch result for param: ', param )
        print(grid.best_params_)

        #save best parameters:
        pickle.dump(grid.best_params_, open('results/params/rf_'+param+'_'+str(steps_out)+'.pkl', 'wb'))
        best_model = grid.best_estimator_

        #record results
        predict_test[param] = pd.Series(best_model.predict(x_test))
        predict_train[param] = pd.Series(best_model.predict(x_train))

        if param == 'dangerous':
            predict_test['dangerous_proba'] = pd.Series(best_model.predict_proba(x_test)[:,1])
            predict_train['dangerous_proba'] = pd.Series(best_model.predict_proba(x_train)[:,1])

            #record baseline and truth
            predict_test['true'] = pd.Series(np.array(y_test).reshape(-1))#y_test.flatten())
            predict_train['true'] = pd.Series(np.array(y_train).reshape(-1))
            predict_test['baseline'] = x_df['dangerous_forecast'][-len(y_test):]
            predict_train['baseline'] = x_df['dangerous_forecast'][:len(y_train)]

    return predict_train, predict_test #, true, baseline

# def scenario_accuracy_indirect(predict, true):
#     pred = utils.get_all_scenarios(predict['speed'], predict['cos_wind_dir'],predict['sin_wind_dir'], b_scenarios=True)
#     true = utils.get_all_scenarios(true['speed'], true['cos_wind_dir'],true['sin_wind_dir'], b_scenarios=True)
#
#     #calculate prediction accuracies
#     pred_score = metrics.accuracy_score(true, pred).round(3)
#     return  pred_score
#
#
# def binary_accuracy_indirect(predict, true):
#     pred = utils.get_all_dangerous_scenarios(predict['speed'], predict['cos_wind_dir'],predict['sin_wind_dir'])
#     true = utils.get_all_dangerous_scenarios(true['speed'], true['cos_wind_dir'],true['sin_wind_dir'])
#     pred_auc = metrics.roc_auc_score(true,pred).round(3)
#     return  pred_auc #, base_auc, pred_score, base_score,
#
# #dangerous vs. not dangerous classification
# def binary_accuracy_from_scenario(predict, true, b_scenarios =True):
#     if b_scenarios:
#         pred = predict['scenario'].apply(lambda x: 0 if (x<=3) else 1)
#     else:
#         print('no b scnenario')
#     pred_auc = metrics.roc_auc_score(true['dangerous'],pred).round(3)
#     print('binary from scenario, confusion matrix', metrics.confusion_matrix(true['dangerous'], pred ))
#     return pred_auc
#
# def get_mae_indirect(predict, true, baseline):
#     speed = metrics.mean_absolute_error( true['speed'], predict['speed'])
#     speed_base=metrics.mean_absolute_error(true['speed'], baseline['speed'])
#     angle = metrics.mean_absolute_error( true['angle'],predict['angle'])
#     angle_base=metrics.mean_absolute_error( true['angle'], baseline['angle'])
#     return speed, speed_base, angle, angle_base
#


if __name__ == "__main__":
    print('running main_rf')
    args = parser.parse_args()
    print(args)

    #get data
    measurement=prep.prepare_measurement()
    # measurement.columns
    forecast = prep.prepare_forecast()
    #keep useful columns
    measurement= measurement[['speed', 'cos_wind_dir', 'sin_wind_dir',
    'temp', 'radiation', 'precip','season','scenario','dangerous']]

    #small sample size
    # measurement = measurement.iloc[3000:4000, :]

    #prediction steps
    t_list= args.t_list

    steps_in = args.steps_in

    grid_params = {
        'bootstrap': [True, False],
         'max_depth': [5,6],
         'min_samples_split': [4, 6],
         'n_estimators': [100, 150]}

    # predict_train, predict_test = run_rf(steps_in=1, steps_out=1)

    for t in t_list:
        #run model
        predict_train, predict_test= run_rf(steps_in, steps_out=t)

        # #calculate angles from sin and cosine
        # predict['angle'] = predict.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']),axis = 1)
        # true['angle'] = true.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']), axis = 1)
        # base['angle'] = base.apply(lambda row: utils.get_angle_in_degree(row['cos_wind_dir'],row['sin_wind_dir']), axis = 1)

        predict_train['dangerous_indirect'] = utils.get_all_dangerous_scenarios(predict_train['speed'], predict_train['cos_wind_dir'],predict_train['sin_wind_dir'])
        predict_test['dangerous_indirect'] = utils.get_all_dangerous_scenarios(predict_test['speed'], predict_test['cos_wind_dir'],predict_test['sin_wind_dir'])

        predict_train.to_csv('results/rf_result_train_'+str(t)+'.csv', index=False)
        predict_test.to_csv('results/rf_result_test_'+str(t)+'.csv', index=False)


        # # #calculate mae for regression
        # mae_speed, mae_speed_base, mae_angle, mae_angle_base = get_mae_indirect(predict, true, base)
        # #calculate accuracy & auc for scenario prediction
        # pred_scenario_indirect = scenario_accuracy_indirect(predict, true)
        # pred_dangerous_indirect= binary_accuracy_indirect(predict, true)
        #predict dangerous based on scenario
        # pred_dangerous_from_scenario = binary_accuracy_from_scenario(predict, true)

        # #accuracy for direct prediction
        # pred_scenario= metrics.accuracy_score( true['scenario'], predict['scenario']).round(3)
        # base_scenario = metrics.accuracy_score(true['scenario'], base['scenario']).round(3)
        # pred_dangerous= metrics.roc_auc_score( true['dangerous'], predict['dangerous']).round(3)
        # base_dangerous= metrics.roc_auc_score(true['dangerous'],base['dangerous']).round(3)

        #do confusion matrix:
        # pred_dangerous_from_scenario = binary_accuracy_from_scenario(predict, true)
        # print('binary from direct, confusion matrix', metrics.confusion_matrix(true['dangerous'], predict['dangerous']))

    #     #record accuracy
    #     accuracy = accuracy.append({'past_n_steps': str(steps_in),
    #                                       'pred_n_steps': str(t),
    #
    #                                       #scenario scores
    #                                       'xgb_scenario_accu': pred_scenario,
    #                                       'xbg_indirect_scienario_accu':pred_scenario_indirect,
    #                                       'benchmark_scenario_accu': base_scenario,
    #
    #                                       # dangerous auc
    #                                       'xbg_dangerous_rocauc':pred_dangerous,
    #                                       'benchmark_dangerous_rocauc':base_dangerous,
    #                                       'xbg_dangerous_from_scenario_rocauc': pred_dangerous_from_scenario,
    #                                       'xbg_dangerous_from_speed_rocauc':pred_dangerous_indirect,
    #
    #                                       'xbg_speed_mae': mae_speed,
    #                                       'benchmark_speed_mae': mae_speed_base,
    #                                       'xgb_angle_mae': mae_angle,
    #                                       'benchmark_angle_mae': mae_angle_base}, ignore_index=True)
    #     #record predicted speed
    #     pred_speed = pd.concat([pred_speed, predict['speed'].rename('speed_t+'+str(t))], axis=1)
    #     #record predicted angle
    #     pred_angle = pd.concat([pred_angle, predict['angle'].rename('angle_t+'+str(t))], axis=1)
    #
    # #output results df
    # accuracy.to_csv('results/xgboost_accuracy_gridsearch_'+str(t)+'.csv', index=False)
    # predict_train = predict_train[['dangerous','dangerous_proba','dangerous_indirect','true','baseline']]

    # predict_test = predict_test[['dangerous','dangerous_proba','dangerous_indirect','true','baseline']]


    # pred_angle.to_csv('results/xgboost_pred_angle_in_' + str(args.steps_in) + '_depth_' + str(args.max_depth) + '_estim_' + str(args.n_estimators) + '.csv', index=False)
    # pred_speed.to_csv('results/xgboost_pred_speed_in_' + str(args.steps_in) + '_depth_' + str(args.max_depth) + '_estim_' + str(args.n_estimators) + '.csv', index=False)
