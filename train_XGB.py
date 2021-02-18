import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from imblearn.over_sampling import SMOTE


# from datetime import timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

#import local functions
from utils import utils_scenario as utils, data_preparation as prep, data_process as proc

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=48,
                            help="number of in time steps")

parser.add_argument("--t_list", type=int, nargs="+", default=[1,2,3,4,5,6],
                            help="list of prediction time steps")

# main function to train xgb models and save models as pickle files
def train_xgb(measurement, forecast, steps_in, steps_out):
    #flag message
    print('running xgb for steps_out=', steps_out)
    #Parameter list:
    param_list = ['scenario','dangerous'] #['speed','cos_wind_dir','sin_wind_dir','scenario','dangerous']

    for param in param_list:
        print(param)
        #train on the entire data
        x_df, y_df, x, y = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, param)

        #gridsearch
        if param in ['speed','cos_wind_dir','sin_wind_dir']:
            xgb_model = XGBRegressor()
            splitter = KFold(n_splits=4, shuffle=True)
            score = 'neg_mean_absolute_error'

        if param in ['scenario', 'dangerous']:
            xgb_model = XGBClassifier()
            splitter = StratifiedKFold(n_splits=4, shuffle = True)
            score = 'accuracy'

            if (param == 'dangerous'):
                sm = SMOTE(sampling_strategy = 0.6, random_state=0)
                x, y = sm.fit_resample(x, y)
                score = 'roc_auc'

        grid = GridSearchCV(xgb_model,
                            param_grid = grid_params, scoring = score,
                            cv = splitter.split(x, y))
        grid.fit(x, y)
        best_model = grid.best_estimator_

        #print grid parameters
        print('gridsearch result for param: ', param )
        print(grid.best_params_)

        #save model into a pickle file
        pickle.dump(best_model, open('trained_models/'+str(param)+'_t_'+str(steps_out), 'wb'))
    return




if __name__ == "__main__":
    print('running train_xgb')
    args = parser.parse_args()
    print(args)

    #get data
    measurement=prep.prepare_measurement()
    forecast = prep.prepare_forecast()

    #keep selected features
    measurement= measurement[['speed', 'cos_wind_dir', 'sin_wind_dir',
    'temp', 'radiation', 'precip','season','scenario','dangerous']]

    print('measurement features used to construct x_df:', measurement.columns.to_list())

    # small sample size
    # measurement = measurement.iloc[:6000, :]

    #prediction steps
    t_list= args.t_list # np.arange(1,49,1)
    steps_in = args.steps_in #48

    #parameter search space
    grid_params = {
         'max_depth':[4,5,6],
         'min_child_weight':[6],
         'gamma': [0, 0.05],
         'learning_rate': [0.1],
         'n_estimators': [100, 150]}

    #run models per prediction step
    for t in t_list:
        #run model and save model
        train_xgb(measurement, forecast, steps_in, t)
