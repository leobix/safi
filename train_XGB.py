import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

#import local functions
from utils import utils_scenario as utils, data_preparation as prep, data_process as proc

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=48,
                            help="number of in time steps")

parser.add_argument("--t_list", type=list, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48],
                            help="list of prediction time steps")

# main function to train xgb models and save models as pickle files
def train_xgb(measurement, forecast, steps_in, steps_out):

    #Parameter list:
    param_list =['speed','cos_wind_dir','sin_wind_dir']

    predict = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})
    true = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})
    baseline = pd.DataFrame(columns={'speed','cos_wind_dir','sin_wind_dir'})

    for param in param_list:

        #train on the entire data
        x_df, y_df, x, y = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, param)
        xgb = XGBRegressor(max_depth = 5)
        xgb.fit(x, y)

        #save model into a pickle file
        pickle.dump(xgb, open('trained_models/'+str(param)+'_t_'+str(steps_out), 'wb'))
    return


    if __name__ == "__main__":
        args = parser.parse_args()
        print(args)

        #get data
        measurement=prep.prepare_measurement()
        forecast = prep.prepare_forecast()
        #keep useful columns

        measurement= measurement[['speed', 'cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation', 'precip','season']]
        print('measurement features used to construct x_df:', measurement.columns.to_list())

        #prediction steps
        t_list= args.t_list # np.arange(1,49,1)
        steps_in = args.steps_in #48

        #run three models per prediction step
        for t in t_list:
            #run model and save model
            train_xgb(measurement, forecast, steps_in, t)
