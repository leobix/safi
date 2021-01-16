#!/usr/bin/env python3


import argparse

import pandas as pd

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#example of how to call preparation.py
from utils import data_preparation as prep, data_process as proc
from utils.utils_baselines import *

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

##IAI
from julia import Julia


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=48,
                            help="number of in time steps")

parser.add_argument("--steps-out", type=int, default=24,
                            help="number of out time steps")

parser.add_argument("--min-depth", type=int, default=6,
                            help="min depth")

parser.add_argument("--max-depth", type=int, default=7,
                            help="max depth (not included)")

parser.add_argument("--local", action='store_true',
                            help="if used on local computer because of iai compatibility")

parser.add_argument("--supercloud", action='store_true',
                            help="if used on supercloud because of iai compatibility")

parser.add_argument("--classification", action='store_true',
                            help="classification models will run")

parser.add_argument("--autobalance", action='store_true',
                            help="for data unbalance")

parser.add_argument("--regression_sparsity", action='store_true',
                            help="for using regression sparsity, regression_sparsity specifies whether to use linear regresssion predictions in the leaves of the tree. The default value is 0 indicating only a constant term will be used for prediction. Set to :all to fit a linear regression in each leaf.")

parser.add_argument("--filename", type=str, default='v1',
                            help="filename for Trees")

parser.add_argument("--class-criterion", type=str, default='misclassification',
                            help="classification criterion, choose between misclassification, gini and entropy")




def main(args):
    # call data_preparation.py
    measurement = prep.prepare_measurement()
    forecast = prep.prepare_forecast()

    # keep useful columns
    measurement = measurement[['speed', 'cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation', 'precip', 'season']]

    # call data_process.py

    steps_in = args.steps_in
    steps_out = args.steps_out

    x_df, y_df, x, y_speed = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, 'speed')
    _, _, _, y_cos = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, 'cos_wind_dir')
    _, _, _, y_sin = proc.prepare_x_y(measurement, forecast, steps_in, steps_out, 'sin_wind_dir')
    y_scenarios = get_all_scenarios(np.array(y_speed), np.array(y_cos), np.array(y_sin), b_scenarios=True)
    y_dangerous = get_all_dangerous_scenarios(np.array(y_speed), np.array(y_cos), np.array(y_sin))
    X_train, X_test, y_train_dangerous, y_test_dangerous = train_test_split(x, y_dangerous, test_size=0.2,
                                                                            shuffle=False)
    _, _, y_train_scenarios, y_test_scenarios = train_test_split(x, y_scenarios, test_size=0.2, shuffle=False)
    _, _, y_train_speed, y_test_speed = train_test_split(x, y_speed, test_size=0.2, shuffle=False)
    _, _, y_train_cos, y_test_cos = train_test_split(x, y_cos, test_size=0.2, shuffle=False)
    _, _, y_train_sin, y_test_sin = train_test_split(x, y_sin, test_size=0.2, shuffle=False)
    #
    names = list(x_df.columns)
    names.remove('present_time')
    names.remove('forecast_time')
    X_train2 = pd.DataFrame(X_train)
    X_train2.columns = names
    (X_train_reg, y_train_speed_reg), _ = iai.split_data('regression', X_train2, np.array(y_train_speed), train_proportion=0.9999)
    (_, y_train_cos_reg), _ = iai.split_data('regression', X_train2, np.array(y_train_cos), train_proportion=0.9999)
    (_, y_train_sin_reg), _ = iai.split_data('regression', X_train2, np.array(y_train_sin), train_proportion=0.9999)
    ###BASELINES
    y_test_baseline_speed, y_test_baseline_cos_wind, y_test_baseline_sin_wind, y_baseline_dangerous_scenarios, y_baseline_scenarios = get_baselines(
        x_df, x)

    if args.regression_sparsity:
        regression_sparsity = 'all'
    else:
        regression_sparsity = 0

    ###Regression
    #Grids
    grid_speed = iai.GridSearch(
        iai.OptimalTreeRegressor(
            random_seed=1,
        ),
        max_depth=range(args.min_depth, args.max_depth),
        regression_sparsity = regression_sparsity,
        #minbucket = [10, 20],
    )

    grid_cos = iai.GridSearch(
        iai.OptimalTreeRegressor(
            random_seed=1,
        ),
        max_depth=range(args.min_depth, args.max_depth),
        regression_sparsity=regression_sparsity,
        #minbucket=[10, 20],
    )

    grid_sin = iai.GridSearch(
        iai.OptimalTreeRegressor(
            random_seed=1,
        ),
        max_depth=range(args.min_depth, args.max_depth),
        regression_sparsity=regression_sparsity,
        #minbucket=[10, 20],
    )

    #Fit
    grid_speed.fit(X_train_reg, np.array(y_train_speed_reg).reshape(-1))
    lnr_speed = grid_speed.get_learner()
    lnr_speed.write_html("Trees/" + args.filename + "_Regression_tree_speed_in" + str(args.steps_in) + "_out" + str(args.steps_out) + "_sparsity_" + str(regression_sparsity) +".html")
    lnr_speed.write_json("Trees/" + args.filename + "_Regression_tree_speed_in" + str(args.steps_in) + "_out" + str(args.steps_out) + "_sparsity_" + str(regression_sparsity)  + ".json")
    print(grid_speed.get_grid_results())

    grid_cos.fit(X_train_reg, np.array(y_train_cos_reg).reshape(-1))
    lnr_cos = grid_cos.get_learner()
    lnr_cos.write_html("Trees/" + args.filename + "_Regression_tree_cos_in" + str(args.steps_in) + "_out" + str(args.steps_out) + "_sparsity_" + str(regression_sparsity) +".html")
    lnr_cos.write_json("Trees/" + args.filename + "_Regression_tree_cos_in" + str(args.steps_in) + "_out" + str(
        args.steps_out)+ "_sparsity_" + str(regression_sparsity) + ".json")
    print(grid_cos.get_grid_results())

    grid_sin.fit(X_train_reg, np.array(y_train_sin_reg).reshape(-1))
    lnr_sin = grid_sin.get_learner()
    lnr_sin.write_html("Trees/" + args.filename + "_Regression_tree_sin_in" + str(args.steps_in) + "_out" + str(args.steps_out) + "_sparsity_" + str(regression_sparsity)  +".html")
    lnr_sin.write_json("Trees/" + args.filename + "_Regression_tree_sin_in" + str(args.steps_in) + "_out" + str(
        args.steps_out) + "_sparsity_" + str(regression_sparsity) + ".json")
    print(grid_sin.get_grid_results())

    #Predict
    y_hat_speed = grid_speed.predict(X_test)
    y_hat_cos = grid_cos.predict(X_test)
    y_hat_sin = grid_sin.predict(X_test)
    y_hat_scenario_from_regression = get_all_scenarios(y_hat_speed, y_hat_cos, y_hat_sin, b_scenarios=True)
    y_hat_dangerous_from_regression = get_all_dangerous_scenarios(y_hat_speed, y_hat_cos, y_hat_sin)

    #Score
    print("MAE speed is: ", mean_absolute_error(y_test_speed, y_hat_speed))
    print("MAE baseline speed is: ", mean_absolute_error(y_test_speed, y_test_baseline_speed))

    print("MAE cos is: ", mean_absolute_error(y_test_cos, y_hat_cos))
    print("MAE baseline cos is: ", mean_absolute_error(y_test_cos, y_test_baseline_cos_wind))

    print("MAE sin is: ", mean_absolute_error(y_test_sin, y_hat_sin))
    print("MAE baseline sin is: ", mean_absolute_error(y_test_sin, y_test_baseline_sin_wind))

    print("\nRegression Based Scores:")
    print("Regression based scenarios, Accuracy:",
          accuracy_score(y_test_scenarios, y_hat_scenario_from_regression))
    print("Regression based dangerous, Accuracy:", accuracy_score(y_test_dangerous, y_hat_dangerous_from_regression))
    print("Regression based dangerous, AUC:", roc_auc_score(y_test_dangerous, y_hat_dangerous_from_regression))

    if args.classification:

        ###Classification
        #Grids
        grid_scenarios = iai.GridSearch(
            iai.OptimalTreeClassifier(
                random_seed=1,
                criterion = args.class_criterion
            ),
            max_depth=range(args.min_depth, args.max_depth),
        )

        grid_dangerous = iai.GridSearch(
            iai.OptimalTreeClassifier(
                random_seed=1,
                criterion = args.class_criterion
            ),
            max_depth=range(args.min_depth, args.max_depth),
        )

        grid_scenarios.fit(X_train2, y_train_scenarios)
        lnr_scenarios = grid_scenarios.get_learner()
        lnr_scenarios.write_html("Trees/" + args.filename + "_Classification_tree_scenarios_in" + str(args.steps_in) + "_out" + str(args.steps_out) +".html")
        lnr_scenarios.write_json(
            "Trees/" + args.filename + "_Classification_tree_scenarios_in" + str(args.steps_in) + "_out" + str(
                args.steps_out) + ".json")
        print(grid_scenarios.get_grid_results())

        grid_dangerous.fit(X_train2, y_train_dangerous)
        lnr_dangerous = grid_dangerous.get_learner()
        lnr_dangerous.write_html("Trees/" + args.filename + "_Classification_tree_dangerous_in" + str(args.steps_in) + "_out" + str(args.steps_out) +".html")
        lnr_dangerous.write_json(
            "Trees/" + args.filename + "_Classification_tree_dangerous_in" + str(args.steps_in) + "_out" + str(
                args.steps_out) + ".json")
        print(grid_dangerous.get_grid_results())

        if args.autobalance:
            grid_scenarios_autobalance = iai.GridSearch(
                iai.OptimalTreeClassifier(
                    random_seed=1,
                ),
                max_depth=range(args.min_depth, args.max_depth),
            )

            grid_dangerous_autobalance = iai.GridSearch(
                iai.OptimalTreeClassifier(
                    random_seed=1,
                ),
                max_depth=range(args.min_depth, args.max_depth),
            )

            grid_scenarios_autobalance.fit(X_train2, y_train_scenarios, sample_weight='autobalance')
            lnr_scenarios_autobalance = grid_scenarios_autobalance.get_learner()
            lnr_scenarios_autobalance.write_html(
                "Trees/" + args.filename + "_Classification_tree_scenarios_autobalance_in" + str(args.steps_in) + "_out" + str(args.steps_out) + ".html")
            lnr_scenarios_autobalance.write_json(
                "Trees/" + args.filename + "_Classification_tree_scenarios_autobalance_in" + str(
                    args.steps_in) + "_out" + str(args.steps_out) + ".json")
            print(grid_scenarios_autobalance.get_grid_results())

            grid_dangerous_autobalance.fit(X_train2, y_train_dangerous, sample_weight='autobalance')
            lnr_dangerous_autobalance = grid_dangerous_autobalance.get_learner()
            lnr_dangerous_autobalance.write_html(
                "Trees/" + args.filename + "_Classification_tree_dangerous_autobalance_in" + str(args.steps_in) + "_out" + str(args.steps_out) + ".html")
            lnr_dangerous_autobalance.write_json(
                "Trees/" + args.filename + "_Classification_tree_dangerous_autobalance_in" + str(
                    args.steps_in) + "_out" + str(args.steps_out) + ".json")
            print(grid_dangerous_autobalance.get_grid_results())

            print("Classification based scenarios autobalance, Accuracy: ",
                  lnr_scenarios_autobalance.score(X_test, y_test_scenarios, criterion='misclassification'))

            print("Classification based dangerous autobalance: Accuracy: ",
                  lnr_dangerous_autobalance.score(X_test, y_test_dangerous, criterion='misclassification'))

            print("Classification based dangerous autobalance: AUC: ",
                  lnr_dangerous_autobalance.score(X_test, y_test_dangerous, criterion='auc'))

        print("Baselines:")
        print("Baseline scenarios, Accuracy:", accuracy_score(y_test_scenarios, y_baseline_scenarios))
        print("Naive Baseline dangerous, Accuracy: ", 1 - np.sum(y_test_dangerous) / len(y_test_dangerous))
        print("Baseline dangerous, Accuracy:", accuracy_score(y_test_dangerous, y_baseline_dangerous_scenarios))
        print("Baseline dangerous, AUC:", roc_auc_score(y_test_dangerous, y_baseline_dangerous_scenarios))

        print("\nRegression Based Scores:")
        print("Regression based scenarios, Accuracy:",
              accuracy_score(y_test_scenarios, y_hat_scenario_from_regression))
        print("Regression based dangerous, Accuracy:", accuracy_score(y_test_dangerous, y_hat_dangerous_from_regression))
        print("Regression based dangerous, AUC:", roc_auc_score(y_test_dangerous, y_hat_dangerous_from_regression))

        print("\nClassification Based Scores:")
        print("Classification based scenarios, Accuracy: ", lnr_scenarios.score(X_test, y_test_scenarios, criterion='misclassification'))
        print("Classification based dangerous, Accuracy: ", lnr_dangerous.score(X_test, y_test_dangerous, criterion='misclassification'))
        print("Classification based dangerous, AUC: ", lnr_dangerous.score(X_test, y_test_dangerous, criterion='auc'))


if __name__ == "__main__":
   args = parser.parse_args()
   print(args)
   if args.local:
        Julia(compiled_modules=False)
   elif args.supercloud:
       Julia(sysimage='/home/gridsan/groups/IAI/images/2.0.0/julia-1.4.2/sys.so', compiled_modules = False)
   else:
        Julia(sysimage='../sys.so')
   from interpretableai import iai
   main(args)


#sample_weight=:autobalance
