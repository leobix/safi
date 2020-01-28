#!/usr/bin/env python3


import argparse

from utils import *
from data_preperation import * 
from numpy.random import seed
from tensorflow import set_random_seed



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps-in", type=int, default=48,
                            help="number of in time steps")

parser.add_argument("--steps-out", type=int, default=48,
                            help="number of out time steps")

parser.add_argument("--lr", type=float, default=0.001,
                            help="Adam learning rate")

parser.add_argument("--drop-out", type=float, default=0.5,
                            help="drop out")

parser.add_argument("--cell1size", type=int, default=64,
                            help="cell size for lstm")

parser.add_argument("--epochs", type=int, default=20,
                            help="epochs")

parser.add_argument("--batch_size", type=int, default=64,
                            help="batch_size")

parser.add_argument("--test_size", type=float, default=0.15,
                            help="test_size")

def main(args):

	seed(6)
	set_random_seed(6)
	data=prepare_data(one_hot=False)
	data_merge = prepare_data_with_forecast(data)

	df= data_merge[['speed', 'cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation',
       'scenario_num', 'cos_hour', 'sin_hour',
       'cos_day', 'sin_day', 'daily_min_speed', 'daily_min_hour',
       'daily_max_speed', 'daily_max_hour']] 
       #'precip', 'season', 'day', 'night'


	X_train, X_test, y_train, y_test, y_unscaled, sc_X, sc_y = scale_data_RNN(args.steps_in, args.steps_out, df, test_size=args.test_size)
	model = create_model(X_train, args.steps_in, args.steps_out, args.lr, args.drop_out, args.cell1size)
	history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_test, y_test), verbose=2, shuffle=True)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)


