from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from utils_scenario import *


def get_baselines(x_df, x, test_size = 0.2):
    y_baseline_speed = np.array(x_df['speed_forecast'])
    y_baseline_cos_wind = np.array(x_df['cos_wind_dir_forecast'])
    y_baseline_sin_wind = np.array(x_df['sin_wind_dir_forecast'])
    _, _, _, y_test_baseline_speed = train_test_split(x, y_baseline_speed, test_size = test_size, shuffle = False)
    _, _, _, y_test_baseline_cos_wind = train_test_split(x, y_baseline_cos_wind, test_size = test_size, shuffle = False)
    _, _, _, y_test_baseline_sin_wind = train_test_split(x, y_baseline_sin_wind, test_size = test_size, shuffle = False)
    y_baseline_dangerous_scenarios = get_all_dangerous_scenarios(y_test_baseline_speed, y_test_baseline_cos_wind, y_baseline_sin_wind)
    y_baseline_scenarios = get_all_scenarios(y_test_baseline_speed, y_test_baseline_cos_wind, y_baseline_sin_wind, b_scenarios=True)
    return y_test_baseline_speed, y_test_baseline_cos_wind, y_test_baseline_sin_wind, y_baseline_dangerous_scenarios, y_baseline_scenarios