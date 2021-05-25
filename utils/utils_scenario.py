import numpy as np

def get_angle_in_degree(cos, sin):
    #check if cos within reasonable range:
    if (cos>=-1) & (cos <=1):
        angle = 360 * np.arccos(cos) / (2*np.pi)
        if sin < 0:
            angle = 360 - angle
    #check if sin within reasonable range:
    elif (sin>=-1) & (sin <=1):
        angle = 360 * np.arcsin(sin) / (2*np.pi)
        if cos < 0:
            angle = 180 - angle
        if angle < 0:
            angle += 360
    else:
        angle=0
        #print("cos", cos)
        #print(sin)
        #print('cos and sin out of range, returned 0')
    #because we care about the reverse angle for the scenarios
    return angle #(angle + 180) % 360

def test_angle(cos, sin):
    angle_cos = 360 *np.arccos(cos ) /( 2 *np.pi)
    if sin <=0:
        angle_cos += 180
    angle_sin = 360 *np.arcsin(sin ) /( 2 *np.pi)
    if cos <=0:
        angle_sin +=180
    return angle_cos, angle_sin

def is_favorable(cos, sin):
    # angle is between NO and E
    angle = get_angle_in_degree(cos, sin)
    # NO
    if angle > 303.75:
        return True
    # under E
    elif angle < 101.25:
        return True
    return False

def is_South(cos, sin):
    angle = get_angle_in_degree(cos, sin)
    # Direction S, SSO, SSE
    if 146.25 <= angle <= 213.75:
        return True
    return False


def is_S1(speed, cos, sin):
    if speed >= 4:
        return True
    elif speed > 1 and is_favorable(cos, sin):
        return True
    return False


def is_S2(speed, cos, sin):
    if 2 <= speed < 4 and not is_favorable(cos, sin):
        return True
    elif 0.5 <= speed <= 1 and is_favorable(cos, sin):
        return True
    return False


def is_S2b(speed, cos, sin):
    if 1 <= speed <= 2 and not is_favorable(cos, sin) and not is_South(cos, sin):
        return True
    return False


def is_S3(speed, cos, sin):
    if 0.5 <= speed < 1 and not is_favorable(cos, sin) and not is_South(cos, sin):
        return True
    elif speed < 0.5 and is_favorable(cos, sin):
        return True
    return False


def is_S3b(speed, cos, sin):
    if speed < 2 and is_South(cos, sin):
        return True
    return False


def is_S4(speed, cos, sin):
    if speed < 0.5 and not is_favorable(cos, sin):
        return True
    return False


def get_scenario(speed, cos, sin, b_scenarios):
    if is_S1(speed, cos, sin):
        return 1
    elif is_S2(speed, cos, sin):
        return 2
    elif is_S2b(speed, cos, sin):
        return 2 + b_scenarios
    elif is_S3(speed, cos, sin):
        return 3 + b_scenarios
    elif is_S3b(speed, cos, sin):
        return 3 + 2*b_scenarios
    elif is_S4(speed, cos, sin):
        return 4 + 2*b_scenarios
    # print("There is a problem.")
    # print("speed: ", speed, " cos: ", cos, " sin: ", sin)

def get_dangerous_scenario(speed, cos, sin):
    if is_S1(speed, cos, sin):
        return 0
    elif is_S2(speed, cos, sin):
        return 0
    elif is_S2b(speed, cos, sin):
        return 0
    elif is_S3(speed, cos, sin):
        return 1
    elif is_S3b(speed, cos, sin):
        return 1
    elif is_S4(speed, cos, sin):
        return 1
    # print("There is a problem.")
    # print("speed: ", speed, " cos: ", cos, " sin: ", sin)


def get_all_scenarios(y_speed, y_cos, y_sin, b_scenarios = False):
    y = []
    for i in range(len(y_speed)):
        y.append(get_scenario(y_speed[i], y_cos[i], y_sin[i], b_scenarios))
    return np.array(y)

def get_all_dangerous_scenarios(y_speed, y_cos, y_sin):
    y = []
    for i in range(len(y_speed)):
        y.append(get_dangerous_scenario(y_speed[i], y_cos[i], y_sin[i]))
    return np.array(y)


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
