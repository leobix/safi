import sys
sys.path.append('../')
import pandas as pd
from utils import data_process

def is_favorable(cos, sin):
    # angle is between NO and E
    angle = data_process.get_angle_in_degree(cos, sin)
    # NO
    if angle > 303.75:
        return True
    # under E
    elif angle < 101.25:
        return True
    return False

def is_South(cos, sin):
    angle = data_process.get_angle_in_degree(cos, sin)
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

def get_str_scenario(speed, cos, sin):
    if is_S1(speed, cos, sin):
        return 'S1'
    elif is_S2(speed, cos, sin):
        return 'S2'
    elif is_S2b(speed, cos, sin):
        return 'S2b'
    elif is_S3(speed, cos, sin):
        return 'S3'
    elif is_S3b(speed, cos, sin):
        return 'S3b'
    elif is_S4(speed, cos, sin):
        return 'S4'
    return ''

def get_directions_label(angle):
    directions = pd.read_csv('../utils/directions.csv',sep=';',decimal=',')
    result = ' '
    for idx,row in directions.iterrows():
        if (row['lower'] <= angle) & (angle < row['upper']):
            result = row['name']
    return result

def round_numtech(x,dg):
    try:
        if dg:
            return round(x,dg)
        else :
            return round(x)
    except:
        return ' '
    
def color_direction(val):
    neg_directions = ['ESE','SE','SSE','S','SSO','SO','OSO','O','ONO']
    color = 'red' if val in neg_directions else 'green'
    return f'background-color: {color}'