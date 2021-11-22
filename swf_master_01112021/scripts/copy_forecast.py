######################
#### Dependencies ####
######################
import os
import sys

#############################
#### Define path and day ####
#############################
sys_date = sys.argv[1]
sys_year = sys_date.split('/')[2][-2:]
sys_month = sys_date.split('/')[1]
sys_day = sys_date.split('/')[0]

################################
#### Copy files to SAVE_DIR ####
################################
MAIN_DIR = '//Sa-modat-cs-pr/plumair/SAFI/Sorties/' + '20' + sys_year + '/'
SAVE_DIR = '../data/raw/forecasts/'

sub_dir = [x for x in os.listdir(MAIN_DIR) if (x[-8:] == 'se_previ') and (x[5:10] == sys_month + '_' + sys_day)]

for s in sub_dir:
    print(s)
    # create saving folder
    if (os.path.exists(SAVE_DIR + s[0:13]) == False):
        os.mkdir(SAVE_DIR + s[0:13])
    
    # prediction day
    dir_day = os.listdir(MAIN_DIR + s)
    
    for d in dir_day:
        if (len(d) == 10):
            # create saving sub folder
            if (os.path.exists(SAVE_DIR + s[0:13] + '/' + d) == False):
                os.mkdir(SAVE_DIR + s[0:13] + '/' + d)
            copy_path = MAIN_DIR + s + '/' + d + '/meteo.txt'
            save_path = SAVE_DIR + s[0:13] + '/' + d + '/meteo.txt'
            t = os.system('copy ' + copy_path.replace('/','''\\''') + ' ' + save_path.replace('/','\\'))
            if t:
                print(s[0:13],d,t)