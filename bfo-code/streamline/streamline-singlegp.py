import warnings
warnings.filterwarnings('ignore')


import math
import torch

from matplotlib import pyplot as plt

import os
import argparse
import pandas as pd
from pandas import factorize
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
#from sklearn.model_selection import KFold

import gpytorch
import itertools

from m3gpr.config import get_cfg_defaults_all
from m3gpr.load_data import Data_set
from m3gpr.models import singleGP_gpytorch,rf_sklearn,singleGP_gpytorch_train,singleGP_gpytorch_reference,MultitaskGPModel,MTMOGP
from m3gpr.train import CV_Trainer


def main():
    cfg = get_cfg_defaults_all()
    os.chdir('/Users/chenya68/Documents/GitHub/BFO')
    parser = argparse.ArgumentParser(description="Configure files")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args(['--cfg', 'data/harpoon/harpoon_dataset.yaml'])
    cfg.merge_from_file(args.cfg)

    cfg.freeze()
    setup_data = Data_set(cfg, 1)
    print('MODEL_NAME',cfg.MODEL.MODEL_NAME)
    print('Y_SCALE',cfg.MODEL.Y_SCALE)
    print('num_tasks',setup_data.num_tasks)
    
    df_x, df_y,ls_X_train,ls_X_test,ls_y_train,ls_y_test, obj_y_scaler= setup_data.Load_data_set()
    print(df_x.shape)
    print(df_y.shape)

    sgp_cv_trainer = CV_Trainer(cfg,ls_X_train,ls_X_test,ls_y_train,ls_y_test,obj_y_scaler)
    combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err = sgp_cv_trainer.set_up_sgp_cv()

    if cfg.MODEL.SPLIT == 'by-task':
        num_tasks = cfg.DATA.NUM_TASKS
    else:
        num_tasks = 1
    num_outputs = cfg.DATA.NUM_OUTPUTS
    num_total_outputs = num_outputs*num_tasks
    arr_metrics = np.zeros((3,1+num_total_outputs)) #row:mae, r2, avg_std; col: total mean, task1(output1), task2(output2),...

    print('number of tasks that are being compared: ',num_total_outputs)

    best_idx = np.argmin(mean_arr_cv_mae[1,:])
    #best_idx = np.argmax(mean_arr_cv_r2[1,:])

    print('best_idx',best_idx)
    print('best combinations: ',combs[best_idx])

    arr_metrics[0,0] = np.min(mean_arr_cv_mae[1,:])
    arr_metrics[1,0] = np.max(mean_arr_cv_r2[1,:])
    arr_metrics[2,0] = mean_arr_cv_err[1,best_idx]

    print('best mean mae %.3f' % arr_metrics[0,0])
    print('best mean r2 %.3f' % arr_metrics[1,0])
    print('best mean err %.3f' % arr_metrics[2,0])
    
    cols_target_new = ['total']
    for task_ind in range(num_tasks):
        for out_ind in range(num_outputs):
            j = out_ind + num_outputs*task_ind
    #for j in range(num_total_outputs):
            j_name = 'Task_'+str(task_ind+1) + '_Output_'+str(out_ind+1)
            cols_target_new.append(j_name)
            arr_metrics[0,j+1] = ls_arr_cv_mae[j][1,best_idx]
            arr_metrics[1,j+1] = ls_arr_cv_r2[j][1,best_idx]
            arr_metrics[2,j+1] = ls_arr_cv_err[j][1,best_idx]
            print('best mae of target %d is %.3f' %(j,ls_arr_cv_mae[j][1,best_idx]))
            print('best r2 of target %d is %.3f' %(j,ls_arr_cv_r2[j][1,best_idx]))
            print('best err of target %d is %.3f' % (j,ls_arr_cv_err[j][1,best_idx]))
    
    df_test_metrics = pd.DataFrame(np.round(arr_metrics,3),columns = cols_target_new, 
                               index = ['MAE','R2','AVG_STD'])
    print(df_test_metrics)
    
    output_path = '/'.join([cfg.PATH.RESULT,cfg.MISC.DATA_NAME,cfg.MODEL.MODEL_NAME,cfg.MODEL.SPLIT,cfg.MISC.DATE])
    df_test_metrics.to_csv(output_path+'/df_test_metrics.csv')

if __name__ == "__main__":
    main()