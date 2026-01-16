import os
import re
import pandas as pd
from pandas import factorize
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import torch

class Data_set(object):
    def __init__(self, cfg, seed=0):
        ## Misc options
        self.Data_name = cfg.MISC.DATA_NAME
        self.model_name = cfg.MODEL.MODEL_NAME
        self.seed = seed

        ## Data options
        self.num_tasks = cfg.DATA.NUM_TASKS
        self.num_outputs = cfg.DATA.NUM_OUTPUTS
        
        ## Model Parameters
        self.cate_transform = cfg.MODEL.CATE_TRANS
        self.split_option = cfg.MODEL.SPLIT
        self.cv = cfg.MODEL.CV
        self.n_cv = cfg.MODEL.N_CV
        self.x_scale=cfg.MODEL.X_SCALE
        self.y_scale=cfg.MODEL.Y_SCALE

        self.y_scaler = []
        if self.y_scale != 'no-y-scale' and self.cv!='none':
            for _ in range(self.n_cv):
                if self.y_scale == 'y-minmax':
                    y_scaler = MinMaxScaler()
                elif self.y_scale == 'y-stand':
                    y_scaler = StandardScaler()
                elif self.y_scale == 'y-minmax':
                    y_scaler = MinMaxScaler()
                self.y_scaler.append(y_scaler)
        if self.y_scale != 'no-y-scale' and self.cv=='none':
            if self.y_scale == 'y-minmax':
                    self.y_scaler = MinMaxScaler()
            elif self.y_scale == 'y-stand':
                    self.y_scaler = StandardScaler()
            elif self.y_scale == 'y-minmax':
                    self.y_scaler = MinMaxScaler()


        ##path
        self.date = cfg.MISC.DATE
        self.data_path = cfg.PATH.DATA_PATH
        #self.output_path = cfg.PATH.RESULT
        self.output_path = '/'.join([cfg.PATH.RESULT,self.Data_name,self.model_name,self.split_option,self.date,'num_tasks_'+str(self.num_tasks)])
        if not os.path.exists(self.output_path):
            print(f'Creating folder {self.output_path}')
            os.makedirs(self.output_path,exist_ok = True)
        



    def Load_data_set(self):
        if self.Data_name == 'DPS':
            X_all_outputs_with_replicates, Y_list = self.dps_dataset()
            return X_all_outputs_with_replicates, Y_list
        elif self.Data_name == 'Viscosity':
            X_all_outputs_with_replicates, Y_list = self.Viscosity_dataset()
            return X_all_outputs_with_replicates, Y_list
        elif self.Data_name == 'Harpoon':
            #df_x, df_y_wide, df_y_long = self.harpoon_dataset()
            df_x, df_y = self.harpoon_dataset()
            if self.cv == 'none':
                #return df_x, df_y_wide, df_y_long
                return df_x, df_y
            elif self.cv == 'kfold':
                kfold = KFold(n_splits=self.n_cv, shuffle=True,random_state=0)

                ls_X_train = []
                ls_X_test = []
                ls_y_train = []
                ls_y_test = []

                X = df_x.to_numpy()
                Y = df_y.to_numpy()

                count = 0
                for train_ix, test_ix in kfold.split(X):
                    X_train, X_test=X[train_ix,:], X[test_ix,:]
                    y_train, y_test= Y[train_ix],Y[test_ix]

                    df_X_train = pd.DataFrame(X_train,columns = df_x.columns)
                    df_X_test = pd.DataFrame(X_test,columns = df_x.columns)

                    df_y_train = pd.DataFrame(y_train,columns = df_y.columns)
                    df_y_test = pd.DataFrame(y_test,columns = df_y.columns)

                    if self.x_scale != 'no-x-scale':
                        if self.x_scale != 'no-x-scale':
                            if self.x_scale == 'x-minmax':
                                x_scaler = MinMaxScaler()
                            elif self.x_scale == 'x-stand':
                                x_scaler = StandardScaler()
                        xct = ColumnTransformer([('x_scaler',x_scaler,df_X_train.columns.difference(['task_ind']))], 
                            remainder = 'passthrough')
                        scaled_X_train=xct.fit_transform(df_X_train)
                        scaled_X_test=xct.transform(df_X_test)
                    else:
                        scaled_X_train= X_train
                        scaled_X_test=X_test

                    if self.y_scale != 'no-y-scale':
                        scaled_y_train = self.y_scaler[count].fit_transform(df_y_train)
                    else:
                        scaled_y_train = df_y_train.to_numpy()

                    scaled_y_test = df_y_test.to_numpy()

                    ls_X_train.append(torch.Tensor(scaled_X_train))
                    ls_X_test.append(torch.Tensor(scaled_X_test))
                    ls_y_train.append(torch.Tensor(scaled_y_train))
                    ls_y_test.append(torch.Tensor(scaled_y_test))
                    count +=1
                return df_x, df_y,ls_X_train,ls_X_test,ls_y_train,ls_y_test,self.y_scaler
    
    def harpoon_dataset(self):
        data_path_x = self.data_path+'/harpoon_df_x_24.csv'
        data_path_y = self.data_path+'/harpoon_df_y_24.csv'
        df_x = pd.read_csv(data_path_x,usecols = [1,2,3,4])
        #df_x = pd.read_excel(data_path,sheet_name = 0, usecols = [1,2,3,4])
        df_x.columns = [re.sub('[^A-Za-z0-9Δ]+', '_', element) for element in df_x.columns]

        df_y_1 = pd.read_csv(data_path_y,usecols = [1,4,7],skiprows = lambda x: x in [1])
        df_y_1.columns = [re.sub('[^A-Za-z0-9Δ]+', '_', element) for element in df_y_1.columns]
        

        df_y_2 = pd.read_csv(data_path_y,usecols = [2,5,8],skiprows = lambda x: x in [1])
        df_y_2.columns = df_y_1.columns
        

        df_y_3 = pd.read_csv(data_path_y,usecols = [3,6,9],skiprows = lambda x: x in [1])
        df_y_3.columns = df_y_1.columns

        
        cols_target = list(df_y_1.columns)[0:2]
        cols_cate = ['Buffer_Type',
            'Sugar_Salt',
            'Additive']
        
        if self.num_tasks == 3:
            total_df_y_long = pd.concat((df_y_1[cols_target],df_y_2[cols_target],df_y_3[cols_target]),axis = 0)
        

            df_y_1.columns = [c+'_1' for c in df_y_1.columns]
            df_y_2.columns = [c+'_2' for c in df_y_2.columns]
            df_y_3.columns = [c+'_3' for c in df_y_3.columns]

            total_df_y_wide = pd.concat((df_y_1[['_Dimer_HMW__1','_Monomer_1']],
                        df_y_2[['_Dimer_HMW__2','_Monomer_2']],
                        df_y_3[['_Dimer_HMW__3','_Monomer_3']]),axis = 1)
        elif self.num_tasks == 2:
            total_df_y_long = pd.concat((df_y_1[cols_target],df_y_3[cols_target]),axis = 0)
        

            df_y_1.columns = [c+'_1' for c in df_y_1.columns]
            df_y_3.columns = [c+'_3' for c in df_y_3.columns]

            total_df_y_wide = pd.concat((df_y_1[['_Dimer_HMW__1','_Monomer_1']],
                        df_y_3[['_Dimer_HMW__3','_Monomer_3']]),axis = 1)
        
        total_df_y_long.reset_index(inplace = True, drop = True)

        print(self.cate_transform)
        df_x_cate_trans = df_x.copy()
        if self.cate_transform =='label':
            #convert categorical columns to labels
            for x_name in cols_cate:
                labels, categories = factorize(df_x[x_name])
                df_x_cate_trans[x_name+"_label"] = labels
            df_x_cate_trans.drop(cols_cate,axis = 1,inplace = True)
            #display(df_x.head())
        
        #onehot encoding
        elif self.cate_transform =='ohe':
            df_x_cate_trans = pd.get_dummies(df_x, columns = cols_cate)
            df_x_cate_trans.columns = [re.sub('[^A-Za-z0-9Δ]+', '_', element) for element in df_x_cate_trans.columns]
            #convert boolean True False to 0 1
            bool_cols = df_x_cate_trans.select_dtypes(include='bool').columns
            df_x_cate_trans[bool_cols] = df_x_cate_trans[bool_cols].astype(int)

        #cols_feature_new = list(df_x_cate_trans.columns)
        #whether to treat each task separately
        if self.split_option == 'combine' or self.model_name == 'MTMO' or self.model_name == 'MTMO-LMGP':
            ls_X = []
            for i in range(self.num_tasks):
                df_x_copy = df_x_cate_trans.copy()
                df_x_copy['task_ind'] = i
                ls_X.append(df_x_copy)

            df_X = pd.concat(ls_X)
            df_X.reset_index(inplace=True, drop = True)
            df_Y = total_df_y_long
            #print(df_X['task_ind'].unique())

        else:
            df_X = df_x_cate_trans
            df_Y = total_df_y_wide

        #return df_X, total_df_y_wide,total_df_y_long
        return df_X, df_Y