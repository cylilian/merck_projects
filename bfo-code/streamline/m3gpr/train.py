
import itertools
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import gpytorch
from m3gpr.models import *
from gpytorch.constraints import GreaterThan
#from m3gpr.models import singleGP_gpytorch,rf_sklearn,singleGP_gpytorch_train,singleGP_gpytorch_reference,MultitaskGPModel,MTMOGP

class CV_Trainer(object):
    def __init__(self,cfg,ls_X_train,ls_X_test,ls_y_train,ls_y_test,obj_y_scaler,cov):
        self.model_name = cfg.MODEL.MODEL_NAME
        self.n_cv = cfg.MODEL.N_CV#cfg.MODEL.N_CV is number of cv
        self.num_outputs = cfg.DATA.NUM_OUTPUTS
        if cfg.MODEL.SPLIT == 'by-task':
            self.num_tasks = cfg.DATA.NUM_TASKS
        else:
            self.num_tasks = 1
        self.num_total_output = self.num_outputs*self.num_tasks
        self.num_cols = self.num_total_output*self.n_cv

        self.y_scale=cfg.MODEL.Y_SCALE
        self.obj_y_scaler = obj_y_scaler
        self.ls_X_train = ls_X_train
        self.ls_X_test = ls_X_test
        self.ls_y_train = ls_y_train
        self.ls_y_test = ls_y_test
        self.err_metric = cfg.MODEL.METRIC
        if self.err_metric == 'MAE':
            self.metric_row = 0
        elif self.err_metric == 'RMSE':
            self.metric_row = 3
        elif self.err_metric == 'NMSE':
            self.metric_row = 2

        self.link_rank =cfg.MODEL.LIK_RANK
        self.cov = cov
    def set_up_cv_model(self):
        if self.model_name == 'singleGP':
            combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err = self.set_up_sgp_cv()
        elif self.model_name == 'MO':
            combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err = self.set_up_mo_cv()
        elif self.model_name == 'MTMO':
            combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err = self.set_up_mtmo_cv()
        elif self.model_name == 'MTMO-LMGP':
            combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err = self.set_up_mtmolmgp_cv()
        return combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err
    
    def set_up_sgp_cv(self):
        ls_model_from_combs = [[] for _ in range(self.num_total_output)]
        ls_init_len = [3.0]#[3.0]#[2.0,3.0,4.0]
        ls_lr = [0.25]#[0.25]#[0.1,0.15,0.2,0.25]
        ls_n_iter = [300]#[300]#[200,300]
        combs = list(itertools.product(ls_init_len,ls_lr,ls_n_iter))

        mean_arr_cv_mae = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_r2 = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_err = np.zeros((2,len(combs))) #train,test

        ls_arr_cv_mae = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_r2 = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_err = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]

        for i in range(0,len(combs)):
            init_len,lr,n_iter = combs[i]
            arr_train_metrics = np.zeros((11,self.num_cols))
            arr_test_metrics = np.zeros((11,self.num_cols))
            
            for batch_ind in range(self.n_cv):
                t_train_x = self.ls_X_train[batch_ind]
                t_test_x = self.ls_X_test[batch_ind]
                
                #for j in range(self.num_outputs):
                for j in range(self.num_total_output):
                    a = batch_ind + self.n_cv*j
                    t_train_y = self.ls_y_train[batch_ind][:,j].flatten()
                    t_test_y = self.ls_y_test[batch_ind][:,j].flatten()

                    train_mean,train_lower,train_upper,model = singleGP_gpytorch_train(t_train_x,t_train_y,training_iter = n_iter,init_len_scale = init_len,my_lr = lr)
                    #print(train_mean)
                    ls_model_from_combs[j].append(model)
                    
                    # Set into eval mode
                    model.eval()
                    model.likelihood.eval()

                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        test_pred = model.likelihood(model(t_test_x))   
                        test_mean = test_pred.mean
                        test_lower, test_upper = test_pred.confidence_region()

                    if self.y_scale=='no-y-scale':
                        arr_train_y = t_train_y.detach().numpy()
                        arr_test_y = t_test_y.detach().numpy()
                        arr_test_mean = test_mean.detach().numpy()
                        arr_test_lower = test_lower.detach().numpy()
                        arr_test_upper = test_upper.detach().numpy()

                        arr_train_mean = train_mean.detach().numpy()
                        arr_train_lower = train_lower.detach().numpy()
                        arr_train_upper = train_upper.detach().numpy()
                    else:
                        y_scaler = self.obj_y_scaler[batch_ind]
                        tmp_mean = y_scaler.mean_
                        tmp_scale = y_scaler.scale_

                        arr_train_y = t_train_y.detach().numpy()
                        arr_test_y = t_test_y.detach().numpy()
                        arr_test_mean = test_mean.detach().numpy()
                        arr_test_lower = test_lower.detach().numpy()
                        arr_train_mean = train_mean.detach().numpy()
                        arr_train_lower = train_lower.detach().numpy()

                        arr_train_y = arr_train_y*tmp_scale[j] + tmp_mean[j]
                        arr_test_mean = arr_test_mean*tmp_scale[j] + tmp_mean[j]
                        arr_test_lower = arr_test_lower*tmp_scale[j] + tmp_mean[j]
                        arr_train_mean = arr_train_mean*tmp_scale[j] + tmp_mean[j]
                        arr_train_lower = arr_train_lower*tmp_scale[j] + tmp_mean[j]


                        """
                        arr_train_y = y_scaler.inverse_transform(t_train_y)
                        arr_test_y = t_test_y.detach().numpy()

                        arr_train_mean = y_scaler.inverse_transform(train_mean)
                        arr_train_lower = y_scaler.inverse_transform(train_lower)
                        arr_train_upper = y_scaler.inverse_transform(train_upper)

                        arr_test_mean = y_scaler.inverse_transform(test_mean)
                        arr_test_lower = y_scaler.inverse_transform(test_lower)
                        arr_test_upper = y_scaler.inverse_transform(test_upper)
                        """

                    y_true = arr_test_y
                    y_pred = arr_test_mean
                    y_train = arr_train_y
                    mean_train = arr_train_mean
                    std_train = arr_train_mean - arr_train_lower
                    std_test = arr_test_mean - arr_test_lower

                    train_comp = np.concatenate((y_train.reshape(-1,1),mean_train.reshape(-1,1)),axis = 1)
                    df_train_comp = pd.DataFrame(train_comp,columns = ['true','pred'])
                    df_train_comp['upper'] = df_train_comp['pred'] + std_train.squeeze()
                    df_train_comp['lower'] = df_train_comp['pred'] - std_train.squeeze()
                    df_train_comp['mode'] = 'train'
    
                    test_comp = np.concatenate((y_true.reshape(-1,1),y_pred.reshape(-1,1)),axis = 1)
                    df_test_comp = pd.DataFrame(test_comp,columns = ['true','pred'])
                    df_test_comp['upper'] = df_test_comp['pred'] + std_test.squeeze()
                    df_test_comp['lower'] = df_test_comp['pred'] - std_test.squeeze()
                    df_test_comp['mode'] = 'test'
    

                    arr_test_metrics[0,a] = metrics.mean_absolute_error(y_true, y_pred)
                    arr_test_metrics[1,a] = metrics.median_absolute_error(y_true, y_pred)
                    arr_test_metrics[2,a] = metrics.mean_squared_error(y_true, y_pred)/np.var(y_true, ddof=0)#Normalized MSE
                    arr_test_metrics[3,a] = metrics.root_mean_squared_error(y_true, y_pred)
                    arr_test_metrics[4,a] = metrics.mean_absolute_percentage_error(y_true, y_pred)
                    arr_test_metrics[5,a] = metrics.max_error(y_true, y_pred)
                    arr_test_metrics[6,a] = metrics.explained_variance_score(y_true, y_pred)
                    arr_test_metrics[7,a] = metrics.r2_score(y_true, y_pred)
                    arr_test_metrics[8,a] = np.mean(std_test)
                    arr_test_metrics[9,a] = np.min(std_test)
                    arr_test_metrics[10,a] = np.max(std_test)

                    arr_train_metrics[0,a] = metrics.mean_absolute_error(y_train, mean_train)
                    arr_train_metrics[1,a] = metrics.median_absolute_error(y_train, mean_train)
                    arr_train_metrics[2,a] = metrics.mean_squared_error(y_train, mean_train)/np.var(y_train, ddof=0)#Normalized MSE
                    arr_train_metrics[3,a] = metrics.root_mean_squared_error(y_train, mean_train)
                    arr_train_metrics[4,a] = metrics.mean_absolute_percentage_error(y_train, mean_train)
                    arr_train_metrics[5,a] = metrics.max_error(y_train, mean_train)
                    arr_train_metrics[6,a] = metrics.explained_variance_score(y_train, mean_train)
                    arr_train_metrics[7,a] = metrics.r2_score(y_train, mean_train)
                    arr_train_metrics[8,a] = np.mean(std_train)
                    arr_train_metrics[9,a] = np.min(std_train)
                    arr_train_metrics[10,a] = np.max(std_train)
                    arr_train_metrics[8,a] = np.mean(std_train)
                    arr_train_metrics[9,a] = np.min(std_train)
                    arr_train_metrics[10,a] = np.max(std_train)
      
                    ls_arr_cv_mae[j][0,i] = np.mean(arr_train_metrics[self.metric_row,self.n_cv*j:self.n_cv*(j+1)])
                    ls_arr_cv_mae[j][1,i] = np.mean(arr_test_metrics[self.metric_row,self.n_cv*j:self.n_cv*(j+1)])

                    ls_arr_cv_r2[j][0,i] = np.mean(arr_train_metrics[6,self.n_cv*j:self.n_cv*(j+1)])
                    ls_arr_cv_r2[j][1,i] = np.mean(arr_test_metrics[6,self.n_cv*j:self.n_cv*(j+1)])

                    ls_arr_cv_err[j][0,i] = np.mean(arr_train_metrics[8,self.n_cv*j:self.n_cv*(j+1)])
                    ls_arr_cv_err[j][1,i] = np.mean(arr_test_metrics[8,self.n_cv*j:self.n_cv*(j+1)])

            mean_arr_cv_mae[0,i] = np.mean(arr_train_metrics[self.metric_row,:])
            mean_arr_cv_mae[1,i] = np.mean(arr_test_metrics[self.metric_row,:])

            mean_arr_cv_r2[0,i] = np.mean(arr_train_metrics[6,:])
            mean_arr_cv_r2[1,i] = np.mean(arr_test_metrics[6,:])

            mean_arr_cv_err[0,i] = np.mean(arr_train_metrics[8,:])
            mean_arr_cv_err[1,i] = np.mean(arr_test_metrics[8,:])


        return combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err
    
    def set_up_mo_cv(self):
        ls_model_from_combs = []

        ls_lr = [0.1]#[0.25,0.35,0.45]#[0.1,0.15,0.2,0.25,0.3,0.35,0.40]
        ls_n_iter = [150]#[500,700,900]#[200,300,400,500,700]
        combs = list(itertools.product(ls_lr,ls_n_iter))

        mean_arr_cv_mae = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_r2 = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_err = np.zeros((2,len(combs))) #train,test

        ls_arr_cv_mae = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_r2 = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_err = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]

        for i in range(0,len(combs)):
            lr,n_iter = combs[i]
            arr_train_metrics = np.zeros((11,self.num_cols))
            arr_test_metrics = np.zeros((11,self.num_cols))
            pre_model = None
            for batch_ind in range(self.n_cv):
                t_train_x = self.ls_X_train[batch_ind]
                t_train_y = self.ls_y_train[batch_ind]
                t_test_x = self.ls_X_test[batch_ind]
                t_test_y = self.ls_y_test[batch_ind]

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_total_output,rank = self.link_rank)
                """
                # Add a constraint to ensure variance is >= 1e-4
                likelihood.task_covar_module.register_constraint(
                    "raw_var", GreaterThan(1e-4)
                )
                """
                model = MultitaskGPModel(t_train_x, t_train_y, likelihood, num_tasks= self.num_total_output, rank = self.num_outputs)
                """
                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=self.num_outputs, 
                    rank=self.link_rank if self.link_rank is not None else 0
                )
                model = MOGP(t_train_x, t_train_y, likelihood, output_rank = self.num_outputs)
                
                """

                """
                if batch_ind>0:
                    source_state_dict = pre_model.state_dict()
                    model.load_state_dict(source_state_dict)
                    #model.covar_module.base_kernel.lengthscale = pre_model.covar_module.base_kernel.lengthscale
                    #model.likelihood.noise = pre_model.likelihood.noise
                # Find optimal model hyperparameters
                """
                model.train()
                likelihood.train()
                
                all_params = set(model.parameters())
                frozen_params = {model.covar_module.task_covar_module.covar_factor}
                optimizable_params = list(all_params - frozen_params)
                optimizer = torch.optim.Adam(optimizable_params, lr=lr)  # Includes GaussianLikelihood parameters
                
                # Use the adam optimizer
                #optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                training_iterations = n_iter#500
                loss_list = []
                for _ in range(training_iterations):
                    optimizer.zero_grad()
                    with gpytorch.settings.cholesky_jitter(1e-5):
                        output = model(t_train_x)
                        #print('output.shape',output.shape)
                        #print('t_train_y.shape',t_train_y.shape)
                        loss = -mll(output, t_train_y)
                        loss_list.append(loss.item())
                        loss.backward()
                        #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                        optimizer.step()
                
                plt.plot(loss_list)
                plt.title('Neg. Loss', fontsize='small')

                ls_model_from_combs.append(model)
                if batch_ind == 0:
                    pre_model = model

                # Set into eval mode
                model.eval()
                likelihood.eval()

                

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_pred = likelihood(model(t_test_x))   
                    test_mean = test_pred.mean
                    test_lower, test_upper = test_pred.confidence_region()
                    train_pred = likelihood(model(t_train_x))
                    train_mean = train_pred.mean
                    train_lower, train_upper = train_pred.confidence_region()
    
                if self.y_scale=='no-y-scale':
                    arr_train_y = t_train_y.detach().numpy()
                    arr_test_y = t_test_y.detach().numpy()
                    arr_test_mean = test_mean.detach().numpy()
                    arr_test_lower = test_lower.detach().numpy()
                    arr_test_upper = test_upper.detach().numpy()

                    arr_train_mean = train_mean.detach().numpy()
                    arr_train_lower = train_lower.detach().numpy()
                    arr_train_upper = train_upper.detach().numpy()
                else:
                    y_scaler = self.obj_y_scaler[batch_ind]
                    arr_train_y = y_scaler.inverse_transform(t_train_y)
                    arr_test_y = t_test_y.detach().numpy()

                    arr_train_mean = y_scaler.inverse_transform(train_mean)
                    arr_train_lower = y_scaler.inverse_transform(train_lower)
                    arr_train_upper = y_scaler.inverse_transform(train_upper)

                    arr_test_mean = y_scaler.inverse_transform(test_mean)
                    arr_test_lower = y_scaler.inverse_transform(test_lower)
                    arr_test_upper = y_scaler.inverse_transform(test_upper)

                #for j in range(self.num_outputs):
                for j in range(self.num_total_output):
                    a = batch_ind + self.n_cv*j

                    r_train_y = arr_train_y[:,j]
                    r_train_mean_y =arr_train_mean[:,j]
                    r_train_lower_y =arr_train_lower[:,j]
                    r_train_upper_y =arr_train_upper[:,j]
                    r_train_std = r_train_mean_y - r_train_lower_y

                    r_test_y = arr_test_y[:,j]
                    r_test_mean_y =arr_test_mean[:,j]
                    r_test_lower_y =arr_test_lower[:,j]
                    r_test_upper_y =arr_test_upper[:,j]
                    r_test_std = r_test_mean_y - r_test_lower_y
    
                    train_comp = np.concatenate((r_train_y.reshape(-1,1),r_train_mean_y.reshape(-1,1)),axis = 1)
                    df_train_comp = pd.DataFrame(train_comp,columns = ['true','pred'])
                    df_train_comp['upper'] = r_train_upper_y
                    df_train_comp['lower'] = r_train_lower_y
                    df_train_comp['mode'] = 'train'
    
                    test_comp = np.concatenate((r_test_y.reshape(-1,1),r_test_mean_y.reshape(-1,1)),axis = 1)
                    df_test_comp = pd.DataFrame(test_comp,columns = ['true','pred'])
                    df_test_comp['upper'] = r_test_upper_y
                    df_test_comp['lower'] = r_test_lower_y
                    df_test_comp['mode'] = 'test'

                    y_true = r_test_y
                    y_pred = r_test_mean_y
                    y_train = r_train_y
                    mean_train = r_train_mean_y
    

                    arr_test_metrics[0,a] = metrics.mean_absolute_error(y_true, y_pred)
                    arr_test_metrics[1,a] = metrics.median_absolute_error(y_true, y_pred)
                    arr_test_metrics[2,a] = metrics.mean_squared_error(y_true, y_pred)/np.var(y_true, ddof=0)#Normalized MSE
                    arr_test_metrics[3,a] = metrics.root_mean_squared_error(y_true, y_pred)
                    arr_test_metrics[4,a] = metrics.mean_absolute_percentage_error(y_true, y_pred)
                    arr_test_metrics[5,a] = metrics.max_error(y_true, y_pred)
                    arr_test_metrics[6,a] = metrics.explained_variance_score(y_true, y_pred)
                    arr_test_metrics[7,a] = metrics.r2_score(y_true, y_pred)
                    arr_test_metrics[8,a] = np.mean(r_test_std)
                    arr_test_metrics[9,a] = np.min(r_test_std)
                    arr_test_metrics[10,a] = np.max(r_test_std)

                    arr_train_metrics[0,a] = metrics.mean_absolute_error(y_train, mean_train)
                    arr_train_metrics[1,a] = metrics.median_absolute_error(y_train, mean_train)
                    arr_train_metrics[2,a] = metrics.mean_squared_error(y_train, mean_train)/np.var(y_train, ddof=0)#Normalized MSE
                    arr_train_metrics[3,a] = metrics.root_mean_squared_error(y_train, mean_train)
                    arr_train_metrics[4,a] = metrics.mean_absolute_percentage_error(y_train, mean_train)
                    arr_train_metrics[5,a] = metrics.max_error(y_train, mean_train)
                    arr_train_metrics[6,a] = metrics.explained_variance_score(y_train, mean_train)
                    arr_train_metrics[7,a] = metrics.r2_score(y_train, mean_train)
                    arr_train_metrics[8,a] = np.mean(r_train_std)
                    arr_train_metrics[9,a] = np.min(r_train_std)
                    arr_train_metrics[10,a] = np.max(r_train_std)
                    arr_train_metrics[8,a] = np.mean(r_train_std)
                    arr_train_metrics[9,a] = np.min(r_train_std)
                    arr_train_metrics[10,a] = np.max(r_train_std)
      
                    ls_arr_cv_mae[j][0,i] = np.mean(arr_train_metrics[self.metric_row,self.n_cv*j:self.n_cv*(j+1)])
                    ls_arr_cv_mae[j][1,i] = np.mean(arr_test_metrics[self.metric_row,self.n_cv*j:self.n_cv*(j+1)])

                    ls_arr_cv_r2[j][0,i] = np.mean(arr_train_metrics[6,self.n_cv*j:self.n_cv*(j+1)])
                    ls_arr_cv_r2[j][1,i] = np.mean(arr_test_metrics[6,self.n_cv*j:self.n_cv*(j+1)])

                    ls_arr_cv_err[j][0,i] = np.mean(arr_train_metrics[8,self.n_cv*j:self.n_cv*(j+1)])
                    ls_arr_cv_err[j][1,i] = np.mean(arr_test_metrics[8,self.n_cv*j:self.n_cv*(j+1)])

            mean_arr_cv_mae[0,i] = np.mean(arr_train_metrics[self.metric_row,:])
            mean_arr_cv_mae[1,i] = np.mean(arr_test_metrics[self.metric_row,:])

            mean_arr_cv_r2[0,i] = np.mean(arr_train_metrics[6,:])
            mean_arr_cv_r2[1,i] = np.mean(arr_test_metrics[6,:])

            mean_arr_cv_err[0,i] = np.mean(arr_train_metrics[8,:])
            mean_arr_cv_err[1,i] = np.mean(arr_test_metrics[8,:])


        return combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err

    def set_up_mtmo_cv(self):
        ls_model_from_combs = []

        ls_lr = [0.01]#[0.1,0.15,0.2,0.25,0.3,0.35,0.40]#[0.1,0.15,0.25]
        ls_n_iter = [1000]#[200,300,400,500,700]#[500,700,900,1100]
        combs = list(itertools.product(ls_lr,ls_n_iter))

        mean_arr_cv_mae = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_r2 = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_err = np.zeros((2,len(combs))) #train,test

        ls_arr_cv_mae = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_r2 = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_err = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]

        for i in range(0,len(combs)):
            print('-------Combination',str(i))
            lr,n_iter = combs[i]
            arr_train_metrics = np.zeros((11,self.num_cols))
            arr_test_metrics = np.zeros((11,self.num_cols))
            pre_model = None
            for batch_ind in range(self.n_cv):
                print('-------Batch',str(batch_ind))
                t_train_x = self.ls_X_train[batch_ind]
                t_train_y = self.ls_y_train[batch_ind]
                t_test_x = self.ls_X_test[batch_ind]
                t_test_y = self.ls_y_test[batch_ind]

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=self.num_outputs, 
                    rank=self.link_rank if self.link_rank is not None else 0,
                )
                model = MTMOGP(t_train_x, t_train_y,likelihood,
                               task_rank = self.num_tasks,output_rank = self.num_outputs)

                if batch_ind>0:
                    source_state_dict = pre_model.state_dict()
                    model.load_state_dict(source_state_dict)
                # Find optimal model hyperparameters
                model.train()
                likelihood.train()

                # Use the adam optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                training_iterations = n_iter#500
                for _ in range(training_iterations):
                    optimizer.zero_grad()
                    output = model(t_train_x)
                    loss = -mll(output, t_train_y)
                    loss.backward()
                    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                    optimizer.step()

                ls_model_from_combs.append(model)
                if batch_ind == 0:
                    pre_model = model

                # Set into eval mode
                model.eval()
                likelihood.eval()

                arr_train_x = t_train_x.detach().numpy()
                arr_test_x = t_test_x.detach().numpy()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_pred = likelihood(model(t_test_x))   
                    test_mean = test_pred.mean
                    test_lower, test_upper = test_pred.confidence_region()
                    train_pred = likelihood(model(t_train_x))
                    train_mean = train_pred.mean
                    train_lower, train_upper = train_pred.confidence_region()
    
                if self.y_scale=='no-y-scale':
                    arr_train_y = t_train_y.detach().numpy()
                    arr_test_y = t_test_y.detach().numpy()
                    arr_test_mean = test_mean.detach().numpy()
                    arr_test_lower = test_lower.detach().numpy()
                    arr_test_upper = test_upper.detach().numpy()

                    arr_train_mean = train_mean.detach().numpy()
                    arr_train_lower = train_lower.detach().numpy()
                    arr_train_upper = train_upper.detach().numpy()
                else:
                    y_scaler = self.obj_y_scaler[batch_ind]
                    arr_train_y = y_scaler.inverse_transform(t_train_y)
                    arr_test_y = t_test_y.detach().numpy()

                    arr_train_mean = y_scaler.inverse_transform(train_mean)
                    arr_train_lower = y_scaler.inverse_transform(train_lower)
                    arr_train_upper = y_scaler.inverse_transform(train_upper)

                    arr_test_mean = y_scaler.inverse_transform(test_mean)
                    arr_test_lower = y_scaler.inverse_transform(test_lower)
                    arr_test_upper = y_scaler.inverse_transform(test_upper)

                for task_ind in range(self.num_tasks):
                    for j in range(self.num_outputs):
                        mtmo_ind = j + self.num_outputs*task_ind
                        row_train_inds = arr_train_x[:,-1]==task_ind
                        r_train_y = arr_train_y[row_train_inds,j]
                        r_train_mean_y =arr_train_mean[row_train_inds,j]
                        r_train_lower_y =arr_train_lower[row_train_inds,j]
                        r_train_upper_y =arr_train_upper[row_train_inds,j]
                        r_train_std = r_train_mean_y - r_train_lower_y

                        row_test_inds = arr_test_x[:,-1]==task_ind
                        r_test_y = arr_test_y[row_test_inds,j]
                        r_test_mean_y =arr_test_mean[row_test_inds,j]
                        r_test_lower_y =arr_test_lower[row_test_inds,j]
                        r_test_upper_y =arr_test_upper[row_test_inds,j]
                        r_test_std = r_test_mean_y - r_test_lower_y
    
                        train_comp = np.concatenate((r_train_y.reshape(-1,1),r_train_mean_y.reshape(-1,1)),axis = 1)
                        df_train_comp = pd.DataFrame(train_comp,columns = ['true','pred'])
                        df_train_comp['upper'] = r_train_upper_y
                        df_train_comp['lower'] = r_train_lower_y
                        df_train_comp['mode'] = 'train'
    
                        test_comp = np.concatenate((r_test_y.reshape(-1,1),r_test_mean_y.reshape(-1,1)),axis = 1)
                        df_test_comp = pd.DataFrame(test_comp,columns = ['true','pred'])
                        df_test_comp['upper'] = r_test_upper_y
                        df_test_comp['lower'] = r_test_lower_y
                        df_test_comp['mode'] = 'test'

                        y_true = r_test_y
                        y_pred = r_test_mean_y
                        y_train = r_train_y
                        mean_train = r_train_mean_y
    
                        a = batch_ind + self.n_cv*mtmo_ind
                        #a = mtmo_ind + self.num_total_output*batch_ind
                        arr_test_metrics[0,a] = metrics.mean_absolute_error(y_true, y_pred)
                        arr_test_metrics[1,a] = metrics.median_absolute_error(y_true, y_pred)
                        arr_test_metrics[2,a] = metrics.mean_squared_error(y_true, y_pred)/np.var(y_true, ddof=0)#Normalized MSE
                        arr_test_metrics[3,a] = metrics.root_mean_squared_error(y_true, y_pred)
                        arr_test_metrics[4,a] = metrics.mean_absolute_percentage_error(y_true, y_pred)
                        arr_test_metrics[5,a] = metrics.max_error(y_true, y_pred)
                        arr_test_metrics[6,a] = metrics.explained_variance_score(y_true, y_pred)
                        arr_test_metrics[7,a] = metrics.r2_score(y_true, y_pred)
                        arr_test_metrics[8,a] = np.mean(r_test_std)
                        arr_test_metrics[9,a] = np.min(r_test_std)
                        arr_test_metrics[10,a] = np.max(r_test_std)

                        arr_train_metrics[0,a] = metrics.mean_absolute_error(y_train, mean_train)
                        arr_train_metrics[1,a] = metrics.median_absolute_error(y_train, mean_train)
                        arr_train_metrics[2,a] = metrics.mean_squared_error(y_train, mean_train)/np.var(y_train, ddof=0)#Normalized MSE
                        arr_train_metrics[3,a] = metrics.root_mean_squared_error(y_train, mean_train)
                        arr_train_metrics[4,a] = metrics.mean_absolute_percentage_error(y_train, mean_train)
                        arr_train_metrics[5,a] = metrics.max_error(y_train, mean_train)
                        arr_train_metrics[6,a] = metrics.explained_variance_score(y_train, mean_train)
                        arr_train_metrics[7,a] = metrics.r2_score(y_train, mean_train)
                        arr_train_metrics[8,a] = np.mean(r_train_std)
                        arr_train_metrics[9,a] = np.min(r_train_std)
                        arr_train_metrics[10,a] = np.max(r_train_std)
                        arr_train_metrics[8,a] = np.mean(r_train_std)
                        arr_train_metrics[9,a] = np.min(r_train_std)
                        arr_train_metrics[10,a] = np.max(r_train_std)
      
                        ls_arr_cv_mae[mtmo_ind][0,i] = np.mean(arr_train_metrics[self.metric_row,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])
                        ls_arr_cv_mae[mtmo_ind][1,i] = np.mean(arr_test_metrics[self.metric_row,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])

                        ls_arr_cv_r2[mtmo_ind][0,i] = np.mean(arr_train_metrics[6,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])
                        ls_arr_cv_r2[mtmo_ind][1,i] = np.mean(arr_test_metrics[6,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])

                        ls_arr_cv_err[mtmo_ind][0,i] = np.mean(arr_train_metrics[8,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])
                        ls_arr_cv_err[mtmo_ind][1,i] = np.mean(arr_test_metrics[8,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])

            mean_arr_cv_mae[0,i] = np.mean(arr_train_metrics[self.metric_row,:])
            mean_arr_cv_mae[1,i] = np.mean(arr_test_metrics[self.metric_row,:])

            mean_arr_cv_r2[0,i] = np.mean(arr_train_metrics[6,:])
            mean_arr_cv_r2[1,i] = np.mean(arr_test_metrics[6,:])

            mean_arr_cv_err[0,i] = np.mean(arr_train_metrics[8,:])
            mean_arr_cv_err[1,i] = np.mean(arr_test_metrics[8,:])


        return combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err
    
    def set_up_mtmolmgp_cv(self):
        qual_ind_lev = {1: 4, 2:2, 3:2} #2nd, 3rd, and 4th columns are categorical vars. col index: number of levels
        ls_model_from_combs = []

        ls_lr = [0.01,0.02]
        ls_n_iter = [100]
        combs = list(itertools.product(ls_lr,ls_n_iter))

        mean_arr_cv_mae = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_r2 = np.zeros((2,len(combs))) #train,test
        mean_arr_cv_err = np.zeros((2,len(combs))) #train,test

        ls_arr_cv_mae = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_r2 = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]
        ls_arr_cv_err = [np.zeros((2,len(combs))) for _ in range(self.num_total_output)]

        for i in range(0,len(combs)):
            lr,n_iter = combs[i]
            arr_train_metrics = np.zeros((11,self.num_cols))
            arr_test_metrics = np.zeros((11,self.num_cols))
            pre_model = None
            for batch_ind in range(self.n_cv):
                t_train_x = self.ls_X_train[batch_ind]
                t_train_y = self.ls_y_train[batch_ind]
                t_test_x = self.ls_X_test[batch_ind]
                t_test_y = self.ls_y_test[batch_ind]

                model = GP_Lmgp(t_train_x, t_train_y, qual_ind_lev=qual_ind_lev,quant_index = [0],
                   quant_correlation_class= 'RBFKernel',is_mix_reduce= False,
                   lik_rank= self.link_rank,lr = lr,n_iter = n_iter)

                if batch_ind>0:
                    source_state_dict = pre_model.state_dict()
                    model.load_state_dict(source_state_dict)
                # Find optimal model hyperparameters
                model.fit(optim_type='adam_torch',num_restarts = 1)

                ls_model_from_combs.append(model)
                if batch_ind == 0:
                    pre_model = model

                # Set into eval mode
                model.eval()
                model.likelihood.eval()

                arr_train_x = t_train_x.detach().numpy()
                arr_test_x = t_test_x.detach().numpy()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_pred = model.likelihood(model(t_test_x))   
                    test_mean = test_pred.mean
                    test_lower, test_upper = test_pred.confidence_region()
                    train_pred = model.likelihood(model(t_train_x))
                    train_mean = train_pred.mean
                    train_lower, train_upper = train_pred.confidence_region()
                    #transformed categorical value
                    x_test_cate_zeta = model.transform_categorical(t_test_x[:,model.qual_kernel_columns[0]].clone().type(torch.int64).to('cpu'),
                                                  perm_dict = model.perm_dict[0], zeta = model.zeta[0])
                    x_test_cate_latent = model.A_matrix[0](x_test_cate_zeta.float().to('cpu'))
                    x_test_new= torch.cat([x_test_cate_latent,t_test_x[...,model.quant_index.long()].to(**model.tkwargs)],dim=-1)
        

                    x_train_cate_zeta = model.transform_categorical(t_train_x[:,model.qual_kernel_columns[0]].clone().type(torch.int64).to('cpu'),
                                                  perm_dict = model.perm_dict[0], zeta = model.zeta[0])
                    x_train_cate_latent = model.A_matrix[0](x_train_cate_zeta.float().to('cpu'))
                    x_train_new= torch.cat([x_train_cate_latent,t_train_x[...,model.quant_index.long()].to(**model.tkwargs)],dim=-1)
                    if model.is_mix_reduce:
                        x_test_gplvm = model.W_matrix(x_test_new.float().to('cpu'))
                        x_train_gplvm = model.W_matrix(x_train_new.float().to('cpu'))
    
                if self.y_scale=='no-y-scale':
                    arr_train_y = t_train_y.detach().numpy()
                    arr_test_y = t_test_y.detach().numpy()
                    arr_test_mean = test_mean.detach().numpy()
                    arr_test_lower = test_lower.detach().numpy()
                    arr_test_upper = test_upper.detach().numpy()

                    arr_train_mean = train_mean.detach().numpy()
                    arr_train_lower = train_lower.detach().numpy()
                    arr_train_upper = train_upper.detach().numpy()
                else:
                    y_scaler = self.obj_y_scaler[batch_ind]
                    arr_train_y = y_scaler.inverse_transform(t_train_y)
                    arr_test_y = t_test_y.detach().numpy()

                    arr_train_mean = y_scaler.inverse_transform(train_mean)
                    arr_train_lower = y_scaler.inverse_transform(train_lower)
                    arr_train_upper = y_scaler.inverse_transform(train_upper)

                    arr_test_mean = y_scaler.inverse_transform(test_mean)
                    arr_test_lower = y_scaler.inverse_transform(test_lower)
                    arr_test_upper = y_scaler.inverse_transform(test_upper)

                for task_ind in range(self.num_tasks):
                    for j in range(self.num_outputs):
                        mtmo_ind = j + self.num_outputs*task_ind
                        row_train_inds = arr_train_x[:,-1]==task_ind
                        r_train_y = arr_train_y[row_train_inds,j]
                        r_train_mean_y =arr_train_mean[row_train_inds,j]
                        r_train_lower_y =arr_train_lower[row_train_inds,j]
                        r_train_upper_y =arr_train_upper[row_train_inds,j]
                        r_train_std = r_train_mean_y - r_train_lower_y

                        row_test_inds = arr_test_x[:,-1]==task_ind
                        r_test_y = arr_test_y[row_test_inds,j]
                        r_test_mean_y =arr_test_mean[row_test_inds,j]
                        r_test_lower_y =arr_test_lower[row_test_inds,j]
                        r_test_upper_y =arr_test_upper[row_test_inds,j]
                        r_test_std = r_test_mean_y - r_test_lower_y
    
                        train_comp = np.concatenate((r_train_y.reshape(-1,1),r_train_mean_y.reshape(-1,1)),axis = 1)
                        df_train_comp = pd.DataFrame(train_comp,columns = ['true','pred'])
                        df_train_comp['upper'] = r_train_upper_y
                        df_train_comp['lower'] = r_train_lower_y
                        df_train_comp['mode'] = 'train'
    
                        test_comp = np.concatenate((r_test_y.reshape(-1,1),r_test_mean_y.reshape(-1,1)),axis = 1)
                        df_test_comp = pd.DataFrame(test_comp,columns = ['true','pred'])
                        df_test_comp['upper'] = r_test_upper_y
                        df_test_comp['lower'] = r_test_lower_y
                        df_test_comp['mode'] = 'test'

                        y_true = r_test_y
                        y_pred = r_test_mean_y
                        y_train = r_train_y
                        mean_train = r_train_mean_y
    
                        a = batch_ind + self.n_cv*mtmo_ind
                        #a = mtmo_ind + self.num_total_output*batch_ind
                        arr_test_metrics[0,a] = metrics.mean_absolute_error(y_true, y_pred)
                        arr_test_metrics[1,a] = metrics.median_absolute_error(y_true, y_pred)
                        arr_test_metrics[2,a] = metrics.mean_squared_error(y_true, y_pred)/np.var(y_true, ddof=0)#Normalized MSE
                        arr_test_metrics[3,a] = metrics.root_mean_squared_error(y_true, y_pred)
                        arr_test_metrics[4,a] = metrics.mean_absolute_percentage_error(y_true, y_pred)
                        arr_test_metrics[5,a] = metrics.max_error(y_true, y_pred)
                        arr_test_metrics[6,a] = metrics.explained_variance_score(y_true, y_pred)
                        arr_test_metrics[7,a] = metrics.r2_score(y_true, y_pred)
                        arr_test_metrics[8,a] = np.mean(r_test_std)
                        arr_test_metrics[9,a] = np.min(r_test_std)
                        arr_test_metrics[10,a] = np.max(r_test_std)

                        arr_train_metrics[0,a] = metrics.mean_absolute_error(y_train, mean_train)
                        arr_train_metrics[1,a] = metrics.median_absolute_error(y_train, mean_train)
                        arr_train_metrics[2,a] = metrics.mean_squared_error(y_train, mean_train)/np.var(y_train, ddof=0)#Normalized MSE
                        arr_train_metrics[3,a] = metrics.root_mean_squared_error(y_train, mean_train)
                        arr_train_metrics[4,a] = metrics.mean_absolute_percentage_error(y_train, mean_train)
                        arr_train_metrics[5,a] = metrics.max_error(y_train, mean_train)
                        arr_train_metrics[6,a] = metrics.explained_variance_score(y_train, mean_train)
                        arr_train_metrics[7,a] = metrics.r2_score(y_train, mean_train)
                        arr_train_metrics[8,a] = np.mean(r_train_std)
                        arr_train_metrics[9,a] = np.min(r_train_std)
                        arr_train_metrics[10,a] = np.max(r_train_std)
                        arr_train_metrics[8,a] = np.mean(r_train_std)
                        arr_train_metrics[9,a] = np.min(r_train_std)
                        arr_train_metrics[10,a] = np.max(r_train_std)
      
                        ls_arr_cv_mae[mtmo_ind][0,i] = np.mean(arr_train_metrics[0,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])
                        ls_arr_cv_mae[mtmo_ind][1,i] = np.mean(arr_test_metrics[0,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])

                        ls_arr_cv_r2[mtmo_ind][0,i] = np.mean(arr_train_metrics[6,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])
                        ls_arr_cv_r2[mtmo_ind][1,i] = np.mean(arr_test_metrics[6,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])

                        ls_arr_cv_err[mtmo_ind][0,i] = np.mean(arr_train_metrics[8,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])
                        ls_arr_cv_err[mtmo_ind][1,i] = np.mean(arr_test_metrics[8,self.n_cv*mtmo_ind:self.n_cv*(mtmo_ind+1)])

            mean_arr_cv_mae[0,i] = np.mean(arr_train_metrics[0,:])
            mean_arr_cv_mae[1,i] = np.mean(arr_test_metrics[0,:])

            mean_arr_cv_r2[0,i] = np.mean(arr_train_metrics[6,:])
            mean_arr_cv_r2[1,i] = np.mean(arr_test_metrics[6,:])

            mean_arr_cv_err[0,i] = np.mean(arr_train_metrics[8,:])
            mean_arr_cv_err[1,i] = np.mean(arr_test_metrics[8,:])


        return combs,ls_model_from_combs,ls_arr_cv_mae,ls_arr_cv_r2,ls_arr_cv_err,mean_arr_cv_mae,mean_arr_cv_r2,mean_arr_cv_err