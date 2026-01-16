import math
import torch
import gpytorch
from matplotlib import pyplot as plt
#%matplotlib inline

import os
import pandas as pd
from pandas import factorize
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import seaborn as sns
import re
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import gpytorch

from hackathon_config import get_cfg_defaults_all


def load_input_data(input_path = None,index_col = 0,n_subset = 5000):
    df = pd.read_csv(input_path,index_col=index_col)
    df.columns = [re.sub('[^A-Za-z0-9Δ%]+', '_', element) for element in df.columns]
    df.columns  = [re.sub('%','_PCT',element) for element in df.columns]
    col_names = df.columns
    cols_group = col_names[-23:]
    cols_feature = list(col_names.difference(cols_group))
    arr_task_ohe = df[cols_group].to_numpy()
    arr_task_label = np.where(arr_task_ohe==1)[1]
    df['task_ind'] = arr_task_label
    df_x = df[cols_feature+['task_ind']]
    df_x = df_x.iloc[:n_subset,:]
    return df_x

def load_output_data(output_path = None,index_col = 0,n_subset = 5000):
    df = pd.read_csv(output_path,index_col=index_col)
    df.columns = [re.sub('[^A-Za-z0-9Δ%]+', '_', element) for element in df.columns]
    df.columns  = [re.sub('%','_PCT',element) for element in df.columns]
    df = df.iloc[:n_subset,:]
    return df



def rf_sklearn(X_train,y_train,X_test,y_test,feature_names,tmp_col_y,figPath,is_plot_imp = False):
    rf = RandomForestRegressor(random_state= 10)
    rf.fit(X_train,y_train)
    y_train_pred = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    """
    test_var = metrics.mean_squared_error(y_test, y_pred)
    train_var = metrics.mean_squared_error(y_train, y_train_pred)
    test_std = np.sqrt(test_var)
    train_std = np.sqrt(train_var)
    """

    test_std = np.abs(y_test - y_pred)
    train_std = np.abs(y_train - y_train_pred)
    
    importances=rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    if is_plot_imp:
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title(tmp_col_y + " Feature importances")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        figName = tmp_col_y + 'featureImp.jpg'
        plt.savefig(os.path.join(figPath,figName))
    
    return y_train_pred, train_std, y_pred, test_std, forest_importances


# We will use the simplest form of GP model, exact inference

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def simpleGP_gpytorch(t_train_x,t_test_x,t_train_y,training_iter = 50):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(t_train_x, t_train_y, likelihood)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(t_train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, t_train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    
    # Set into eval mode
    model.eval()
    model.likelihood.eval()


    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_pred = model.likelihood(model(t_test_x))   
        test_mean = test_pred.mean
        test_lower, test_upper = test_pred.confidence_region()
        train_pred = model.likelihood(model(t_train_x))
        train_mean = train_pred.mean
        train_lower, train_upper = train_pred.confidence_region()
    return test_mean, test_lower, test_upper, train_mean, train_lower, train_upper


def mdape(y_true, y_pred):
    """
    Calculate the Median Absolute Percentage Error (MdAPE).

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: MdAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = np.abs((y_true - y_pred) / y_true) * 100
    return np.median(ape)


def train_fun(df_x,df_y,figPath = None,x_scale_label = 'no-x-scale',
              y_scale_label = 'no-y-scale',model_option = None):
    ##Split the data into training and testing sets
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_x, df_y, 
                                                                test_size=0.2, random_state=0,
                                                                stratify=df_x['task_ind'])
    
    xct = ColumnTransformer([('x_mm_scaler',MinMaxScaler(),df_X_train.columns.difference(['task_ind']))], 
                         remainder = 'passthrough')
    #scaled_X_train=x_mm_scaler.fit_transform(df_X_train) 
    #scaled_X_test=x_mm_scaler.transform(df_X_test)

    scaled_X_train=xct.fit_transform(df_X_train) 
    scaled_X_test=xct.transform(df_X_test)

    
    if x_scale_label != 'no-x-scale':
        t_train_x = torch.Tensor(scaled_X_train)
        t_test_x = torch.Tensor(scaled_X_test)
    else:
        t_train_x = torch.Tensor(df_X_train.to_numpy())
        t_test_x = torch.Tensor(df_X_test.to_numpy())
        #t_train_y = torch.Tensor(scaled_y_train).flatten()
    
    #simpleGP and rf
    cols_target = list(df_y.columns)

    #y_plot_scale_label='no-y-scale'

    uncertainty_figure_option = 'errorbar' #shade or errorbar
    is_set_axis_limit = False

    plot_axis_lb = df_y.min(axis = 0) - 0.4*df_y.std(axis = 0)
    plot_axis_ub = df_y.max(axis = 0) + 0.4* df_y.std(axis = 0)

    num_outputs = len(cols_target)
    num_tasks = len(df_x['task_ind'].unique())

    num_total_output = num_outputs*num_tasks
    arr_train_metrics = np.zeros((11,num_total_output))
    arr_test_metrics = np.zeros((11,num_total_output))

    #model_option = cfg.MODEL.MODEL_NAME
    #y_scale_label = cfg.MODEL.Y_SCALE

    # Initialize plots
    f, y_axes = plt.subplots(num_tasks, num_outputs, figsize=(num_outputs*8, num_tasks*7))
    y_axes = y_axes.ravel()
    cols_target_wide = []
    cols_feature_new = df_X_train.columns.difference(['task_ind'])
    for task_ind in range(num_tasks):
        y_train_task = df_y_train.iloc[scaled_X_train[:,-1]==task_ind,:].to_numpy()
        y_test_task = df_y_test.iloc[scaled_X_test[:,-1]==task_ind,:].to_numpy()

        t_train_x_task = t_train_x[t_train_x[:,-1]==task_ind]
        t_test_x_task = t_test_x[t_test_x[:,-1]==task_ind]

        arr_train_x = t_train_x_task.detach().numpy()
        arr_test_x = t_test_x_task.detach().numpy()
        for j,_ in enumerate(cols_target):
            a = j + len(cols_target)*task_ind
            tmp_col_y = 'Task_'+str(task_ind+1) + '_Output_'+str(j+1)
            cols_target_wide.append(tmp_col_y)
            if model_option == 'rf':
                X_train = scaled_X_train[scaled_X_train[:,-1]==task_ind,:-1]
                X_test = scaled_X_test[scaled_X_test[:,-1]==task_ind,:-1]
                y_train = df_y_train.iloc[scaled_X_train[:,-1]==task_ind,j].to_numpy()
                y_true = df_y_test.iloc[scaled_X_test[:,-1]==task_ind,j].to_numpy()
                mean_train_pred, std_train_pred, mean_test_pred, std_test_pred, _, = rf_sklearn(X_train,
                                                                                      y_train,
                                                                                      X_test,
                                                                                      y_true,
                                                                                      cols_feature_new,
                                                                                      tmp_col_y,figPath,
                                                                                      is_plot_imp = False)
                y_pred = mean_test_pred
            elif model_option == 'simpleGP':
                y_scaler = StandardScaler()
                scaled_y_train = y_scaler.fit_transform(y_train_task[:,j].reshape(-1,1))
                """
                if y_scale_label == 'y-robust' and model_option == 'simpleGP':
                    y_scaler = RobustScaler()
                    scaled_y_train = y_scaler.fit_transform(y_train_task[:,j].reshape(-1,1))
                elif y_scale_label == 'y-stand' and model_option == 'simpleGP':
                    y_scaler = StandardScaler()
                    scaled_y_train = y_scaler.fit_transform(y_train_task[:,j].reshape(-1,1))
                elif y_scale_label == 'y-minmax' and model_option == 'simpleGP':
                    y_scaler = MinMaxScaler()
                    scaled_y_train = y_scaler.fit_transform(y_train_task[:,j].reshape(-1,1))
                else:
                    scaled_y_train = y_train_task[:,j]
                """

                t_train_y_task = torch.Tensor(scaled_y_train).flatten()
                test_mean,test_lower,test_upper,train_mean,train_lower,train_upper = simpleGP_gpytorch(t_train_x_task,t_test_x_task,t_train_y_task,training_iter = 50)
    
      
                t_test_y_task = torch.Tensor(y_test_task[:,j].astype(np.float32))
                arr_test_y = t_test_y_task.detach().numpy()

                if y_scale_label=='no-y-scale':
                    arr_train_y = t_train_y_task.detach().numpy()
                    arr_test_mean = test_mean.detach().numpy()
                    arr_test_lower = test_lower.detach().numpy()
                    arr_test_upper = test_upper.detach().numpy()

                    arr_train_mean = train_mean.detach().numpy()
                    arr_train_lower = train_lower.detach().numpy()
                    arr_train_upper = train_upper.detach().numpy()
                else:
                    arr_train_y = y_scaler.inverse_transform(scaled_y_train)
                    arr_train_mean = y_scaler.inverse_transform(train_mean.reshape(-1,1))
                    arr_train_lower = y_scaler.inverse_transform(train_lower.reshape(-1,1))
                    arr_train_upper = y_scaler.inverse_transform(train_upper.reshape(-1,1))

                    arr_test_mean = y_scaler.inverse_transform(test_mean.reshape(-1,1))
                    arr_test_lower = y_scaler.inverse_transform(test_lower.reshape(-1,1))
                    arr_test_upper = y_scaler.inverse_transform(test_upper.reshape(-1,1))

                y_true = arr_test_y
                y_pred = arr_test_mean
                y_train = arr_train_y
                mean_train_pred = arr_train_mean
                std_train_pred = arr_train_mean - arr_train_lower
                std_test_pred = arr_test_mean - arr_test_lower

            train_comp = np.concatenate((y_train.reshape(-1,1),mean_train_pred.reshape(-1,1)),axis = 1)
            df_train_comp = pd.DataFrame(train_comp,columns = ['true','pred'])
            df_train_comp['upper'] = df_train_comp['pred'] + std_train_pred.squeeze()
            df_train_comp['lower'] = df_train_comp['pred'] - std_train_pred.squeeze()
            df_train_comp['mode'] = 'train'
    
            test_comp = np.concatenate((y_true.reshape(-1,1),y_pred.reshape(-1,1)),axis = 1)
            df_test_comp = pd.DataFrame(test_comp,columns = ['true','pred'])
            df_test_comp['upper'] = df_test_comp['pred'] + std_test_pred.squeeze()
            df_test_comp['lower'] = df_test_comp['pred'] - std_test_pred.squeeze()
            df_test_comp['mode'] = 'test'
    
            df_comp = pd.concat([df_train_comp,df_test_comp])
            df_comp_sorted = df_comp.sort_values(by = ['true'],ascending=True)

            #plot a parity line
            y_axes[a].plot(df_comp_sorted['true'], df_comp_sorted['true'], '--',c = 'black')
    
            # Plot training data as blue stars
            y_axes[a].plot(df_train_comp['true'], df_train_comp['pred'], 'k*',c = 'blue',markersize=10)

            # Plot training data as red stars
            y_axes[a].plot(df_test_comp['true'], df_test_comp['pred'], 'k*',c = 'red',markersize=15)
            # Predictive mean as blue line
            y_axes[a].plot(df_comp_sorted['true'], df_comp_sorted['pred'],c = 'blue')
    
            if model_option != 'rf':
                if uncertainty_figure_option == 'shade':
                    # Shade in confidence
                    y_axes[a].fill_between(x = df_comp_sorted['true'],y1 = df_comp_sorted['lower'], y2 = df_comp_sorted['upper'], color='b', alpha=.15)
                else:
                    yerr = df_comp_sorted['pred'] - df_comp_sorted['lower']
                    yerr = yerr.values.tolist()
                    yerr_train = df_train_comp['pred'] - df_train_comp['lower']
                    yerr_train = yerr_train.values.tolist()
                    yerr_test = df_test_comp['pred'] - df_test_comp['lower']
                    yerr_test = yerr_test.values.tolist()
                    y_axes[a].errorbar(x = df_train_comp['true'], y = df_train_comp['pred'], yerr = yerr_train, capsize=1, fmt='none', ecolor = 'blue')
                    y_axes[a].errorbar(x = df_test_comp['true'], y = df_test_comp['pred'], yerr = yerr_test, capsize=1, fmt='none', ecolor = 'red')
    
            if is_set_axis_limit:
                y_axes[a].set_xlim([plot_axis_lb[j],plot_axis_ub[j]])
                y_axes[a].set_ylim([plot_axis_lb[j],plot_axis_ub[j]])

            if model_option == 'rf':
                y_axes[a].legend(['Parity','Train','Test','GP Mean'])
            else:
                y_axes[a].legend(['Parity','Train','Test','GP Mean','GP Train Confidence','GP Test Confidence'])
            y_axes[a].set_title('Task_'+str(task_ind+1) + '_Output_'+str(j+1))
            y_axes[a].set_xlabel('actual')
            y_axes[a].set_ylabel('pred')
    

            arr_test_metrics[0,a] = np.round(metrics.mean_absolute_error(y_true, y_pred),2)
            arr_test_metrics[1,a] = np.round(metrics.median_absolute_error(y_true, y_pred),2)
            arr_test_metrics[2,a] = np.round(metrics.mean_squared_error(y_true, y_pred),2)
            arr_test_metrics[3,a] = round(metrics.root_mean_squared_error(y_true, y_pred),2)
            arr_test_metrics[4,a] = round(mdape(y_true, y_pred),2)
            arr_test_metrics[5,a] = round(metrics.max_error(y_true, y_pred),2)
            arr_test_metrics[6,a] = round(metrics.explained_variance_score(y_true, y_pred),2)
            arr_test_metrics[7,a] = round(metrics.r2_score(y_true, y_pred),2)
            if model_option != 'rf':
                arr_test_metrics[8,a] = round(np.mean(std_test_pred),2)
                arr_test_metrics[9,a] = round(np.min(std_test_pred),2)
                arr_test_metrics[10,a] = round(np.max(std_test_pred),2)

            arr_train_metrics[0,a] = round(metrics.mean_absolute_error(y_train, mean_train_pred),2)
            arr_train_metrics[1,a] = round(metrics.median_absolute_error(y_train, mean_train_pred),2)
            arr_train_metrics[2,a] = round(metrics.mean_squared_error(y_train, mean_train_pred),2)
            arr_train_metrics[3,a] = round(metrics.root_mean_squared_error(y_train, mean_train_pred),2)
            arr_train_metrics[4,a] = round(mdape(y_train, mean_train_pred),2)
            arr_train_metrics[5,a] = round(metrics.max_error(y_train, mean_train_pred),2)
            arr_train_metrics[6,a] = round(metrics.explained_variance_score(y_train, mean_train_pred),2)
            arr_train_metrics[7,a] = round(metrics.r2_score(y_train, mean_train_pred),2)
            if model_option != 'rf':
                arr_train_metrics[8,a] = round(np.mean(std_train_pred),2)
                arr_train_metrics[9,a] = round(np.min(std_train_pred),2)
                arr_train_metrics[10,a] = round(np.max(std_train_pred),2)

    
    if uncertainty_figure_option == 'shade' and is_set_axis_limit:
        plt.savefig(figPath+'/true-pred-shade.jpg')
    elif uncertainty_figure_option == 'shade' and ~is_set_axis_limit:
        plt.savefig(figPath+'/true-pred-shade-zoomin.jpg')
    elif uncertainty_figure_option == 'errorbar' and is_set_axis_limit:
        plt.savefig(figPath+'/true-pred-errorbar.jpg')
    elif uncertainty_figure_option == 'errorbar' and ~is_set_axis_limit:
        plt.savefig(figPath+'/true-pred-errorbar-zoomin.jpg')

    df_test_metrics = pd.DataFrame(arr_test_metrics,columns = cols_target_wide, 
                               index = ['MAE','MAE2','MSE','RMSE','MAPE','MAXE','EVS','R2','AVG_STD','MIN_STD','MAX_STD'])
    print(df_test_metrics)

    df_train_metrics = pd.DataFrame(arr_train_metrics,columns = cols_target_wide, 
                               index = ['MAE','MAE2','MSE','RMSE','MAPE','MAXE','EVS','R2','AVG_STD','MIN_STD','MAX_STD'])
    print(df_train_metrics)

    df_train_metrics.to_csv(figPath+'/df_train_metrics.csv')
    df_test_metrics.to_csv(figPath+'/df_test_metrics.csv')


def main():
    # Your main program logic here
    cfg = get_cfg_defaults_all()
    cfg.freeze()
    df_x = load_input_data(input_path = cfg.PATH.INPUT,index_col = 0,n_subset = 5000)
    df_y = load_output_data(output_path = cfg.PATH.INPUT,index_col = 0,n_subset = 5000)
    train_fun(df_x,df_y,figPath = cfg.PATH.OUTPUT,x_scale_label = cfg.MODEL.X_SCALE,
              y_scale_label = cfg.MODEL.X_SCALE,model_option = cfg.MODEL.MODEL_NAME)

if __name__ == "__main__":
    main()