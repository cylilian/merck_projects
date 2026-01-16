# We will use the simplest form of GP model, exact inference
import torch
import gpytorch
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, ShuffleSplit

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        #print('initial covar_x.shape',covar_x.shape)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def singleGP_gpytorch(t_train_x,t_test_x,t_train_y,training_iter = 50):
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

def singleGP_gpytorch_train(t_train_x,t_train_y,training_iter = 50,init_len_scale = 4.0,my_lr=0.1):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(t_train_x, t_train_y, likelihood)
    model.covar_module.base_kernel.lengthscale = init_len_scale
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=my_lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss_list = []
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(t_train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, t_train_y).sum()
        loss_list.append(loss.item())
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    #plt.plot(loss_list)
    #plt.title('Neg. Loss', fontsize='small')
    
    model.eval()
    model.likelihood.eval()

    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        train_pred = model.likelihood(model(t_train_x))
        train_mean = train_pred.mean
        train_lower, train_upper = train_pred.confidence_region()
    return train_mean, train_lower, train_upper,model

def singleGP_gpytorch_reference(model,t_train_x,t_test_x):
    
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


def rf_sklearn(X_train,y_train,X_test,random_state= 0,cv_option = 'kfold',n_cv = 5):
    #rf = RandomForestRegressor(random_state= 10)
    #rf.fit(X_train,y_train)
    rf = RandomForestRegressor()
    #define search grid
    n_estimators = [16,32,64,128] # number of trees in the random forest
    max_features = [32,64,128,256] # number of features in consideration at every split
    max_depth = [8,16,32,64]
    #max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 4,8] # minimum sample number to split a node
    min_samples_leaf = [1,2,4] # minimum sample number that can be stored in a leaf node
    bootstrap = [True,False] # method used to sample data points
    criterion = ['squared_error','absolute_error']

    search_grid = {'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'criterion': criterion,
    'bootstrap': bootstrap}
    
    if cv_option == 'kfold':
        rf_grid = GridSearchCV(estimator = rf, 
                               param_grid = search_grid, 
                               cv = cv_option, 
                               scoring=['neg_mean_absolute_percentage_error','neg_mean_squared_error', 'r2'],
                               refit='r2',
                               verbose=1,
                               n_jobs = -1,
                               return_train_score=True)
    elif cv_option == 'shuffle-split':
        ss = ShuffleSplit(n_splits=n_cv, test_size=0.2, random_state=random_state)
        rf_grid = GridSearchCV(estimator = rf, 
                               param_grid = search_grid, 
                               cv = ss, 
                               scoring=['neg_mean_absolute_percentage_error','neg_mean_squared_error', 'r2'],
                               refit='r2',
                               verbose=1,
                               n_jobs = -1,
                               return_train_score=True)

    
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_train_pred = best_rf.predict(X_train)
    y_pred = best_rf.predict(X_test)
    """
    test_var = metrics.mean_squared_error(y_test, y_pred)
    train_var = metrics.mean_squared_error(y_train, y_train_pred)
    test_std = np.sqrt(test_var)
    train_std = np.sqrt(train_var)
    """

    #test_std = np.abs(y_test - y_pred)
    #train_std = np.abs(y_train - y_train_pred)
    return y_train_pred, y_pred, rf_grid

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_tasks, rank = 0):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MTMOGP(gpytorch.models.ExactGP):

    def __init__(
        self,
        train_X,
        train_Y,
        likelihood,
        data_kernel = 'Matern',
        task_rank = None,
        output_rank = None
    ) -> None:

        num_outputs = train_Y.shape[-1]
        num_tasks = len(torch.unique(train_X[..., -1]))

        super(MTMOGP, self).__init__(train_X, train_Y,likelihood)
        self.task_rank = task_rank if task_rank is not None else num_tasks
        self.output_rank = output_rank if output_rank is not None else num_outputs

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_outputs
        )
        
        if data_kernel == 'Matern':
            self.data_kernel = gpytorch.kernels.MaternKernel()
        else:
            self.data_kernel = gpytorch.kernels.RBFKernel()
        self.task_kernel = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank = self.task_rank) #default rank is 1
        self.output_kernel = gpytorch.kernels.IndexKernel(num_tasks=num_outputs, rank = self.output_rank) #default rank is 1
        
        self.to(train_X)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        task_term = self.task_kernel(x[..., -1].long())
        data_and_task_x = self.data_kernel(x[..., :-1]).mul(task_term)
        output_x = self.output_kernel.covar_matrix
        covar_x = gpytorch.lazy.KroneckerProductLazyTensor(data_and_task_x, output_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)




#from gpytorch.kernels import Kernel


import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel

def index_to_slices(index):
    """
    take a numpy array of integers (index) and return a  nested list of slices such that the slices describe the start, stop points for each integer in the index.

    e.g.
    >>> index = np.asarray([0,0,0,1,1,1,2,2,2])
    returns
    >>> [[slice(0,3,None)],[slice(3,6,None)],[slice(6,9,None)]]

    or, a more complicated example
    >>> index = np.asarray([0,0,1,1,0,2,2,2,1,1])
    returns
    >>> [[slice(0,2,None),slice(4,5,None)],[slice(2,4,None),slice(8,10,None)],[slice(5,8,None)]]
    """
    if len(index) == 0:
        return []

    # contruct the return structure
    ind = np.asarray(index, dtype=int)
    ret = [[] for i in range(ind.max() + 1)]

    # find the switchpoints
    ind_ = np.hstack((ind, ind[0] + ind[-1] + 1))
    switchpoints = np.nonzero(ind_ - np.roll(ind_, +1))[0]

    [
        ret[ind_i].append(slice(*indexes_i))
        for ind_i, indexes_i in zip(
            ind[switchpoints[:-1]], zip(switchpoints, switchpoints[1:])
        )
    ]
    return ret





class HierarchicalGPModel(gpytorch.models.ExactGP):
    """
    A kernel which can represent a simple hierarchical model.
    
    To construct this kernel, you must pass a list of kernels. The first kernel
    will be assumed to be the 'base' kernel, and will be computed everywhere.
    For every additional kernel, we assume another layer in the hierarchy.
    """

    def __init__(self, train_x, train_y, likelihood):
        super(HierarchicalGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        """
        # Define the global and local kernels
        self.global_kernel = ScaleKernel(RBFKernel())
        self.local_kernel = ScaleKernel(RBFKernel())
        self.kernels = [self.global_kernel,self.local_kernel]
        self.input_max = train_x.shape[1] - 1
        self.extra_dims = range(self.input_max, self.input_max + len(self.kernels)-1)
        """
    
    
    def forward(self, x):
        #print('x.shape',x.shape)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        #print('initial covar_x.shape',covar_x.shape)
        
        """
        #compute global kernel for each x
        #covar_x = self.kernels[0](x)
        #compute local kernel
        slices = [index_to_slices(x[:,i]) for i in self.extra_dims]
        print(slices)
        ls_result = [[[covar_x[s,s] + k(x[s]) + covar_x[s, s] for s in slices_i] for slices_i in slices_k] for k, slices_k in zip(self.kernels[1:], slices)]
        covar_x0 = ls_result[0][0][0]
        covar_x1 = ls_result[0][1][0]
        covar_x2 = ls_result[0][2][0]

        covar_x_local = torch.block_diag(covar_x0.to_dense(),covar_x1.to_dense(),covar_x2.to_dense())
        print('covar_x.shape',covar_x.to_dense().shape)
        print('covar_x_local.shape',covar_x_local.shape)
        #covar_x_total = covar_x.to_dense() + covar_x_local
        #return gpytorch.distributions.MultivariateNormal(mean_x, covar_x_total)"
        """
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
