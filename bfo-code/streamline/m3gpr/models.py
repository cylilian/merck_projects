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


from gpytorch.priors import NormalPrior,LogNormalPrior,SmoothedBoxPrior,HorseshoePrior
from gpytorch.constraints import GreaterThan,Positive


from torch import nn
import torch.nn.functional as F
import math
from typing import Dict,List,Optional
from copy import deepcopy
from tqdm import tqdm

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


## Basic MTMO class for LMGP
class MultiOutputMultiTaskGP(gpytorch.models.ExactGP):

    def __init__(
        self,
        train_X,
        train_Y,
        data_kernel,
        noise_indices,
        fix_noise:bool=False,
        lb_noise:float=1e-4,
        task_rank = None,
        output_rank = None,
        lik_rank = None
    ) -> None:

        num_outputs = train_Y.shape[-1]
        num_tasks = len(torch.unique(train_X[..., -1]))
        self._num_tasks = num_tasks
        self._num_outputs = num_outputs
        
        self.task_rank = task_rank if task_rank is not None else num_tasks
        self.output_rank = output_rank if output_rank is not None else num_outputs
        self.lik_rank = lik_rank if lik_rank is not None else 0
        # initializing likelihood
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs,rank = self.lik_rank)
        super(MultiOutputMultiTaskGP, self).__init__(train_X, train_Y,likelihood)

        self.likelihood.register_prior('raw_noise_prior',HorseshoePrior(0.01,lb_noise),'raw_noise')
        if self.lik_rank == 0:
            self.likelihood.register_prior('raw_task_noises_prior',HorseshoePrior(0.01,lb_noise),'raw_task_noises')    
        else:
            self.likelihood.register_prior('task_noise_covar_factor_prior',NormalPrior(0.,1),'task_noise_covar_factor')

        if fix_noise:
            self.likelihood.raw_noise.requires_grad_(False)
            self.likelihood.noise_covar.noise =torch.tensor(4.9901e-05)

        
        #define prior for mean module
        mean_list = [gpytorch.means.ConstantMean(NormalPrior(0,1)) for t in range(num_outputs)]
        self.mean_module = gpytorch.means.MultitaskMean(
            mean_list, num_tasks=num_outputs
        )
        
        self.data_kernel = data_kernel
        
        
        if not isinstance(data_kernel,gpytorch.kernels.Kernel):
            raise RuntimeError(
                "specified data kernel is not a `gpytorch.kernels.Kernel` instance"
            )

        #define kernel for gplvm on mixed variables
        self.data_kernel2 = gpytorch.kernels.RBFKernel()
        self.data_kernel2.register_prior(
                    'lengthscale_prior',SmoothedBoxPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )

        self.task_kernel = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank = self.task_rank) #default rank is 1
        self.output_kernel = gpytorch.kernels.IndexKernel(num_tasks=num_outputs, rank = self.output_rank) #default rank is 1
        
        self.task_kernel.register_prior("covar_factor_prior",NormalPrior(0.,1),lambda m: m._parameters['covar_factor'])
        self.task_kernel.register_prior("raw_var_prior",NormalPrior(0.,1),lambda m: m._parameters['raw_var'])

        self.output_kernel.register_prior("covar_factor_prior",NormalPrior(0.,1),lambda m: m._parameters['covar_factor'])
        self.output_kernel.register_prior("raw_var_prior",NormalPrior(0.,1),lambda m: m._parameters['raw_var'])
        
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        task_term = self.task_kernel(x[..., -1].long())
        data_and_task_x = self.data_kernel(x[..., :-1]).mul(task_term)
        output_x = self.output_kernel.covar_matrix
        covar_x = gpytorch.lazy.KroneckerProductLazyTensor(data_and_task_x, output_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def predict(
        self,x:torch.Tensor,return_std:bool=False,include_noise:bool=False
    ):

        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_res = self.likelihood(self.forward(x))   
            mean = pred_res.mean
            lower, upper = pred_res.confidence_region()
        return mean, lower, upper

## optimization function using torch
def fit_model_torch(
    model,
    lr_default,
    num_iter,
    model_param_groups:Optional[List]=None,
    num_restarts:int=0,
    break_steps:int = 50) -> float:
    '''Optimize the likelihood/posterior of a standard GP model using `torch.optim.Adam`.

    :param model: A model instance derived from the `models.GPR` class.
    :type model: models.GPR

    :param model_param_groups: list of parameters to optimizes or dicts defining parameter
        groups. If `None` is specified, then all parameters with `.requires_grad`=`True` are 
        included. Defaults to `None`.
    :type model_param_groups: list, optional

    :param lr_default: The default learning rate for all parameter groups. To use different 
        learning rates for some groups, specify them `model_param_groups`. 
    :type lr_default: float, optional

    :param num_iter: The number of optimization steps from each starting point. This is the only
        termination criterion for the optimizer.
    :type num_iter: float, optional

    :param num_restarts: The number of times to restart the local optimization from a 
        new starting point. Defaults to 5
    :type num_restarts: int, optional

    :returns: the best (negative) log-likelihood/log-posterior found
    :rtype: float
    '''  
    model.train()
    
    # objective
    #mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    
    f_inc = math.inf
    current_state_dict = model.state_dict()


    loss_hist_total = []

    for i in range(num_restarts+1):
        optimizer = torch.optim.Adam(
            model.parameters() if model_param_groups is None else model_param_groups, 
            lr=lr_default)
        loss_hist = []
        epochs_iter = tqdm(range(num_iter),desc='Epoch',position=0,leave=True)
        for j in epochs_iter:
            # zero gradients from previous iteration
            optimizer.zero_grad()
            # output from model
            output = model(*model.train_inputs)
            # calculate loss and backprop gradients
            #loss = -mll(output,model.train_targets)
            loss = -model.likelihood(output).log_prob(model.train_targets)
            loss.backward()
            optimizer.step()

            acc_loss = loss.item()
            desc = f'Epoch {j} - loss {acc_loss:.4f}'
            epochs_iter.set_description(desc)
            epochs_iter.update(1)
            loss_hist.append(acc_loss)

            if j > break_steps and j%break_steps == 0:
                if ( (torch.mean(torch.Tensor(loss_hist)[j-break_steps:j]) - loss_hist[j]) <= 0 ):
                    break
        
        loss_hist_total.append(loss_hist)

        if loss.item()<f_inc:
            current_state_dict = deepcopy(model.state_dict())
            f_inc = loss.item()
    
    model.load_state_dict(current_state_dict)

    return f_inc, loss_hist_total

############################################
class Linear_MAP(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        
    def forward(self, input, transform = lambda x: x):
        return F.linear(input,transform(self.weight), self.bias)

######################################################################## GPLVM  #####################################################
class GPLVM(nn.Module):
    def __init__(self, GP_Lmgp, input_size, num_classes, name):
        super(GPLVM, self).__init__()
        
        self.gplvm = Linear_MAP(input_size, num_classes, bias = False)
        GP_Lmgp.register_parameter(name, self.gplvm.weight)
        GP_Lmgp.register_prior(name = 'gplvm_prior_'+name, prior=gpytorch.priors.NormalPrior(0,1) , param_or_closure=name)

    def forward(self, x, transform = lambda x: x):
        x = self.gplvm(x, transform)
        return x


######################################################################## LMGP  #####################################################
class LMGP(nn.Module):
    def __init__(self, GP_Lmgp, input_size, num_classes,name):
        super(LMGP, self).__init__()
        
        
        self.fci = Linear_MAP(input_size, num_classes, bias = False)
        GP_Lmgp.register_parameter(name, self.fci.weight)
        GP_Lmgp.register_prior(name = 'latent_prior_'+name, prior=gpytorch.priors.NormalPrior(0,1) , param_or_closure=name)

    def forward(self, x, transform = lambda x: x):
        x = self.fci(x, transform)
        return x

##inporating LMGP into GPR
class GP_Lmgp(MultiOutputMultiTaskGP):
    """The GP_Lmgp extends GPs to handle categorical inputs

    :note: Binary categorical variables should not be treated as qualitative inputs. There is no 
        benefit from applying a latent variable treatment for such variables. Instead, treat them
        as numerical inputs.

    :param train_x: The training inputs (size N x d). Qualitative inputs needed to be encoded as 
        integers 0,...,L-1 where L is the number of levels. For best performance, scale the 
        numerical variables to the unit hypercube.
    """
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        quant_correlation_class:str='RBFKernel',
        my_nu = 2.5, #nu for MaternKernel
        qual_ind_lev = {},
        quant_index = [],
        task_index = -1,
        multiple_noise = False,
        lv_dim:int=2,
        fix_noise:bool=False,
        fixed_length_scale:bool=False,
        fixed_omega=torch.tensor([1.0]),
        lb_noise:float=1e-4,
        encoding_type = 'one-hot',
        uniform_encoding_columns = 2,
        lv_columns = [] ,
        base='single_constant',
        seed_number=1,
        device="cpu",
        dtype= torch.float,
        is_mix_reduce = False,
        task_rank=None,
        output_rank = None,
        lik_rank=None,
        lr = None,
        n_iter = None
    ) -> None:
        
        tkwargs = {}  # or dict()
        tkwargs['dtype'] = dtype
        tkwargs['device'] =torch.device(device)
        self.tkwargs=tkwargs

        self.lr = lr
        self.n_iter = n_iter
        if fixed_length_scale:
            self.fixed_omega=fixed_omega.to(**self.tkwargs)
        else:
            self.fixed_omega=None

        train_x=self.fill_nan_with_mean(train_x)
        ###############################################################################################
        ###############################################################################################
        self.seed=seed_number
        self.calibration_source_index=0    ## It is supposed the calibration parameter is for high fidelity needs
        qual_index = list(qual_ind_lev.keys())
        all_index = set(range(train_x.shape[-1]))
        #quant_index = list(all_index.difference(qual_index))
        num_levels_per_var = list(qual_ind_lev.values())
        #------------------- lm columns --------------------------
        lm_columns = list(set(qual_index).difference(lv_columns))
        if len(lm_columns) > 0:
            qual_kernel_columns = [*lv_columns, lm_columns]
        else:
            qual_kernel_columns = lv_columns
        #########################
        if len(qual_index) > 0:
            train_x = torch.tensor(train_x)#.to(**self.tkwargs)
        #
        train_x=train_x.to(**self.tkwargs)
        train_y=train_y.to(**self.tkwargs)
        #train_y=train_y.reshape(-1)#.to(**self.tkwargs)
        
        if multiple_noise:
            noise_indices = list(range(0,num_levels_per_var[-1]))
        else:
            noise_indices = []

        if len(qual_index) == 1 and num_levels_per_var[0] < 2:
            temp = quant_index.copy()
            temp.append(qual_index[0])
            quant_index = temp.copy()
            qual_index = []
            lv_dim = 0
        elif len(qual_index) == 0:
            lv_dim = 0

        quant_correlation_class_name = quant_correlation_class

        if len(qual_index) == 0:
            lv_dim = 0
    

        if len(qual_index) > 0:
            ####################### Defined multiple kernels for seperate variables ###################
            qual_kernels = []
            for i in range(len(qual_kernel_columns)):
                qual_kernels.append(gpytorch.kernels.RBFKernel(
                    active_dims=torch.arange(lv_dim) + lv_dim * i) )
                #qual_kernels[i].initialize(**{'lengthscale':1.0})
                #qual_kernels[i].raw_lengthscale.requires_grad_(False)  
            qual_kernels[0].register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )

        if len(quant_index) == 0:
            print('---no numerical columns----')
            correlation_kernel = qual_kernels[0]
            for i in range(1, len(qual_kernels)):
                correlation_kernel *= qual_kernels[i]
        else:
            try:
                quant_correlation_class = getattr(gpytorch.kernels,quant_correlation_class)
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % quant_correlation_class
                )
            if quant_correlation_class_name == 'RBFKernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * lv_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= torch.exp,inv_transform= torch.log)
                )
            
            elif quant_correlation_class_name == 'MaternKernel':
                quant_kernel = quant_correlation_class(
                    nu = my_nu,
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns)*lv_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
                )
                #####################
            
            if quant_correlation_class_name == 'RBFKernel':
                quant_kernel.register_prior(
                    'lengthscale_prior', SmoothedBoxPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )
                
            elif quant_correlation_class_name == 'MaternKernel':
                quant_kernel.register_prior(
                    #'lengthscale_prior', SmoothedBoxPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )
            
            #########Product between qual kernels and quant kernels############
            if len(qual_index) > 0:
                temp = qual_kernels[0]
                for i in range(1, len(qual_kernels)):
                    temp *= qual_kernels[i]
                correlation_kernel = temp*quant_kernel #+ qual_kernel + quant_kernel
            else:
                correlation_kernel = quant_kernel
            
        
        super(GP_Lmgp,self).__init__(
            train_X=train_x,train_Y=train_y,noise_indices=noise_indices,
            data_kernel=correlation_kernel,
            fix_noise=fix_noise,lb_noise=lb_noise,
            task_rank= task_rank, output_rank = output_rank, lik_rank = lik_rank
        )
        
        # register index and transforms
        self.register_buffer('quant_index',torch.tensor(quant_index))
        self.register_buffer('qual_index',torch.tensor(qual_index))

        self.quant_kernel = quant_kernel
        self.correlation_kernel = correlation_kernel
        self.qual_kernels = qual_kernels
        
        self.qual_kernel_columns = qual_kernel_columns
        self.is_mix_reduce = is_mix_reduce
        ######################## latent variable mapping  ########################
        self.num_levels_per_var = num_levels_per_var
        self.lv_dim = lv_dim
        self.uniform_encoding_columns = uniform_encoding_columns
        self.encoding_type = encoding_type
        self.perm =[]
        self.zeta = []
        self.random_zeta=[]
        self.perm_dict = []
        self.A_matrix = []
        
        self.epsilon=None
        self.epsilon_f=None
        self.embeddings_Dtrain=[]
        self.count=train_x.size()[0]
        if len(qual_kernel_columns) > 0:
            for i in range(len(qual_kernel_columns)):
                if type(qual_kernel_columns[i]) == int:
                    num = self.num_levels_per_var[qual_index.index(qual_kernel_columns[i])]
                    cat = [num]
                else:
                    cat = [self.num_levels_per_var[qual_index.index(k)] for k in qual_kernel_columns[i]]
                    num = sum(cat)

                zeta, perm, perm_dict = self.zeta_matrix(num_levels=cat, lv_dim = self.lv_dim)
                self.zeta.append(zeta)
                self.perm.append(perm)
                self.perm_dict.append(perm_dict)       
                ###################################  latent map #################################   
                model_lmgp = LMGP(self, input_size= num, num_classes=lv_dim, 
                        name ='latent'+ str(qual_kernel_columns[i])).to(**self.tkwargs)
                self.A_matrix.append(model_lmgp)
        ######## GPLVM transformation if there are quantitative variables########################
        
        if is_mix_reduce and len(quant_index) > 0:
            dim_x_new = self.lv_dim+len(quant_index)
            self.W_matrix= GPLVM(self, input_size=  dim_x_new, num_classes=lv_dim, 
                        name ='gplvm'+ str(dim_x_new)).to(**self.tkwargs)
        
        ##################################################################################
        if fixed_length_scale == True:
            self.covar_module.base_kernel.raw_lengthscale.data = self.fixed_omega #torch.tensor([self.omega, self.omega])  # Set the desired value
            self.covar_module.base_kernel.raw_lengthscale.requires_grad = False  # Fix the hyperparameter
        ###################################  Mean Function #################################   
        #i=0
        self.base=base
        self.num_sources=int(torch.max(train_x[:,-1]))
        size=train_x.shape[1]
        self.single_base_register(size,base_type=self.base,wm='mean_module')

    def forward(self,x:torch.Tensor):
        
        Numper_of_pass=1
        
        size_sigma_sum = x.size(0)*self._num_outputs
        #size_sigma_sum = x.size(0)
        Sigma_sum=torch.zeros(size_sigma_sum,size_sigma_sum, dtype=torch.float64).to(self.tkwargs['device'])
        
        mean_x_sum=torch.zeros(x.size(0),self._num_outputs, dtype=torch.float64).to(self.tkwargs['device'])
        #print('mean_x_sum.shape',mean_x_sum.shape)
        task_term = self.task_kernel(x[..., -1].long())

        for NP in range(Numper_of_pass):
            x_forward_raw=x[..., :-1].clone()
            nd_flag = 0
            if x.dim() > 2:
                xsize = x.shape
                x = x.reshape(-1, x.shape[-1])
                nd_flag = 1
            
            x_new= x
            x_gplvm = x
            if len(self.qual_kernel_columns) > 0:
                embeddings = []
                for i in range(len(self.qual_kernel_columns)):
                    temp= self.transform_categorical(x=x[:,self.qual_kernel_columns[i]].clone().type(torch.int64).to(self.tkwargs['device']), 
                        perm_dict = self.perm_dict[i], zeta = self.zeta[i])
                dimm=x_forward_raw.size()[0]
                
                embeddings.append(self.A_matrix[i](temp.float().to(**self.tkwargs)))
                x_new= torch.cat([embeddings[0],x[...,self.quant_index.long()].to(**self.tkwargs)],dim=-1)
                #print('x_new.shape after reduce cate',x_new.shape)
            
            if self.is_mix_reduce and len(self.quant_index) > 0:
                x_gplvm = self.W_matrix(x_new.float().to(**self.tkwargs))
                #print('x_gplvm.shape after reduce all',x_gplvm.shape)
                #x_new = x_gplvm
            
            if nd_flag == 1:
                x_new = x_new.reshape(*xsize[:-1], -1)
            
        #################### Multiple baises (General Case) ####################################  
            
            if self.is_mix_reduce and len(self.quant_index) > 0:
                    mean_x = self.mean_module(x_gplvm).to(**self.tkwargs)
            else:
                    mean_x = self.mean_module(x_new).to(**self.tkwargs)
            
            #data_kernel is a product kernel of cate kernel and quant kernel
            data_and_task_x = self.data_kernel(x_new).mul(task_term)
            output_x = self.output_kernel.covar_matrix
            covar_x = gpytorch.lazy.KroneckerProductLazyTensor(data_and_task_x, output_x)

            if self.is_mix_reduce and len(self.quant_index) > 0:
                #data_kernel2 is one kernel for both cate and quant variables
                data_and_task_x_gplvm = self.data_kernel2(x_gplvm).mul(task_term)
                covar_x_mixed = gpytorch.lazy.KroneckerProductLazyTensor(data_and_task_x_gplvm, output_x)
                covar_x = covar_x.mul(covar_x_mixed)

            mean_x_sum+=mean_x
            
            Sigma_sum += covar_x.evaluate()

        # End of the loop for forward pasess ----> Compute ensemble mean and covariance
        k = Numper_of_pass
        ensemble_mean = mean_x_sum/k
        ensemble_covar = torch.zeros_like(Sigma_sum) 
        ensemble_covar= Sigma_sum/k
        ensemble_covar=gpytorch.lazy.NonLazyTensor(ensemble_covar)
        Sigma_sum=0
        #print('ensemble_mean.shape',ensemble_mean.shape)
        #print('ensemble_covar.shape',ensemble_covar.evaluate().shape)

        return gpytorch.distributions.MultitaskMultivariateNormal(ensemble_mean.float(),ensemble_covar.float())
    
    ################################################################ Mean Functions #####################################################################################
    
    def single_base_register(self,size=1,base_type='single_zero',wm='mean_module'):
        if base_type=='single_constant':
            #mean_list = [gpytorch.means.ConstantMean(NormalPrior(0.,1)) for t in range(self._num_tasks)]
            setattr(self,wm, self.mean_module)
        elif base_type=='single_zero':
            setattr(self,wm, gpytorch.means.ZeroMean()) 

    ################################################################ Fit #####################################################################################
    def fit(self,add_prior:bool=True,num_restarts:int=64,theta0_list:Optional[List[np.ndarray]]=None,jac:bool=True,
            options:Dict={},n_jobs:int=-1,method = 'L-BFGS-B',constraint=False,bounds=False,regularization_parameter:List[int]=[0,0],optim_type='scipy'):
        print("## Learning the model's parameters has started ##")
        #optim_type=='adam_torch':
        fit_model_torch(model=self,
                    model_param_groups=None,
                    lr_default=self.lr,
                    num_iter=self.n_iter,
                    num_restarts=num_restarts,
                    break_steps= 50)
        
        print("## Learning the model's parameters is successfully finished ##")




    def fill_nan_with_mean(self,train_x):
        # Check if there are any NaNs in the tensor
        if torch.isnan(train_x).any():
            # Compute the mean of non-NaN elements column-wise
            col_means = torch.nanmean(train_x, dim=0)
            # Find indices where NaNs are located
            nan_indices = torch.isnan(train_x)
            # Replace NaNs with the corresponding column-wise mean
            train_x[nan_indices] = col_means.repeat(train_x.shape[0], 1)[nan_indices]

        return train_x
    ############################  Prediction and Visualization  ###############################
    
    def predict(self, Xtest,return_std=True, include_noise = True):
        with torch.no_grad():
            return super().predict(Xtest, return_std = return_std, include_noise= include_noise)
    
    def predict_with_grad(self, Xtest,return_std=True, include_noise = True):
        return super().predict(Xtest, return_std = return_std, include_noise= include_noise)
    
    @classmethod
    def show(cls):
        plt.show()
        
    def get_params(self, name = None):
        params = {}
        print('###################Parameters###########################')
        for n, value in self.named_parameters():
             params[n] = value
        if name is None:
            print(params)
            return params
        else:
            if name == 'Mean':
                key = 'mean_module.constant'
            elif name == 'Sigma':
                key = 'covar_module.raw_outputscale'
            elif name == 'Noise':
                key = 'likelihood.noise_covar.raw_noise'
            elif name == 'Omega':
                for n in params.keys():
                    if 'raw_lengthscale' in n and params[n].numel() > 1:
                        key = n
            print(params[key])
            return params[key]

    def zeta_matrix(self,
        num_levels:int,
        lv_dim:int,
        batch_shape=torch.Size()
    ) -> None:

        if any([i == 1 for i in num_levels]):
            raise ValueError('Categorical variable has only one level!')

        if lv_dim == 1:
            raise RuntimeWarning('1D latent variables are difficult to optimize!')
        
        for level in num_levels:
            if lv_dim > level - 0:
                lv_dim = min(lv_dim, level-1)
                raise RuntimeWarning(
                    'The LV dimension can atmost be num_levels-1. '
                    'Setting it to %s in place of %s' %(level-1,lv_dim)
                )
    
        from itertools import product
        levels = []
        for l in num_levels:
            levels.append(torch.arange(l))

        perm = list(product(*levels))
        perm = torch.tensor(perm, dtype=torch.int64)

        #-------------Mapping-------------------------
        perm_dic = {}
        for i, row in enumerate(perm):
            temp = str(row.tolist())
            if temp not in perm_dic.keys():
                perm_dic[temp] = i

        #-------------One_hot_encoding------------------
        for ii in range(perm.shape[-1]):
            if perm[...,ii].min() != 0:
                perm[...,ii] -= perm[...,ii].min()
            
        perm_one_hot = []
        for i in range(perm.size()[1]):
            perm_one_hot.append( torch.nn.functional.one_hot(perm[:,i]) )

        perm_one_hot = torch.concat(perm_one_hot, axis=1)

        return perm_one_hot, perm, perm_dic

    #################################### transformation functions####################################

    def transform_categorical(self, x:torch.Tensor,perm_dict = [], zeta = []) -> None:
        if x.dim() == 1:
            x = x.reshape(-1,1)
        # categorical should start from 0
        if self.training == False:
            x = torch.tensor(x)
        if self.encoding_type == 'one-hot':
            index = [perm_dict[str(row.tolist())] for row in x]

            if x.dim() == 1:
                x = x.reshape(len(x),)

            return zeta[index,:]

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
