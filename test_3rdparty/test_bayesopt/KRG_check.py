## Bayesian Optimization 
# Check training of KRG krg model
# 1. Get points of cost function in 1D
# 2. Train KRG model
# 3. Evaluate KRG model on a grid of points
# 4. Plot KRG model and training data


import numpy as np 
import pandas as pd
import scipy.optimize as so

from smt.applications.ego import EGO
from smt.sampling_methods import LHS
import smt.surrogate_models

import sys

import matplotlib.pyplot as plt

def fun_npt(x, fun, **kwargs):
    """Evaluate function fun on n points at once
    x must be a 2d array (rows = nr of points, cols = dimension)
    """
    npt, dim = x.shape
    out = np.zeros((npt, 1))
    for i in range(npt):
        out[i, :] = fun(x[i,:], **kwargs) 
    return out


def costfun(x, verbose=True):
    '''Evaluate cost function on one point x'''
    #f = sum((x-1)**2 + 4*np.cos(x) + 2*np.cos(2*x) + 10*np.cos(10*x))
    xlim = 1
    if x<=xlim and x>=-xlim:
        f = x**2
    else:
        f = xlim
    if verbose:
        print('costfun: evaluation %2.10f +++ %2.10f' %(x, f) )
    return f 
    
##################################################################
# With known function
if 0:
    ndim = 1
    xlimits = np.array([[-5, 5]])
    xlimits = np.repeat(xlimits, ndim, axis=0)
    
    #number of points in the initial DOE
    ndoe = 15
    
    x = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 200)).T
    y = fun_npt(x, costfun)
    
    #Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
    sampling = LHS(xlimits=xlimits, random_state=1)
    xdoe = sampling(ndoe)
    ydoe = fun_npt(xdoe, costfun) # or load already computed points 
    
    # KRG
    krg = smt.surrogate_models.KRG(print_global=True, theta0=[0.01], n_start=20,
        corr='matern52', theta_bounds=[0.01, 100], poly='constant')
    # regression types: constant, linear, quadratic
    # correlation types: abs_exp, squar_exp, act_exp, matern52, matern32, gower
    krg.set_training_values(xdoe, ydoe)
    krg.train()
    ypred = krg.predict_values(x)
    yvar = krg.predict_variances(x)
    
    # EGO object?
    ego = EGO(surrogate=krg)
    ego.gpr = krg
    crit = ego.EI(points=x, y_data=ydoe, x_data=xdoe)
    
    # Plot
    fig, ax = plt.subplots(1)
    # True
    ax.plot(x, y, '--', color='black')
    # DOE
    ax.plot(xdoe, ydoe, '.')
    # Prediction
    ax.plot(x, ypred)
    # Variance
    nsig = 3
    var_plus = ypred + nsig * np.sqrt(yvar)
    var_minus = ypred - nsig * np.sqrt(yvar)
    ax.fill_between(x.T[0], var_plus.T[0], var_minus.T[0], alpha=0.3, color="g")
    ## Acquisition function (EI...)
    #ax2 = ax.twinx()
    #ax2.plot(x, crit, color='red')
    # Utils
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['True', 'DOE', 'KRG', 'Var'], loc='best')
    plt.savefig('show_KRG_f.png')
    plt.close('all') 


##################################################################
# With sampled points but somewhat smooth function
if 1:
    ndim = 1
    
    # Load sampled points
    path = '/stck/wjussiau/fenics-python/test/test_bayesopt/'
    import pandas as pd
    #data = pd.read_csv(path + '/J_costfun_nx32.csv')
    #data = pd.read_csv(path + '/J_costfun_m1.csv')
    data = pd.read_csv(path + '/J_costfun.csv')

    select = 32
    xdata = np.reshape(data.x1.to_numpy(), (-1,1))[:select] 
    ydata = np.reshape(data.J.to_numpy() , (-1,1))[:select]

    xlimits = np.array([min(xdata), max(xdata)]).T
    
    # Points to be evaluated with GPR
    x = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], 200)).T
    
    # KRG
    krg = smt.surrogate_models.KRG(print_global=True, theta0=[0.01], n_start=10,
        #corr='squar_exp', theta_bounds=[0.01, 20], poly='constant', nugget=0.01)
        corr='matern52', theta_bounds=[0.01, 20], poly='constant')
    krg.theta0 = 2
    # regression types: constant, linear, quadratic
    # correlation types: abs_exp, squar_exp, act_exp, matern52, matern32, gower
    krg.set_training_values(xdata, ydata)
    krg.train()
    ypred = krg.predict_values(x)
    yvar = krg.predict_variances(x)
    
    ## EGO object?
    ego = EGO(surrogate=krg)
    ego.gpr = krg
    ycrit = ego.EI(points=x, y_data=ydata, x_data=xdata)
    #ycrit = ego.SBO(point=x)
    #ycrit = ego.UCB(point=x)
    
    fig, ax = plt.subplots(1)
    # DOE
    ax.plot(xdata, ydata, '.')
    # Prediction
    ax.plot(x, ypred)
    # Variance
    nsig = 3
    var_plus = ypred + nsig * np.sqrt(yvar)
    var_minus = ypred - nsig * np.sqrt(yvar)
    ax.fill_between(x.T[0], var_plus.T[0], var_minus.T[0], alpha=0.3, color="g")
    ## Acquisition function (EI...)
    ax2 = ax.twinx()
    ax2.plot(x, ycrit, color='red')
    # Utils
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['DOE', 'KRG', 'Var'], loc='best')
    plt.savefig(path + 'show_KRG_J.png')
    plt.close('all') 
    
    
    
