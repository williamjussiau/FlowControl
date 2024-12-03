"""
Description:
This module runs six popular test problems for global optimization
that are defined and described in blackbox_opt/test_funcs/funcs_def.py.

The user is required to import a blackbox function, specify the
algorithm, a starting point and options.
            func: an imported blackbox function object
            x_0: starting point: numpy array with shape (n,1) --n is the
            dimension of x_0
            alg: selected algorithm: string
            options : a dictionary of options customized for each algorithm
For a full list of options available for DFO see the README file in
DFO_src directory.
For scipy algorithms, options for each alg are available by a call to
scipy.optimize.show_options. These options are available at
http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.show_options.html
author: Anahita Hassanzadeh
email: Anahita.Hassanzadeh@gmail.com
"""
import numpy as np
import blackbox_opt.bb_optimize as bbo
#from blackbox_opt.bb_optimize import bb_optimize
## import the function to be optimized
from blackbox_opt.test_funcs.funcs_def import (ackley, sphere, rosen,
                                               booth, bukin, beale)
def get_results(func, x_0, alg, options):
    """
    This function calls the main blackbox optimization
    code with the specified algorithm, options and starting point and
    prints the best point found along with its optimal value.
    input: func: an imported blackbox function object
            x_0: starting point: numpy array with shape (n,1)
            alg: selected algorithm: string
            options : a dictionary of options customized for each algorithm
    """
    res = bbo.bb_optimize(func, x_0, alg, options)

    print("Printing result for function " + func.__name__ + ":")
    print("best point: {}, with obj: {:.6f}".format(
        np.around(res.x.T, decimals=5), float(res.fun)))
    if alg.lower()=="dfo":
        nf = res.func_eval
    else:
        nf = res.nfev
    print('nr f evaluations: ', nf)
    print("------------- " + alg + " Finished ----------------------\n")
    return res


if __name__ == "__main__":
    # separate the functions based on their dimension. This is merely
    # done to ensure the starting point x_0 will later have the
    # correct dimension


    def costfun(x, xmin, xmax):
        #if x > xmax or x < xmin:
        #    return 1
        #else:
        #print('evaluating: ', x)
        #return ((x[0]-1)**2 + np.cos(3*x[0]))**2 + x[1]**2
        f = sum([(x[i]-i)**2 for i in range(len(x))]) + np.sin(2*x[0] + x[1] + x[2])
        #print('evaluated:', f)
        return f 

    my_costfun = lambda x: costfun(x, xmin=-100, xmax=100)
    all_func_names = [my_costfun]

    # Run all the algorithms and problems with given starting points
    # Specify the starting point and options. For example, try the following
    # options.
    for func in all_func_names:
        #x_0 = np.array([1.5, 0., 2., 1.1, 2.2])
        x_0 = np.array([1.5, 0., 2.])
        #x_0 = np.array([0.5, 0.4])
        #if func in td_func_names:
        #    x_0 = np.array([1.3, 0.7])
        #else:
        #    x_0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

        print("\n\n********* Function " + func.__name__ + "********")
        maxfev = 20
        
        alg = "DFO"
        options = {"maxfev": maxfev, "init_delta": 20,
                   "tol_delta": 1e-25, "tol_f": 1e-26, "tol_norm_g": 1e-5,
                   "sample_gen": "auto", "verbosity": 0}
        res_dfo = get_results(func, x_0, alg, options)

        alg = "Powell"
        options = {"disp": True, "maxfev": maxfev, "ftol": 1e-26}
        res_powell = get_results(func, x_0, alg, options)

        alg = "Nelder-Mead"
        options = {"disp": True, "maxfev": maxfev, "ftol": 1e-26}
        res_nm = get_results(func, x_0, alg, options)

        alg = 'COBYLA'
        options = {"disp": True, "tol": 1e-25, "maxiter": maxfev}
        res_cobyla = get_results(func, x_0, alg, options)

        alg = 'BFGS'
        options = {"maxiter": maxfev, "disp": True, "gtol": 1e-5}
        res_bfgs = get_results(func, x_0, alg, options)

        alg = 'SLSQP'
        options = {"maxiter": maxfev, "disp": True, "ftol": 1e-26}
        res_slsqp = get_results(func, x_0, alg, options)








