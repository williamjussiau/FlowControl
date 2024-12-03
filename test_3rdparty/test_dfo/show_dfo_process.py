"""
Show process of DFO-TR optimization
On 1D function
Should be possible by plotting at each iteration the function
evaluation, the trust region and the approximate model
"""
import importlib as il
import numpy as np
import blackbox_opt_modified.bb_optimize as bbo
il.reload(bbo)
## import the function to be optimized
from blackbox_opt_modified.test_funcs.funcs_def import (ackley, sphere, rosen,
                                               booth, bukin, beale)
import matplotlib.pyplot as plt


THEPATH = '/Volumes/Samsung_T5/Travail/ONERA/' \
+ 'Travail/Productions/Avancement/ALL_FILES/' \
+ 'redo_the_same_thing_but_better/stck/fenics-python/test/test_dfo/'




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
    def costfun(x):
        #if x > xmax or x < xmin:
        #    return 1
        #else:
        #print('evaluating: ', x)
        #return ((x[0]-1)**2 + np.cos(3*x[0]))**2 + x[1]**2
        #f = sum([(x[i]-i)**2 for i in range(len(x))]) + 1.2*np.sin(2*x[0])
        f = 0
        f += 1*np.sin(2*x[0])
        f += 0.5*np.sin(4*x[0]+0.02)
        f += 1.2*abs(x[0]+0.3)
        #f += 0.2*x[0]**2
        #f += 0.1*np.sin(4*x[0])
        #print('evaluated:', f)
        return f

    def costfun_array(f, x):
        '''evaluate function f on array x'''
        npt, ndim = x.shape
        fx = np.empty((npt, ))
        for ii in range(npt):
            fx[ii] = f(x[ii, :])
        return fx


    all_func_names = [costfun]

    # Run all the algorithms and problems with given starting points
    # Specify the starting point and options. For example, try the following
    # options.
    for func in all_func_names:
        #x_0 = np.array([1.5])
        x_0 = np.array([0.6])
        #x_0 = np.array([1.5, 0.4])
        #x_0 = np.array([0.5, 0.4])
        #if func in td_func_names:
        #    x_0 = np.array([1.3, 0.7])
        #else:
        #    x_0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

        print("\n\n********* Function " + func.__name__ + "********")
        maxfev = 20

        alg = "DFO"
        options = {"maxfev": maxfev, "init_delta": 0.1,
                   "tol_delta": 1e-25, "tol_f": 1e-26, "tol_norm_g": 1e-5,
                   "sample_gen": "auto", "verbosity": 1}
        res = get_results(func, x_0, alg, options)
        # delta_data = size of trust region
        # H, g_data = gT x + 1/2 xT H x + alpha
        # ndoe in 1D is 3
        ndoe = 3
        neval = res.func_eval - ndoe

        # Plot
        npt = 151
        x = np.linspace(-5, 5, npt)
        fx = costfun_array(costfun, x.reshape((-1,1)))

        for ieval in range(neval):
            # figure
            fig = plt.figure(figsize = (12,10))
            ax = plt.axes()
            # evaluated points
            ax.plot(res.x_data[:, ndoe:ieval+ndoe], res.y_data.T[:, ndoe:ieval+ndoe], 'ro', markersize=12)
            ax.plot(res.x_data[:, :ndoe], res.y_data.T[:, :ndoe], 'go', markersize=12)
            # true fun
            truefun = ax.plot(x, fx, linewidth=3)
            # quadratic models
            # center
            x0m = res.xc_data[ieval][0,0]
            # quadratic model shifted up
            q1 = res.g_data[ieval][0,0]*(x-x0m) + \
                0.5*res.H_data[ieval][0,0]*(x-x0m)**2 + \
                res.f_data[ieval]# + res.val_data[ieval][0,0]
            q1plt = ax.plot(x, q1, linewidth=4)
            # trust region shited to center
            tr = x0m + res.delta_data[ieval]*np.array([-1, 1])
            #ax.plot(tr, np.array([0, 0]), 'k-|', linewidth=2, markersize=8)
            ax.vlines(tr, ymin=-2, ymax=5, colors='g')
            ax.fill_between(tr, -2, 5, alpha=0.2, color='g')
            #xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)
            ax.grid()
            #ax.axis('equal')
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 5])
            #ax.set_xlabel('x', labelpad=20)
            #ax.set_ylabel('y', labelpad=20)
            #ax.set_title('Cost function')
            #plt.show
            #plt.axis('off')
            plt.savefig(THEPATH + 'costfun_img' + str(ieval) + '.png', bbox_inches='tight', pad_inches=0)
            print('Printing image: ', str(ieval))
            plt.close(fig)

            ## model not shifted and result s from optim
            #m1 = res.g_data[ieval][0,0]*x +\
            #    0.5*res.H_data[ieval][0,0]*x**2
            #s = res.s_data[ieval]
            #val = res.val_data[ieval]
            #tr = res.delta_data[ieval]*np.array([-0.75, 0.75])
            #fig = plt.figure(figsize = (12,10))
            #ax = plt.axes()
            #m1plt = ax.plot(x, m1)
            ## trust region not shifted
            #ax.plot(tr, np.repeat(val, 2), 'k-|')
            ## plot minimum
            #ax.plot(s, val, 'ro')
            #ax.grid()
            #ax.set_xlim([-4, 4])
            #ax.set_ylim([-3, 5])
            #ax.set_xlabel('x', labelpad=20)
            #ax.set_ylabel('y', labelpad=20)
            #ax.set_title('Cost function')
            ##plt.show()
            #plt.savefig(THEPATH + 'costfun_img_notshifted' + str(ieval) + '.png')
            #print('Printing image: ', str(ieval))
            #plt.close(fig)
