import scipy.optimize as so
import numpy as np
from scipy.optimize import rosen, rosen_der


if __name__=='__main__':
    def costfun1d(x):
        return (x-1)**2 + np.sin(x) - np.cos(x)

    def costfun(x):
        print('evaluating point: ', x)
        return (x[0]-1)**2 + (x[1]-2)**2


#bounds = (-3, 3)
#method = 'golden'
#options = dict(maxiter=100)
#res1d = so.minimize_scalar(costfun1d, 
#    bounds=bounds, method=method, tol=1e-3, 
#    options=options)
#print('Optim 1D: \n ', res1d)


print(20*'-')
x0 = [1.3, 0.7]
bounds = [(-3, 3)]*2
#res = so.minimize(rosen, x0, method='BFGS', jac=None, tol=1e-3)
#res = so.minimize(rosen, x0, method='trust-constr', tol=1e-3, options=dict(maxiter=100))
#res = so.minimize(rosen, x0, method='COBYLA', tol=1e-3, options=dict(maxiter=100))
res = so.minimize(costfun, x0=x0, method='Nelder-Mead', tol=1e-3, options=dict(maxiter=100,
    return_all=True, disp=True))
#res = so.differential_evolution(rosen, bounds=bounds, updating='immediate', maxiter=100)
#res = so.differential_evolution(rosen, bounds=bounds, updating='deferred', workers=2, maxiter=100)
#res = so.dual_annealing(rosen, bounds=bounds, maxfun=100)
#res = so.minimize(rosen, x0, method='Nelder-Mead', tol=1e-3)
print('Optim nD: \n ', res)













