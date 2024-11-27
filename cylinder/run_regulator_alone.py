"""
Run SISO regulator with several methods
Check with Matlab
"""

from __future__ import print_function
import time
import numpy as np
import pandas as pd
#import main_flowsolver as flo
#import utils_flowsolver as flu
#import importlib
#importlib.reload(flu)
#importlib.reload(flo)

from scipy import signal as ss
import scipy.io as sio 

import control

import utils_flowsolver as flu

# FEniCS log level
#flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO

if __name__=='__main__':
    t000 = time.time()
    
    print('Run regulator')
    # define controller
    dt = 0.05
    num_steps = 500 
    
    def read_regulator(path):
        '''Read regulator (StateSpace) from provided .mat file path'''
        rd = sio.loadmat(path)
        return rd['A'], rd['B'], rd['C'], rd['D'] 

    #A, B, C, D = read_regulator('/stck/wjussiau/fenics-python/ns/data/nx32/regulator/regulator_struct_D0_KS.mat') 

    A, B, C, D = read_regulator('/stck/wjussiau/fenics-python/ns/data/n1/regulator/K0_o8_D0_S_KS_clpoles1.mat')
    #A = np.array([[-1.101, 0.3733], [0.3733, -0.9561]])
    #B = np.array([[0.7254, -0.06305]]).T
    #C = np.array([0, -0.205])
    #D = np.array([0])

    Kss = ss.StateSpace(A, B, C, D)
    Kss_ = control.StateSpace(A, B, C, D)

    x_ctrl = np.zeros((Kss.A.shape[0],))
    x_ctrl_p = x_ctrl
    x_ctrl_pp = x_ctrl
    xout_ss = x_ctrl
    xout_ct = x_ctrl
    xout_wr = x_ctrl

    y_timeseries = pd.DataFrame(data=np.zeros((num_steps, 4)), columns=['hand', 'ss', 'ct', 'wr'])
    u_timeseries = np.ones((num_steps,))

    import time
    alltimes = dict(th=0, tss=0, tct=0, twr=0)
    for i in range(num_steps):
        # input
        y_meas_err = np.array([ u_timeseries[i] ])

        t0h = time.time()    
        # By hand
        # update controller state (AB1) >> x(t+dt)
        x_ctrl = (Kss.A@(3/2*x_ctrl_p - 1/2*x_ctrl_pp) + Kss.B@y_meas_err)*dt + x_ctrl_p
        # update controller output (no dynamics) >> y(t)=f(x(t))
        y_ab1 = Kss.C@x_ctrl_p + Kss.D@y_meas_err
        u_ctrl = y_ab1
        x_ctrl_pp = x_ctrl_p
        x_ctrl_p  = x_ctrl
        y_timeseries['hand'].iloc[i] = y_ab1
        alltimes['th'] += time.time() - t0h 

        U_arr = np.repeat(y_meas_err, 2)
        T = [0, dt]
        # scipy.signal    
        # start from state x and compute output at times [0, dt]
        t0ss = time.time()
        T_ss, yout_ss, xout_ss = ss.lsim(Kss, U_arr, T, X0=xout_ss, interp=False) # y=y(t)
        xout_ss = xout_ss[1] # this is x(t+dt)
        y_timeseries['ss'].iloc[i] = yout_ss[0]
        alltimes['tss'] += time.time() - t0ss

        # control toolbox
        t0ct = time.time()
        T_ct, yout_ct, xout_ct = control.forced_response(sys=Kss_, U=U_arr, T=T, X0=xout_ct, 
            interpolate=False, return_x=True) # y=y(t) 
        xout_ct = xout_ct[:, 1] # this is x(t+dt)
        y_timeseries['ct'].iloc[i] = yout_ct[0]
        alltimes['tct'] += time.time() - t0ct

        # wrapper
        t0wr = time.time()
        yout_wr, xout_wr = flu.step_controller(Kss_, xout_wr, u_timeseries[i], dt)
        y_timeseries['wr'].iloc[i] = yout_wr
        alltimes['twr'] += time.time() - t0wr


print('Alltimes: ', alltimes)
sio.savemat('./data/temp/y_step.mat', {'y_h': y_timeseries['hand'].to_numpy(),
    'y_ss': y_timeseries['ss'].to_numpy(),
    'y_ct': y_timeseries['ct'].to_numpy(),
    'y_wr': y_timeseries['wr'].to_numpy(),
    'u': u_timeseries})

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




