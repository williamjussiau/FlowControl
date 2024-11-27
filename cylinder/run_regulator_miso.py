"""
Run MISO regulator with several methods
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

    A, B, C, D = read_regulator('/stck/wjussiau/fenics-python/ns/data/n1/regulator/K0_o8_D0_S_KS_clpoles1.mat')

    A = np.array([[-0.4174, 1.0046,   0.2684],
                  [-1.0234, -0.4306,  0.4143],
                  [-0.1843, -0.4580, -0.3376]])
    B = np.array([[-0.7145, -0.5890],
                  [1.351, -0.2938],
                  [0, -0.8479]])
    C = np.array([-1.1201, 2.5260, 1.6555])
    D = np.array([[0, 0]])
    
    Kss = control.StateSpace(A, B, C, D)

    # For log
    y_timeseries = pd.DataFrame(data=np.zeros((num_steps, 3)), columns=['y1', 'y2', 'y1y2'])
    
    # Vecs of 1 and 0
    ONE = np.ones((num_steps,))
    ZERO = np.zeros((num_steps,))

    # Step on channel 1
    xout_wr = np.zeros((Kss.A.shape[0],))
    u_timeseries = np.vstack((ONE, ZERO))
    for i in range(num_steps):
        yout_wr, xout_wr = flu.step_controller(Kss, xout_wr, u_timeseries[:, i], dt)
        y_timeseries['y1'].iloc[i] = yout_wr

    # Step on channel 2
    xout_wr = np.zeros((Kss.A.shape[0],))
    u_timeseries = np.vstack((ZERO, ONE))
    for i in range(num_steps):
        yout_wr, xout_wr = flu.step_controller(Kss, xout_wr, u_timeseries[:, i], dt)
        y_timeseries['y2'].iloc[i] = yout_wr

    # Step on both channels
    xout_wr = np.zeros((Kss.A.shape[0],))
    u_timeseries = np.vstack((ONE, ONE))
    for i in range(num_steps):
        yout_wr, xout_wr = flu.step_controller(Kss, xout_wr, u_timeseries[:, i], dt)
        y_timeseries['y1y2'].iloc[i] = yout_wr

    # Save all
    t = np.linspace(0, (num_steps-1)*dt, num_steps)
    sio.savemat('./data/temp/y_step_miso.mat',
        {'y1': y_timeseries['y1'].to_numpy(),
        'y2': y_timeseries['y2'].to_numpy(),
        'y1y2': y_timeseries['y1y2'].to_numpy(),
        'u': u_timeseries,
        't': t,
        'dt': dt,
        'num_steps': num_steps})

## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------





