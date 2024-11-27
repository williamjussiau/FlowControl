"""
----------------------------------------------------------------------
Check restriction of x to subspace for energy computation
----------------------------------------------------------------------
"""

from __future__ import print_function
import time
import numpy as np
import main_flowsolver as flo
import utils_flowsolver as flu
import importlib
importlib.reload(flu)
importlib.reload(flo)

from scipy import signal as ss
import scipy.io as sio 

import sys

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO

if __name__=='__main__':
    t000 = time.time()
    
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.array([[2.5, 0.0]]), # sensor 
                 'sensor_type': 'v', # u, v, p only >> reimplement make_measurement
                 'actuator_angular_size': 10, # actuator angular size
                 }
    params_time={'dt': 0.0025, 
                 'Tstart': 0, 
                 'num_steps': 100, 
                 'Tc': 0.0} 
    params_save={'save_every': 0, 
                 'save_every_old': 0,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_nx32_dummy/',
                 'compute_norms': True}
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': False, #######
                   'NL': True, ################# NL=False only works with perturbations=True
                   'init_pert': 1}
    params_mesh = {'genmesh': True,
                   'remesh': False,
                   'nx': 32,
                   'meshpath': '/stck/wjussiau/fenics-python/mesh/', 
                   'meshname': '',
                   'xinf': 20, #50, # 20
                   'xinfa': -5, #-30, # -5
                   'yinf': 8, #30, # 8
                   'segments': 360}

    fs = flo.FlowSolver(params_flow=params_flow,
                        params_time=params_time,
                        params_save=params_save,
                        params_solver=params_solver,
                        params_mesh=params_mesh,
                        verbose=True)
    print('__init__(): successful!')

    print('Compute steady state...')
    u_ctrl_steady = 0.0
    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
    fs.load_steady_state(assign=True)
    
    print('Init time-stepping')
    fs.init_time_stepping()
 
    print('Step several times')
    u_ctrl = 0.0 # actual control amplitude (initialized)
    u_ctrl0 = 0.0 # base amplitude of gaussian bump
    tlen = 0.5 # characteristic length of gaussian bump
    tpeak = 1 # peak of gaussian bump

    t0 = time.time()
    # define restriction of x
    subdomain_str = 'x[0]<=10 && x[0]>=-2 && x[1]<=3 && x[1]>=-3'
    subdomain = flo.CompiledSubDomain(subdomain_str)
    mesh = fs.mesh
    #boundary_markers = flo.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    #subdomain.mark(boundary_markers, 1)
    #dxi = flo.Measure('dx', domain=mesh, subdomain_data=boundary_markers)

    class IndicatorFunction(flo.UserExpression):
        def __init__(self, subdomain, **kwargs):
            self.subdomain = subdomain
            super(IndicatorFunction, self).__init__(**kwargs)
        def eval(self, values, x):
            values[0] = 0
            values[1] = 0
            if self.subdomain.inside(x, True):
                values[0] = 1
                values[1] = 1
        def value_shape(self):
            return (2,)
    # interpreted 
    IF = IndicatorFunction(subdomain)
    # compiled
    #IF = flo.Expression([subdomain_str]*2, degree=0, element=fs.V.ufl_element())
    # then project on V
    IF = flu.projectm(IF, V=fs.V)

    uif = flo.Function(fs.V)
    uif.vector()[:] = (fs.u_.vector()[:] - 0*fs.u0.vector()[:]) * IF.vector()[:]
    #flu.write_xdmf('u_true.xdmf', fs.u_, 'u_true')
    #flu.write_xdmf('u_restrict.xdmf', uif, 'u_restrict')
    print('time ', time.time() - t0)

    du = flo.Function(fs.V)

    t00 = time.time()
    # then at each step: norm(IF .* x)
    # but we have to define .* first...
    for i in range(fs.num_steps):
        #u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2)
        u_ctrl = u_ctrl0 * np.exp(-1/2*(fs.t-tpeak)**2/tlen**2) * np.sin(20*fs.t)


        du.vector()[:] = (fs.u_.vector()[:] - fs.u0.vector()[:])*IF.vector()[:]
        dE = flo.norm(du, norm_type='L2', mesh=fs.mesh) / fs.Eb

        # step
        if fs.perturbations:
            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
        else:
            fs.step(u_ctrl)

    if fs.num_steps > 3:
        print('Total time is: ', time.time() - t000)
        print('Iteration 1 time     ---', fs.timeseries.loc[1, 'runtime'])
        print('Iteration 2 time     ---', fs.timeseries.loc[2, 'runtime'])
        print('Mean iteration time  ---', np.mean(fs.timeseries.loc[3:, 'runtime']))
        print('Time/iter/dof        ---', np.mean(fs.timeseries.loc[3:, 'runtime'])/fs.W.dim())
    flo.list_timings(flo.TimingClear.clear, [flo.TimingType.wall])
    
    fs.write_timeseries()

    print('total time:', time.time() - t00)
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------




