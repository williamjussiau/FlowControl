"""
----------------------------------------------------------------------
Run system under multisine excitation
Multiple loops implemented:
    - sine amplitude (A)
    - pure sine frequency (W)
    - sine starting time (T)
    - number of experiments (M) with or without random phase
----------------------------------------------------------------------
"""

from __future__ import print_function
import time
import numpy as np
import main_flowsolver as flo
import utils_flowsolver as flu
import identification_utils as idu
import importlib
import pdb
import getopt
import sys
import json
importlib.reload(flu)
importlib.reload(flo)
importlib.reload(idu)

#from scipy import signal as ss
#import scipy.io as sio 

# FEniCS log level
flo.set_log_level(flo.LogLevel.INFO) # DEBUG TRACE PROGRESS INFO


def main(argv):
    flu.MpiUtils.check_process_rank()

    # Process argv
    ######################################################################
    m_arg = 1
    dry_run = False
    opts, args = getopt.getopt(argv, "M:D") # -M 1, -D
    for opt, arg in opts:
        if opt=='-M':
            m_arg = int(arg)
        if opt=='-D':
            dry_run = True

    # Sleep a bit 
    # to allow reading/writing the same XDMF files from different processes
    #time.sleep(10*m_arg)

    # Base FlowSolver
    ######################################################################
    print('Trying to instantiate FlowSolver...')
    params_flow={'Re': 100.0, 
                 'uinf': 1.0, 
                 'd': 1.0,
                 'sensor_location': np.hstack((np.arange(1, 11).reshape(-1,1), np.zeros((10,1)))), 
                 'sensor_type': 'v'*10,
                 'actuator_angular_size': 10,
                 }
    params_save={'save_every': 2000, 
                 'save_every_old': 2000,
                 'savedir0': '/scratchm/wjussiau/fenics-results/cylinder_o1_ms_1/'} # TODO
    params_solver={'solver_type': 'Krylov', 
                   'equations': 'ipcs',
                   'throw_error': True,
                   'perturbations': True, ####
                   'NL': True, ############## NL=False only works with perturbations=True
                   'init_pert': 0.0,
                   'compute_norms': True}
    params_mesh = flu.make_mesh_param('o1') 
    params_time = {'dt': 0.005, # not recommended to modify dt between runs 
                  'dt_old': 0.005,
                  'restart_order': 1,
                  'Tstart': 200, # start simulation
                  'Trestartfrom': 0, # last file to restart from
                  'num_steps': 40000, # simulate n steps
                  'Tc': 0} # start input

    # NOTE: set seed of np.random here for reproductibility?
    np.random.seed(m_arg) # in slurm: array 1-xxx
    params_save['savedir0'] = '/scratchm/wjussiau/fenics-results/cylinder_o1_ms_' + str(m_arg) + '/' # TODO

    ## Amplitudes of sines (ampl<=2.0 better)
    ampl_list = np.array([1]) # TODO
    
    ## Number of experiments (inner loop) 
    M = 2 # TODO

    ## Frequency
    Fs = 1/params_time['dt']
    Nfreq = 10000 # number of sines # TODO
    ampl_normz = 2/np.sqrt(Nfreq)
    df_u = (Fs/2) / Nfreq
    fbegin = df_u
    fend = (Fs/2) * 0.5 # TODO
    Nfreq_true = int(np.round(fend/df_u))
    # if not all freq, divide things
    idx_in = np.arange(1, Nfreq_true+1)
    freqsin_list = idx_in * df_u
    freqsin_list = freqsin_list.reshape(1, -1) # run all freqs at once 

    ## Time
    T1per = 1/fbegin
    L1per = int(T1per * 1/params_time['dt'])
    # assert(type(L1per) is int)
    Pskip = 4 # TODO
    Psim = 16 # TODO
    # if not all freq, divide things
    Ttot = (Pskip + Psim) * T1per
    Nskip = Pskip * L1per # nr of samples to skip
    Nsim = Psim * L1per # nr of samples to sim
    Ntot = Nskip + Nsim # total nr of samples
    # starting time
    # Tc_list = flu.sample_lco(Tlco=6.0, Tstartlco=200, nsim=48) 
    Tc_list = np.array([params_time['Tstart']]) 
    params_time['num_steps'] = Ntot

    if dry_run:
        print('Dry run only --- With given parameters:')
        print('\t nums steps (nr): ', Ntot)
        print('\t simulation time (s): ', Ttot )
        print('\t amplitude * normz: ', ampl_list[0] * ampl_normz)
        print('\t freq resolution (Hz): ', df_u)
        print('\t file: ', params_save['savedir0'])
        sys.exit()

    # Nested loops
    for ampl in ampl_list:
        print('*** Loop on amplitude: A =', ampl)
        for freqsin in freqsin_list:
            print('*** Loop on frequencies F (single or multisine)')
            for Tc in Tc_list:
                print('*** Loop on Tc: Tc =', Tc)
                params_time['Tc'] = Tc
                for im in range(M):
                    print('*** Loop on experiment: m =', im)
                    fs = flo.FlowSolver(params_flow=params_flow,
                                        params_time=params_time,
                                        params_save=params_save,
                                        params_solver=params_solver,
                                        params_mesh=params_mesh,
                                        verbose=100)
                    print('__init__(): successful!')

                    print('Compute steady state...')
                    u_ctrl_steady = 0.0
                    #fs.compute_steady_state(method='newton', max_iter=25, u_ctrl=u_ctrl_steady)
                    fs.load_steady_state(assign=True)

                    print('Init time-stepping')
                    fs.init_time_stepping()

                    #############################################################
                    print('Define input signal...')
                    phi = 2*np.pi * np.random.rand(*freqsin.shape)
                    u_in = 0
                    tt = fs.Tstart + np.arange(0, fs.num_steps)*fs.dt
                    for ifreq, freq in enumerate(freqsin):
                        u_in += np.sin(2*np.pi * freq * tt + phi[ifreq])
                    u_in *= ampl # amplitude of sine 
                    u_in *= ampl_normz # normalize by sqrt(N)

                    # Define path for saved files
                    savepath = fs.savedir0
                    #fwrite = float(freqsin) if len(freqsin)==1 else len(freqsin)
                    #csvname = 'data_multisine_A={:.2f}_f={:.4f}_T={:.2f}_m={:d}'.format(
                    #    ampl, fwrite, Tc, im)
                    #fwrite = len(freqsin)
                    #csvname = 'data_multisine_A={:.2f}_nf={:d}_m={:d}_arg={:d}'.format(ampl, fwrite, im, m_arg)
                    job_id = str(m_arg) + '_' +  str(im)
                    csvname = 'data_{:s}'.format(job_id) + '.csv'

                    # to replace the dot by a comma:   .replace('.',',')
                    fs.paths['timeseries'] = savepath + csvname
                    ##########################################################


                    ##########################################################
                    # Write summary file info_{job_id}.json
                    summary_dict = {}
                    summary_dict['job_id'] = job_id
                    summary_dict['random_seed'] = m_arg
                    summary_dict['amplitude'] = ampl
                    summary_dict['amplitude_normalization'] = ampl_normz
                    summary_dict['Nrealizations'] = M
                    summary_dict['Fs'] = Fs
                    summary_dict['Nfreq'] = Nfreq
                    summary_dict['Nfreq_true'] = Nfreq_true
                    summary_dict['df_u'] = df_u
                    summary_dict['df_fft'] = Fs/Ntot
                    summary_dict['fbegin'] = fbegin
                    summary_dict['fend'] = fend
                    # leave room for type of freq & oddin/oddnotin/even
                    summary_dict['T1per'] = T1per
                    summary_dict['L1per'] = L1per
                    summary_dict['Pskip'] = Pskip
                    summary_dict['Psim'] = Psim
                    summary_dict['Nperiods'] = Pskip+Psim
                    summary_dict['Ttot'] = Ttot
                    summary_dict['Tstart'] = fs.Tstart
                    summary_dict['dt'] = fs.dt
                    summary_dict['num_steps'] = fs.num_steps
                    summary_dict['Nskip'] = Nskip
                    summary_dict['Nsim'] = Nsim
                    summary_dict['Ntot'] = Ntot
                    summary_dict['idx_in'] = idu.NoIndent(idx_in.tolist())
                    summary_dict['freq_in'] = idu.NoIndent(freqsin.tolist())
                    summary_dict['phase'] = idu.NoIndent(phi.tolist())
                    #summary_dict['time'] = idu.NoIndent(tt.tolist())

                    jsonname = savepath + 'info_{:s}'.format(job_id) + '.json'
                    with open(jsonname, 'w') as jsonfile:
                        json.dump(summary_dict, jsonfile, indent=2, cls=idu.MyEncoder)
                    ##########################################################
 

                    ##########################################################
                    print('Step several times')
                    # Loop over time
                    for i in range(fs.num_steps):
                        # compute control 
                        if fs.t>=fs.Tc:
                            #u_ctrl = ampl * msgen.generate(fs.t - fs.Tc)
                            u_ctrl = u_in[i]

                        # step flow
                        if fs.perturbations:
                            fs.step_perturbation(u_ctrl=u_ctrl, NL=fs.NL, shift=0.0)
                        else:
                            fs.step(u_ctrl)

                    flu.end_simulation(fs)
                    flo.list_timings(flo.TimingClear.clear, [flo.TimingType.wall])
                    fs.write_timeseries()
                    ##########################################################
                   

                    # Clean workspace to prevent memory leaks
                    del fs


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
if __name__=='__main__':
    main(sys.argv[1:])







