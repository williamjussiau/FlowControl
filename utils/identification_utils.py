# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:24:13 2022

@author: wjussiau
"""
############ Identification utils ############
import numpy as np
import matplotlib.pyplot as plt
from _ctypes import PyObj_FromPtr
import json
import re
plt.style.use('seaborn')

import time
import pdb

# TODO
# - non-constant spectrum
# - improve frequency skip (e.g. randomly)
# - normalize? RMS, max abs?

def multisine(N:int, Fs:float, fmin:float, fmax:float, 
              skip_even:bool=False, opt_cf:int=0, plot:bool=False,
              include_fbounds:bool=True) -> np.array:
    """One realization of a multisine signal over a period
    
    For identification, one might need several realizations and periods
    A multisine signal is characterized by a spectrum with constant magnitude 
    on a range of frequencies
    
    Parameters
    ----------
    N : int
        length of one period of signal
    Fs : float
        sampling frequency
    fmin, fmax : float
        min/max freq (in percent of Fe/2)
    skip_even : bool, optionnal
        only odd frequencies
    opt_cf : int, optionnal
        optimize crest factor (i.e. less outliers in y)
    plot : bool, optionnal
        to display
    include_fbounds : bool, optionnal
        include or exclude fmin/fmax when removing frequencies,
        especially useful for f=0
    
    Returns
    -------
    y : np.array
        multisine signal"""
        
    # fmin, fmax as ratios of Fs/2
    Fmin = np.max([fmin, 0.])*Fs/2
    Fmax = np.min([fmax, 1.])*Fs/2

    # Freq
    # if skip_even=True, then step=2 starting from 1 --> odd harmonics
#    if skip_even:
#        freqsin = np.arange(1, N+1, step=2) * Fs/N
#    else:
#        freqsin = np.arange(0, N, step=1) * Fs/N 
    # 'compact' notation
    skip_even = bool(skip_even)
    freqsin = np.arange(skip_even, N+skip_even, step=1+skip_even) * Fs/N
        
    if include_fbounds:
        keepfreq = (freqsin>=Fmin) * (freqsin<=Fmax)
    else:
        keepfreq = (freqsin>Fmin) * (freqsin<Fmax)

    freqsin = freqsin[keepfreq]
    freqsin = freqsin.reshape(-1, 1)

    # Time
    T = 1/Fs * (N-1)
    t = np.linspace(0, T, N) #.reshape(1, -1)

    def make_multisine():
        """Utility function to make multisine series"""
        # Uniform random phases in [0, 2pi]
        # phi = np.linspace(0.1, 2*np.pi, len(freqsin), endpoint=False)
        #phi = np.random.permutation(phi.reshape(freqsin.shape))
        #t1 = time.time()
        phi = 2*np.pi * np.random.rand(*freqsin.shape)
        #tel1 = time.time() - t1
        #t2 = time.time()
        #allsin = 2/np.sqrt(N) * np.sin(2*np.pi * freqsin*t + phi)
        nf = freqsin.shape[0] 
        multisin = np.zeros_like(t)
        for i in range(nf):
            multisin += np.sin(2*np.pi * freqsin[i]*t + phi[i])
        multisin *= 2/np.sqrt(N)
        #tel2 = time.time() - t2
        #print('time elapsed for random phase:', tel1) 
        #print('time elapsed for array(sin(f*t)):', tel2) 
        return multisin
        #return np.sum(allsin, 0), allsin

    y = make_multisine()

    if opt_cf:
        best_cf = crest_factor(y)
        for _ in range(np.max([opt_cf, 10])):  # at least 10 iter
            # Generate new series
            ytry = make_multisine()
            # If crest factor lower, replace series
            cf = crest_factor(ytry)
            if cf < best_cf:
                y = ytry
                best_cf = cf
    
    # Plot
    if plot:
        # Time domain
        fig, ax = plt.subplots()
                
        ax.plot(t, y)
        ax.set_title('Sum of sines')
        ax.set_xlabel('Time (s)')
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.show()
        
        # Frequency domain
        nn = N
        mm = 10*nn
        xx = np.fft.fft(y, nn) / np.sqrt(nn)
        xx_zp = np.fft.fft(y, mm) / np.sqrt(nn)
        ff = np.arange(nn) * Fs/nn
        ff_zp = np.arange(mm) * Fs/mm
        
        fig, ax = plt.subplots()
        
        ax.stem(ff, np.abs(xx), use_line_collection=True)
        ax.plot(ff_zp, np.abs(xx_zp), alpha=0.2, color='r')
        for xline in [Fmin, Fmax]:
            ax.axvline(x=xline, color='k', linestyle='--')    
        ax.set_title('TFD & TFSD of sum-of-sines excitation')
        ax.set_xlabel('Frequency (Hz)')
        #ax.set_xlim([0, Fs/2])
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.show()    
    
    return y
    

def multisine_MP(M, P, unwrap=True, **kwargs):
    """Define M realizations of a multisine excitation on P periods
    
    M: number of realizations
    P: number of periods
    unwrap: realizations are unwrapped and returned as a 1D vector
    **kwargs: see multisine()"""   
    yy = np.zeros((M, kwargs['N']))
    for im in range(M):
        yy[im, :] = multisine(**kwargs)
    yy = np.tile(yy, (1, P))
    yy = yy.ravel() if unwrap else yy
    return yy


def crest_factor(y):
    """Return crest factor of series y, defined as max(abs(y))/rms(y)
    Proportional to norm(y, inf)/norm(y, 2)"""
    return np.max(np.abs(y))/np.sqrt(np.mean(y**2))


def plotsignal(y, Fs, t=None, Fmin=None, Fmax=None):
    """Represent (multisine) signal in time and frequency spaces"""
#    import pdb
#    pdb.set_trace()
    
    N = len(y)
    if t is None:
        T = 1/Fs * (N-1)
        t = np.linspace(0, T, N)
    
    # Time domain
    fig, ax = plt.subplots()
    
    ax.plot(t, y)
    ax.set_title('Sum of sines')
    ax.set_xlabel('Time (s)')
    
    fig.tight_layout()
    plt.show()
    
    # Frequency domain   
    nn = N
    mm = 10*nn
    xx = np.fft.fft(y, nn) / np.sqrt(nn)
    xx_zp = np.fft.fft(y, mm) / np.sqrt(nn)
    ff = np.arange(nn) * Fs/nn
    ff_zp = np.arange(mm) * Fs/mm
    
    fig, ax = plt.subplots()
    
    ax.stem(ff, np.abs(xx), use_line_collection=True)
    ax.plot(ff_zp, np.abs(xx_zp), alpha=0.2, color='r')
    if Fmin is not None and Fmax is not None:
        for xline in [Fmin, Fmax]:
            ax.axvline(x=xline, color='k', linestyle='--')    
    ax.set_title('TFD & TFSD of sum-of-sines excitation')
    ax.set_xlabel('Frequency (Hz)')
    #ax.set_xlim([0, Fs/2])
    
    fig.tight_layout()
    plt.show() 
        
    
class multisin_generator():
    """Class for generating multisine signals without storing complete 
    arrays of values
    
    The class is intended as follows: create an object with specific 
    properties (Fs, N...) and use generate(t) to create the signal at a
    specific time step. Note that a signal of intended period T may be 
    evaluated at times t>=T and indeed shows said periodicity."""
    def __init__(self,
                 N, Fs, fmin=0.0, fmax=1.0,
                 skip_even=0, include_fbounds=1,
                 freqsin=None, phi=None):
        """ Parameters
            ----------
            N : int
                number of frequencies in [0, Fs] (be careful)
            Fs : float
                sampling frequency
            fmin, fmax : float
                min/max freq (in percent of Fs/2)
            skip_even : bool, optionnal
                only odd frequencies
            include_fbounds : bool, optionnal
                include or exclude fmin/fmax when removing frequencies,
                especially useful for f=0
            freqsin : array (size N), optionnal
                give frequencies of multisines as input
            phi : array (size N), optionnal
                give phase of multisines as input"""
        # Frequencies to be included
        if freqsin is None:
            freqsin = multisin_generator.compute_spectrum(N=N, Fs=Fs, 
                                                          fmin=fmin, fmax=fmax, 
                                                          skip_even=skip_even, 
                                                          include_fbounds=include_fbounds)
        N = len(freqsin)

        # Random phases
        if phi is None:
            phi = 2*np.pi * np.random.rand(*freqsin.shape)
        
        # Assign all
        self.N = N
        self.Fs = Fs
        self.freqsin = freqsin
        self.phi = phi
        
        
    @staticmethod
    def compute_spectrum(N, Fs, fmin=0.0, fmax=1.0, 
                         skip_even=0, include_fbounds=1):
        '''Distribute N frequencies equidistantly in [fmin, fmax]*Fs/2'''
        Fmin = np.max([fmin, 0.])*Fs/2
        Fmax = np.min([fmax, 1.])*Fs/2  
        freqsin = np.arange(skip_even, N+skip_even, step=1+skip_even) * Fs/N 
        if include_fbounds:
            keepfreq = (freqsin>=Fmin) * (freqsin<=Fmax)
        else:
            keepfreq = (freqsin>Fmin) * (freqsin<Fmax)
        freqsin = freqsin[keepfreq]
        #freqsin = freqsin.reshape(-1, 1)
        return freqsin
    
    @staticmethod
    def compute_harmonics(f0, N, Fs, fmin=0.0, fmax=1.0,
                          skip_even=0, include_fbounds=1):
        '''Distribute harmonics of f0 : [f0, 2*f0,...] and ensure less than
        half of sampling frequency, and between given bounds
        Kina the same as compute_spectrum but another philosophy'''
        Fmin = np.max([fmin, 0.])*Fs/2
        Fmax = np.min([fmax, 1.])*Fs/2  
        freqsin = f0 * np.arange(skip_even, N+skip_even, step=1+skip_even)
        if include_fbounds:
            keepfreq = (freqsin>=Fmin) * (freqsin<=Fmax)
        else:
            keepfreq = (freqsin>Fmin) * (freqsin<Fmax)
        freqsin = freqsin[keepfreq]
        #freqsin = freqsin.reshape(-1, 1)
        return freqsin

        
    def generate(self, t, vectorized=True):
        """Generate output of multisine signal at time step t"""
        if vectorized:
            S = np.sum(np.sin(2*np.pi * self.freqsin * t + self.phi))
        else:
            S = 0
            for i in range(self.N):
                S += np.sin(2*np.pi * self.freqsin[i] * t + self.phi[i])
        return S * 1/np.sqrt(self.N)


# Dump JSON file without indenting lists
# Adapted from: https://stackoverflow.com/questions/42710879/write-two-dimensional-list-to-json-file
class NoIndent(object):
    """Value wrapper: object becomes NoIndent class with value field"""
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value


class MyEncoder(json.JSONEncoder):
    """Encoder for not-indenting list-like objects
    Usage: 
        - make dictionary and mark objects to not be indented with NoIndent class
            dict(list_to_dump=idu.NoIdent(my_list))
        - dump dictionary to file using MyEncoder as encoder
            with open(...) as jsonfile:
                json.dump(..., cls=idu.MyEncoder)
    """
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        '''Default encoder properties'''
        # Encode numpy as Python types
        if isinstance(obj, np.generic):
            return obj.item()
        #if isinstance(obj, np.ndarray):
        #    return self.FORMAT_SPEC.format(id(obj.tolist()))
        if isinstance(obj, NoIndent):
            return self.FORMAT_SPEC.format(id(obj)) 
        #return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
        #            else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded


if __name__=="__main__":
    N = 50
    Fs = 10
    fmin = 0.
    fmax = 1.

    skip_even = 0
    include_fbounds = 1

    t0 = time.time()
    y = multisine(N=N, Fs=Fs, fmin=fmin, fmax=fmax, 
                  skip_even=skip_even, opt_cf=0, plot=0,
                  include_fbounds=include_fbounds)
    plotsignal(y, Fs)
    telapsed = time.time() - t0
    print('Multisine array -- Elapsed: ', telapsed)
    #
    #M = 1
    #P = 3
    #yy = multisine_MP(M=M, P=P, N=N, Fs=Fs, fmin=fmin, fmax=fmax, 
    #                  skip_even=0, opt_cf=0, plot=False,
    #                  unwrap=True, include_fbounds=1)
    #plotsignal(yy, Fs)
    
    ## Test multisine generator
    t1 = time.time()
    # Create object
    MG = multisin_generator(N=10, Fs=Fs, fmin=fmin, fmax=fmax,
                            skip_even=skip_even, 
                            include_fbounds=include_fbounds)
    # Define timestamps
    T = 1/Fs * (N-1)
    t = np.linspace(0, T, N) #.reshape(1, -1)
    # Generate output at each time
    y_mg = np.zeros((N,))
    for i in range(N):
        y_mg[i] = MG.generate(t[i], vectorized=True)
    # Show time series and fft
    plotsignal(y_mg, Fs, t)
    telapsed = time.time() - t1
    print('Multisine generator (vectorized) -- Elapsed: ', telapsed)


    ## Test periodicity of multisine generator
    # Create object
    freqsin = np.array([0.3333, 0.5, 1.5, 2])
    freqsin = np.arange(0.5, 1.5, 0.3)
    f0 = 0.125
    #freqsin = f0 * np.arange(1, 7) 
    freqsin = multisin_generator.compute_harmonics(f0=f0, N=200, Fs=Fs,
                                                   fmin=0., fmax=0.5,
                                                   include_fbounds=False)
    #freqsin = np.round(np.logspace(-1, np.log10(3), num=30), decimals=2)
    #freqsin = np.array([0.5])
    #freqsin = multisin_generator.compute_spectrum(N=80, Fs=Fs, fmin=0., fmax=0.4,
    #                                              skip_even=0, include_fbounds=1)
    Tf0 = 1/f0
    P = 10
    #freqsin = None
    phi = None
    #freqsin = np.array([0.04])
    MG = multisin_generator(N=12, Fs=10, fmin=0., fmax=1.0,
                            skip_even=False, 
                            include_fbounds=True,
                            freqsin=freqsin, phi=phi)
    # Define timestamps
    T = P*Tf0
    t0 = 0
    t = np.arange(start=t0, stop=t0+T, step=0.1)
    DT = 0.0
    nt = t.shape[0]
    # Generate output at each time
    y_mg = np.zeros((nt,))
    for i in range(nt):
        y_mg[i] = MG.generate(t[i]-t0+DT, vectorized=True)
    # Show time series and fft
    plotsignal(y_mg, Fs, t)
    telapsed = time.time() - t1
    print('Multisine generator (vectorized) -- Elapsed: ', telapsed)












