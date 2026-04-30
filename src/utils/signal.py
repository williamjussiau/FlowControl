"""Signal processing, array utilities, and multisine signal generation for system identification."""

import json
import logging
import re

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# --- Array utilities ---


def compute_signal_frequency(sig, Tf, dt, nzp=10):
    """Estimate the dominant frequency of a periodic signal via zero-padded FFT.

    Parameters
    ----------
    sig :
        1-D signal array.
    Tf :
        Total simulation time; the first half is discarded as transient.
    dt :
        Sampling interval.
    nzp :
        Zero-padding factor (``nzp * len(sig)`` FFT points).

    Returns
    -------
    float
        Frequency of the dominant spectral peak.
    """
    fftstart = int((Tf / 2) / dt)
    sig_cp = sig[fftstart:]
    sig_cp = sig_cp - np.mean(sig_cp)
    Fs = 1 / dt
    nn = len(sig_cp) * nzp
    frq = np.arange(nn) * Fs / nn
    frq = frq[: len(frq) // 2]
    Y = np.fft.fft(sig_cp, nn) / nn
    Y = Y[: len(Y) // 2]
    return frq[np.argmax(np.abs(Y))]


def sample_lco(Tlco, Tstartlco, nsim):
    """Return sampling times evenly spread over one LCO period.

    Parameters
    ----------
    Tlco :
        Period of the limit cycle oscillation.
    Tstartlco :
        Start time of the LCO region.
    nsim :
        Number of sample points.

    Returns
    -------
    np.ndarray
        Array of ``nsim`` sampling times.
    """
    return Tstartlco + Tlco / nsim * np.arange(nsim)


def pad_upto(L, N, v=0):
    """Pad list or np.ndarray L with v up to N elements."""
    if isinstance(L, list):
        return L + (N - len(L)) * [v]
    if isinstance(L, np.ndarray):
        return np.pad(
            L, pad_width=(0, N - L.shape[0]), mode="constant", constant_values=(v)
        )
    raise TypeError("Type not supported for padding")


def saturate(x, xmin, xmax):
    """Saturate scalar x to [xmin, xmax]."""
    return xmin if x < xmin else xmax if x > xmax else x


# --- Multisine signal generation ---


def _keepfreq_mask(freqsin, Fmin, Fmax, include_fbounds):
    if include_fbounds:
        return (freqsin >= Fmin) & (freqsin <= Fmax)
    return (freqsin > Fmin) & (freqsin < Fmax)


def multisine(
    N: int,
    Fs: float,
    fmin: float,
    fmax: float,
    skip_even: bool = False,
    opt_cf: int = 0,
    plot: bool = False,
    include_fbounds: bool = True,
) -> np.ndarray:
    """One realization of a multisine signal over a period.

    A multisine has constant spectral magnitude over [fmin, fmax] (as fractions of Fs/2).

    Parameters
    ----------
    N :
        Length of one period (samples).
    Fs :
        Sampling frequency.
    fmin :
        Minimum frequency as a fraction of ``Fs/2``.
    fmax :
        Maximum frequency as a fraction of ``Fs/2``.
    skip_even :
        If ``True``, use only odd harmonics.
    opt_cf :
        Number of random trials to minimize crest factor (0 = disabled).
    plot :
        If ``True``, plot time and frequency domain signals.
    include_fbounds :
        If ``True``, include ``fmin`` and ``fmax`` in the frequency set.

    Returns
    -------
    np.ndarray
        One period of the multisine signal, shape ``(N,)``.
    """
    Fmin = np.max([fmin, 0.0]) * Fs / 2
    Fmax = np.min([fmax, 1.0]) * Fs / 2

    skip_even = bool(skip_even)
    freqsin = np.arange(skip_even, N + skip_even, step=1 + skip_even) * Fs / N

    freqsin = freqsin[_keepfreq_mask(freqsin, Fmin, Fmax, include_fbounds)].reshape(-1, 1)
    nf = freqsin.shape[0]
    T = (N - 1) / Fs
    t = np.linspace(0, T, N)

    def make_multisine():
        phi = 2 * np.pi * np.random.rand(*freqsin.shape)
        y = np.sum(np.sin(2 * np.pi * freqsin * t + phi), axis=0)
        return y / np.sqrt(nf)

    y = make_multisine()

    if opt_cf:
        best_cf = crest_factor(y)
        for _ in range(opt_cf):
            ytry = make_multisine()
            cf = crest_factor(ytry)
            if cf < best_cf:
                y = ytry
                best_cf = cf

    if plot:
        plotsignal(y, Fs, t=t, Fmin=Fmin, Fmax=Fmax)

    return y


def multisine_MP(M, P, unwrap=True, **kwargs):
    """Generate M realizations of a multisine tiled over P periods.

    Parameters
    ----------
    M :
        Number of independent realizations.
    P :
        Number of periods to tile each realization.
    unwrap :
        If ``True``, return a 1-D array; otherwise shape ``(M, N*P)``.
    **kwargs :
        Keyword arguments forwarded to :func:`multisine` (must include ``N``).

    Returns
    -------
    np.ndarray
        Signal array, shape ``(M*N*P,)`` if ``unwrap=True``, else ``(M, N*P)``.
    """
    yy = np.zeros((M, kwargs["N"]))
    for im in range(M):
        yy[im, :] = multisine(**kwargs)
    yy = np.tile(yy, (1, P))
    return yy.ravel() if unwrap else yy


def crest_factor(y):
    """Crest factor of y: max(|y|) / rms(y)."""
    return np.max(np.abs(y)) / np.sqrt(np.mean(y**2))


def plotsignal(y, Fs, t=None, Fmin=None, Fmax=None):
    """Plot signal y in time and frequency domains."""
    N = len(y)
    if t is None:
        t = np.linspace(0, (N - 1) / Fs, N)

    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_title("Sum of sines")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()

    nn = N
    mm = 10 * nn
    xx = np.fft.fft(y, nn) / np.sqrt(nn)
    xx_zp = np.fft.fft(y, mm) / np.sqrt(nn)
    ff = np.arange(nn) * Fs / nn
    ff_zp = np.arange(mm) * Fs / mm

    fig, ax = plt.subplots()
    ax.stem(ff, np.abs(xx))
    ax.plot(ff_zp, np.abs(xx_zp), alpha=0.2, color="r")
    if Fmin is not None and Fmax is not None:
        for xline in [Fmin, Fmax]:
            ax.axvline(x=xline, color="k", linestyle="--")
    ax.set_title("TFD & TFSD of sum-of-sines excitation")
    ax.set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    plt.show()


class MultisineGenerator:
    """Multisine signal generator that evaluates sample-by-sample without storing arrays.

    Useful for online signal generation during simulation time loops.
    The signal is periodic with period 1/f0 and can be evaluated at any t."""

    def __init__(
        self,
        N,
        Fs,
        fmin=0.0,
        fmax=1.0,
        skip_even=0,
        include_fbounds=1,
        freqsin=None,
        phi=None,
    ):
        if freqsin is None:
            freqsin = MultisineGenerator.compute_spectrum(
                N=N,
                Fs=Fs,
                fmin=fmin,
                fmax=fmax,
                skip_even=skip_even,
                include_fbounds=include_fbounds,
            )
        nfreq = len(freqsin)
        if phi is None:
            phi = 2 * np.pi * np.random.rand(*freqsin.shape)
        self.nfreq = nfreq
        self.Fs = Fs
        self.freqsin = freqsin
        self.phi = phi

    @staticmethod
    def compute_spectrum(N, Fs, fmin=0.0, fmax=1.0, skip_even=0, include_fbounds=1):
        """Distribute N frequencies equidistantly in [fmin, fmax]*Fs/2."""
        Fmin = np.max([fmin, 0.0]) * Fs / 2
        Fmax = np.min([fmax, 1.0]) * Fs / 2
        freqsin = np.arange(skip_even, N + skip_even, step=1 + skip_even) * Fs / N
        return freqsin[_keepfreq_mask(freqsin, Fmin, Fmax, include_fbounds)]

    @staticmethod
    def compute_harmonics(
        f0, nharm, Fs, fmin=0.0, fmax=1.0, skip_even=0, include_fbounds=1
    ):
        """Harmonics of f0: [f0, 2*f0, ...] clipped to [fmin, fmax]*Fs/2."""
        Fmin = np.max([fmin, 0.0]) * Fs / 2
        Fmax = np.min([fmax, 1.0]) * Fs / 2
        freqsin = f0 * np.arange(skip_even, nharm + skip_even, step=1 + skip_even)
        return freqsin[_keepfreq_mask(freqsin, Fmin, Fmax, include_fbounds)]

    def generate(self, t, vectorized=True):
        """Evaluate multisine signal at time t."""
        if vectorized:
            return np.sum(
                np.sin(2 * np.pi * self.freqsin * t + self.phi)
            ) / np.sqrt(self.nfreq)
        S = sum(
            np.sin(2 * np.pi * self.freqsin[i] * t + self.phi[i])
            for i in range(self.nfreq)
        )
        return S / np.sqrt(self.nfreq)


# --- JSON export helpers for multisine data ---


class NoIndent:
    """Wrap a list/tuple to prevent indentation in MyEncoder output."""

    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("Only lists and tuples can be wrapped")
        self.value = value


class MyEncoder(json.JSONEncoder):
    """JSON encoder that serializes NoIndent-wrapped lists on a single line.

    Usage::

        with open(path, "w") as f:
            json.dump({"data": NoIndent(my_list)}, f, cls=MyEncoder, indent=2)
    """

    FORMAT_SPEC = "@@{}@@"
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))

    def __init__(self, **kwargs):
        ignore = {"cls", "indent"}
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        self._registry = {}
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, NoIndent):
            key = id(obj)
            self._registry[key] = obj
            return self.FORMAT_SPEC.format(key)

    def iterencode(self, obj, **kwargs):
        self._registry.clear()
        for encoded in super().iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                obj_id = int(match.group(1))
                no_indent = self._registry[obj_id]
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                encoded = encoded.replace(
                    '"{}"'.format(self.FORMAT_SPEC.format(obj_id)), json_repr
                )
            yield encoded
        self._registry.clear()
