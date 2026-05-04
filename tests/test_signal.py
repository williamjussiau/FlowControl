"""Tests for utils.signal."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from utils.signal import (
    MultisineGenerator,
    MyEncoder,
    NoIndent,
    compute_signal_frequency,
    crest_factor,
    multisine,
    multisine_MP,
    pad_upto,
    sample_lco,
    saturate,
)


# ── pad_upto ──────────────────────────────────────────────────────────────────


class TestPadUpto:
    def test_list_pads_with_zero(self):
        assert pad_upto([1, 2], 4) == [1, 2, 0, 0]

    def test_list_pads_with_custom_value(self):
        assert pad_upto([1], 3, v=9) == [1, 9, 9]

    def test_list_already_full_length(self):
        assert pad_upto([1, 2, 3], 3) == [1, 2, 3]

    def test_ndarray_pads_with_zero(self):
        assert_allclose(pad_upto(np.array([1.0, 2.0]), 4), [1.0, 2.0, 0.0, 0.0])

    def test_ndarray_pads_with_custom_value(self):
        assert_allclose(pad_upto(np.array([5.0]), 3, v=-1.0), [5.0, -1.0, -1.0])

    def test_ndarray_already_full_length(self):
        a = np.array([1.0, 2.0])
        assert_allclose(pad_upto(a, 2), a)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            pad_upto((1, 2), 4)


# ── saturate ─────────────────────────────────────────────────────────────────


class TestSaturate:
    @pytest.mark.parametrize("x,expected", [
        (-5.0, -1.0),
        (-1.0, -1.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (10.0, 1.0),
    ])
    def test_clamp(self, x, expected):
        assert saturate(x, -1.0, 1.0) == expected


# ── sample_lco ────────────────────────────────────────────────────────────────


class TestSampleLco:
    def test_shape(self):
        t = sample_lco(Tlco=1.0, Tstartlco=5.0, nsim=10)
        assert t.shape == (10,)

    def test_first_element_equals_start(self):
        t = sample_lco(Tlco=1.0, Tstartlco=5.0, nsim=10)
        assert_allclose(t[0], 5.0)

    def test_uniform_spacing(self):
        Tlco, nsim = 2.0, 8
        t = sample_lco(Tlco=Tlco, Tstartlco=0.0, nsim=nsim)
        assert_allclose(np.diff(t), Tlco / nsim)

    def test_does_not_reach_next_period(self):
        Tlco, Tstart, nsim = 1.0, 0.0, 4
        t = sample_lco(Tlco, Tstart, nsim)
        assert t[-1] < Tstart + Tlco


# ── crest_factor ──────────────────────────────────────────────────────────────


class TestCrestFactor:
    def test_constant_signal(self):
        y = np.ones(100)
        assert crest_factor(y) == pytest.approx(1.0)

    def test_pure_sine(self):
        t = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
        y = np.sin(t)
        assert crest_factor(y) == pytest.approx(np.sqrt(2), rel=1e-3)

    def test_higher_than_sine_for_impulse_like_signal(self):
        """A signal with one large spike has a higher CF than a pure sine."""
        y = np.zeros(100)
        y[0] = 1.0
        assert crest_factor(y) > np.sqrt(2)


# ── multisine ─────────────────────────────────────────────────────────────────


class TestMultisine:
    def test_output_length(self):
        y = multisine(N=256, Fs=100.0, fmin=0.1, fmax=0.5)
        assert len(y) == 256

    def test_zero_mean(self):
        y = multisine(N=512, Fs=100.0, fmin=0.1, fmax=0.9)
        assert_allclose(np.mean(y), 0.0, atol=1e-10)

    def test_energy_in_band(self):
        """At least 99 % of spectral energy lies within [fmin, fmax]*Fs/2."""
        N, Fs, fmin, fmax = 512, 100.0, 0.1, 0.5
        y = multisine(N=N, Fs=Fs, fmin=fmin, fmax=fmax)
        freqs = np.fft.rfftfreq(N, d=1.0 / Fs)
        power = np.abs(np.fft.rfft(y)) ** 2
        Fmin, Fmax = fmin * Fs / 2, fmax * Fs / 2
        in_band = power[(freqs >= Fmin - 0.5) & (freqs <= Fmax + 0.5)].sum()
        assert in_band / power.sum() > 0.99

    def test_skip_even_zero_at_even_harmonics(self):
        """skip_even=True: FFT bins at even multiples of Fs/N must be zero."""
        N, Fs = 64, 100.0
        rng_state = np.random.get_state()
        np.random.seed(0)
        try:
            y = multisine(N=N, Fs=Fs, fmin=0.0, fmax=1.0, skip_even=True)
        finally:
            np.random.set_state(rng_state)
        spectrum = np.abs(np.fft.rfft(y))
        for k in range(2, N // 2 + 1, 2):
            assert spectrum[k] < 1e-10, f"even harmonic {k} has energy {spectrum[k]}"

    def test_exclude_fbounds(self):
        """include_fbounds=False: boundary frequencies must have zero energy."""
        N, Fs, fmin, fmax = 256, 100.0, 0.1, 0.5
        y = multisine(N=N, Fs=Fs, fmin=fmin, fmax=fmax, include_fbounds=False)
        freqs = np.fft.rfftfreq(N, d=1.0 / Fs)
        spectrum = np.abs(np.fft.rfft(y))
        Fmin, Fmax = fmin * Fs / 2, fmax * Fs / 2
        for k, f in enumerate(freqs):
            if np.isclose(f, Fmin) or np.isclose(f, Fmax):
                assert spectrum[k] < 1e-10, f"boundary freq {f} Hz has energy"


# ── multisine_MP ──────────────────────────────────────────────────────────────


class TestMultisineMP:
    def test_shape_unwrapped(self):
        M, P, N = 3, 2, 64
        y = multisine_MP(M=M, P=P, N=N, Fs=100.0, fmin=0.1, fmax=0.9)
        assert y.shape == (M * P * N,)

    def test_shape_not_unwrapped(self):
        M, P, N = 3, 2, 64
        y = multisine_MP(M=M, P=P, unwrap=False, N=N, Fs=100.0, fmin=0.1, fmax=0.9)
        assert y.shape == (M, P * N)

    def test_tiling_is_periodic(self):
        """Each row (one realization) must be periodic: y[0:N] == y[N:2N]."""
        M, P, N = 2, 3, 128
        y = multisine_MP(M=M, P=P, unwrap=False, N=N, Fs=100.0, fmin=0.1, fmax=0.9)
        assert_allclose(y[0, :N], y[0, N : 2 * N], atol=1e-12)

    def test_realizations_differ(self):
        M, N = 3, 64
        y = multisine_MP(M=M, P=1, unwrap=False, N=N, Fs=100.0, fmin=0.1, fmax=0.9)
        assert not np.allclose(y[0], y[1])
        assert not np.allclose(y[1], y[2])


# ── MultisineGenerator ────────────────────────────────────────────────────────


class TestMultisineGenerator:
    @pytest.fixture
    def gen_fixed(self):
        freqsin = np.array([5.0, 10.0, 20.0])
        phi = np.array([0.1, 0.5, 1.2])
        return MultisineGenerator(N=128, Fs=100.0, freqsin=freqsin, phi=phi)

    def test_generate_matches_analytic_formula(self, gen_fixed):
        t = 0.123
        expected = np.sum(
            np.sin(2 * np.pi * gen_fixed.freqsin * t + gen_fixed.phi)
        ) / np.sqrt(gen_fixed.nfreq)
        assert gen_fixed.generate(t) == pytest.approx(expected)

    def test_vectorized_matches_non_vectorized(self, gen_fixed):
        t = 0.05
        assert gen_fixed.generate(t, vectorized=True) == pytest.approx(
            gen_fixed.generate(t, vectorized=False)
        )

    def test_nfreq_matches_spectrum_size(self):
        N, Fs = 256, 100.0
        gen = MultisineGenerator(N=N, Fs=Fs, fmin=0.1, fmax=0.5)
        expected = MultisineGenerator.compute_spectrum(N, Fs, fmin=0.1, fmax=0.5)
        assert gen.nfreq == len(expected)

    def test_compute_harmonics_all_multiples_of_f0(self):
        f0, nharm, Fs = 10.0, 8, 200.0
        freqs = MultisineGenerator.compute_harmonics(f0, nharm, Fs)
        remainders = freqs % f0
        assert_allclose(remainders, 0.0, atol=1e-10)

    def test_compute_harmonics_in_bounds(self):
        f0, nharm, Fs = 10.0, 20, 200.0
        freqs = MultisineGenerator.compute_harmonics(f0, nharm, Fs)
        assert np.all(freqs <= Fs / 2)


# ── compute_signal_frequency ──────────────────────────────────────────────────


class TestComputeSignalFrequency:
    def _make_sine(self, freq, dt, n_total):
        """Pure sine of given frequency sampled at dt for n_total steps."""
        t = np.arange(n_total) * dt
        return np.sin(2 * np.pi * freq * t)

    def test_recovers_known_frequency(self):
        dt = 0.01
        freq = 5.0
        n_total = 2000
        sig = self._make_sine(freq, dt, n_total)
        Tf = n_total * dt
        result = compute_signal_frequency(sig, Tf=Tf, dt=dt)
        assert result == pytest.approx(freq, rel=1e-2)

    def test_discards_transient_half(self):
        """Signal starts at wrong frequency, switches at Tf/2; correct freq recovered."""
        dt = 0.01
        n_total = 2000
        Tf = n_total * dt
        t = np.arange(n_total) * dt
        sig = np.where(t < Tf / 2, np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 7.0 * t))
        result = compute_signal_frequency(sig, Tf=Tf, dt=dt)
        assert result == pytest.approx(7.0, rel=0.05)

    def test_returns_float(self):
        dt = 0.01
        sig = self._make_sine(3.0, dt, 1000)
        result = compute_signal_frequency(sig, Tf=1000 * dt, dt=dt)
        assert isinstance(result, float)


# ── MyEncoder ─────────────────────────────────────────────────────────────────


class TestMyEncoder:
    def test_noindent_list_on_single_line(self):
        """NoIndent-wrapped lists must appear on one line in the JSON output."""
        data = {"values": NoIndent([1, 2, 3])}
        encoded = json.dumps(data, cls=MyEncoder, indent=2)
        # The list should appear as [1, 2, 3] not spread over multiple indented lines
        assert "[1, 2, 3]" in encoded

    def test_normal_dict_still_indented(self):
        """Non-wrapped objects must still be formatted with the requested indent."""
        data = {"a": 1, "b": 2}
        encoded = json.dumps(data, cls=MyEncoder, indent=2)
        assert "\n" in encoded

    def test_numpy_scalar_serialized(self):
        """np.float64 values must serialize without TypeError."""
        data = {"v": np.float64(3.14)}
        encoded = json.dumps(data, cls=MyEncoder)
        assert json.loads(encoded)["v"] == pytest.approx(3.14)

    def test_roundtrip_values_preserved(self):
        data = {"nums": NoIndent([10, 20, 30])}
        encoded = json.dumps(data, cls=MyEncoder, indent=2)
        decoded = json.loads(encoded)
        assert decoded["nums"] == [10, 20, 30]


# ── NoIndent ──────────────────────────────────────────────────────────────────


class TestNoIndent:
    def test_accepts_list(self):
        obj = NoIndent([1, 2, 3])
        assert obj.value == [1, 2, 3]

    def test_accepts_tuple(self):
        obj = NoIndent((1, 2))
        assert obj.value == (1, 2)

    def test_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            NoIndent(42)
