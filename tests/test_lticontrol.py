"""Unit tests for utils.lticontrol.

Covers: state-space algebra, stability, coprime factorizations, Youla parametrization,
Laguerre basis, balanced reduction, controller residues, slow-fast decomposition, I/O.
"""

import tempfile
from pathlib import Path

import control
import numpy as np
import pytest
import scipy.linalg as la

from utils.lticontrol import (
    balreal,
    basis_laguerre,
    basis_laguerre_canonical,
    basis_laguerre_canonical_ss,
    basis_laguerre_ss,
    controller_residues,
    controller_residues_getidx,
    controller_residues_wrapper,
    isstable,
    isstablecl,
    lncf,
    lqg_regulator,
    read_ss,
    rncf,
    slowfast,
    ss_blkdiag_list,
    ss_hstack,
    ss_hstack_list,
    ss_inv,
    ss_one,
    ss_transpose,
    ss_vstack,
    ss_vstack_list,
    ss_zero,
    sys_hsv,
    write_ss,
    youla,
    youla_laguerre,
    youla_laguerre_K00,
    youla_lqg,
    youla_Q0b,
    youla_Qab,
)

# ---------------------------------------------------------------------------
# Shared test systems
# ---------------------------------------------------------------------------

FREQS = np.logspace(-2, 2, 30)


def _freqresp(G, freqs=FREQS):
    """Evaluate frequency response; returns complex array (nout, nin, nfreq)."""
    return control.frequency_response(G, omega=np.asarray(freqs)).fresp


def _siso_first_order():
    return control.ss([[-1.0]], [[1.0]], [[1.0]], [[0.0]])


def _siso_second_order():
    A = np.array([[-1.0, -2.0], [1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0]])
    return control.ss(A, B, C, D)


def _siso_with_feedthrough():
    """Stable SISO with non-zero D — required for ss_inv tests."""
    return control.ss([[-1.0]], [[1.0]], [[1.0]], [[2.0]])


def _simo_2x1():
    """Stable SIMO plant: 2 outputs, 1 input."""
    A = np.diag([-1.0, -2.0, -3.0])
    B = np.array([[1.0], [0.0], [1.0]])
    C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    D = np.zeros((2, 1))
    return control.ss(A, B, C, D)


def _plant_and_K0(Qx=1.0, Ru=1.0, Qw=1.0, Rv=1.0):
    G = _siso_second_order()
    K0, _, _ = lqg_regulator(G, Qx, Ru, Qw, Rv)
    return G, K0


# ---------------------------------------------------------------------------
# 1. State-space algebra
# ---------------------------------------------------------------------------


class TestSsZeroOne:
    def test_ss_zero_shape(self):
        G = ss_zero()
        assert G.noutputs == 1 and G.ninputs == 1

    def test_ss_zero_dcgain(self):
        assert float(control.dcgain(ss_zero())) == pytest.approx(0.0)

    def test_ss_one_shape(self):
        G = ss_one()
        assert G.noutputs == 1 and G.ninputs == 1

    def test_ss_one_dcgain(self):
        assert float(control.dcgain(ss_one())) == pytest.approx(1.0)

    def test_ss_one_no_dynamics(self):
        assert ss_one().nstates == 0


class TestSsInv:
    def test_inv_roundtrip(self):
        """G * inv(G) should have magnitude 1 at all frequencies (requires D invertible)."""
        G = _siso_with_feedthrough()
        fr = _freqresp(G * ss_inv(G))
        assert np.allclose(np.abs(fr), 1.0, atol=1e-8)

    def test_inv_dcgain(self):
        G = _siso_with_feedthrough()
        assert float(control.dcgain(ss_inv(G))) == pytest.approx(
            1.0 / float(control.dcgain(G)), rel=1e-8
        )


class TestSsTranspose:
    @pytest.mark.parametrize("G", [_siso_first_order(), _siso_second_order(), _simo_2x1()])
    def test_double_transpose_is_identity(self, G):
        fr1 = _freqresp(G)
        fr2 = _freqresp(ss_transpose(ss_transpose(G)))
        assert np.allclose(fr1, fr2, atol=1e-10)

    def test_matrix_relations(self):
        G = _siso_second_order()
        GT = ss_transpose(G)
        assert np.allclose(GT.A, G.A.T)
        assert np.allclose(GT.B, G.C.T)
        assert np.allclose(GT.C, G.B.T)
        assert np.allclose(GT.D, G.D.T)


class TestSsStack:
    def test_vstack_output_dimension(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        G = ss_vstack(G1, G2)
        assert G.noutputs == G1.noutputs + G2.noutputs
        assert G.ninputs == G1.ninputs

    def test_hstack_input_dimension(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        G = ss_hstack(G1, G2)
        assert G.noutputs == G1.noutputs
        assert G.ninputs == G1.ninputs + G2.ninputs

    def test_vstack_list_matches_vstack(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        fr1 = _freqresp(ss_vstack(G1, G2))
        fr2 = _freqresp(ss_vstack_list([G1, G2]))
        assert np.allclose(fr1, fr2, atol=1e-10)

    def test_hstack_list_matches_hstack(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        fr1 = _freqresp(ss_hstack(G1, G2))
        fr2 = _freqresp(ss_hstack_list([G1, G2]))
        assert np.allclose(fr1, fr2, atol=1e-10)

    def test_vstack_freqresp_rows(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        Gv = ss_vstack(G1, G2)
        fr = _freqresp(Gv)
        assert np.allclose(fr[0, 0, :], _freqresp(G1)[0, 0, :], atol=1e-10)
        assert np.allclose(fr[1, 0, :], _freqresp(G2)[0, 0, :], atol=1e-10)

    def test_hstack_freqresp_cols(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        Gh = ss_hstack(G1, G2)
        fr = _freqresp(Gh)
        assert np.allclose(fr[0, 0, :], _freqresp(G1)[0, 0, :], atol=1e-10)
        assert np.allclose(fr[0, 1, :], _freqresp(G2)[0, 0, :], atol=1e-10)

    def test_blkdiag_no_cross_coupling(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        Gd = ss_blkdiag_list([G1, G2])
        fr = _freqresp(Gd)
        assert np.allclose(fr[0, 1, :], 0.0, atol=1e-12)
        assert np.allclose(fr[1, 0, :], 0.0, atol=1e-12)

    def test_blkdiag_diagonal_matches_originals(self):
        G1, G2 = _siso_first_order(), _siso_second_order()
        Gd = ss_blkdiag_list([G1, G2])
        fr = _freqresp(Gd)
        assert np.allclose(fr[0, 0, :], _freqresp(G1)[0, 0, :], atol=1e-10)
        assert np.allclose(fr[1, 1, :], _freqresp(G2)[0, 0, :], atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Stability
# ---------------------------------------------------------------------------


class TestIsstable:
    def test_stable_system(self):
        assert isstable(control.ss([[-1.0]], [[1.0]], [[1.0]], [[0.0]]))

    def test_unstable_system(self):
        assert not isstable(control.ss([[1.0]], [[1.0]], [[1.0]], [[0.0]]))

    def test_marginally_stable_is_not_stable(self):
        # pole at 0 — not strictly stable
        assert not isstable(control.ss([[0.0]], [[1.0]], [[1.0]], [[0.0]]))

    def test_second_order_stable(self):
        assert isstable(_siso_second_order())


class TestIsstablecl:
    def test_lqg_feedback_is_stable(self):
        G = _siso_second_order()
        K0, _, _ = lqg_regulator(G, 1.0, 1.0, 1.0, 1.0)
        assert isstablecl(G, K0, sign=+1)

    def test_destabilizing_feedback(self):
        # simple unstable plant; gains that make things worse
        G = control.ss([[0.5]], [[1.0]], [[1.0]], [[0.0]])
        K = control.ss([[-0.1]], [[1.0]], [[-10.0]], [[0.0]])
        assert not isstablecl(G, K, sign=+1)


# ---------------------------------------------------------------------------
# 3. Coprime factorizations
# ---------------------------------------------------------------------------


class TestRncf:
    @pytest.mark.parametrize(
        "G", [_siso_first_order(), _siso_second_order(), _simo_2x1()]
    )
    def test_factorization_identity(self, G):
        """G(jw) = Nr(jw) * inv(Mr(jw)) at all test frequencies."""
        _, Mr, Nr = rncf(G)
        fr_G = _freqresp(G)
        fr_Mr = _freqresp(Mr)
        fr_Nr = _freqresp(Nr)
        for i in range(FREQS.size):
            reconstructed = fr_Nr[:, :, i] @ np.linalg.inv(fr_Mr[:, :, i])
            assert np.allclose(reconstructed, fr_G[:, :, i], atol=1e-6)

    @pytest.mark.parametrize(
        "G", [_siso_first_order(), _siso_second_order(), _simo_2x1()]
    )
    def test_normalized_column(self, G):
        """[Mr; Nr]^H [Mr; Nr] = I (all-pass column property)."""
        _, Mr, Nr = rncf(G)
        fr_Mr = _freqresp(Mr)
        fr_Nr = _freqresp(Nr)
        m = Mr.ninputs
        for i in range(FREQS.size):
            col = np.vstack([fr_Mr[:, :, i], fr_Nr[:, :, i]])
            assert np.allclose(col.conj().T @ col, np.eye(m), atol=1e-6)


class TestLncf:
    @pytest.mark.parametrize("G", [_siso_first_order(), _siso_second_order()])
    def test_factorization_identity(self, G):
        """G(jw) = inv(Ml(jw)) * Nl(jw) at all test frequencies."""
        _, Ml, Nl = lncf(G)
        fr_G = _freqresp(G)
        fr_Ml = _freqresp(Ml)
        fr_Nl = _freqresp(Nl)
        for i in range(FREQS.size):
            reconstructed = np.linalg.inv(fr_Ml[:, :, i]) @ fr_Nl[:, :, i]
            assert np.allclose(reconstructed, fr_G[:, :, i], atol=1e-6)

    @pytest.mark.parametrize("G", [_siso_first_order(), _siso_second_order()])
    def test_normalized_row(self, G):
        """[Ml, Nl] [Ml, Nl]^H = I (all-pass row property)."""
        _, Ml, Nl = lncf(G)
        fr_Ml = _freqresp(Ml)
        fr_Nl = _freqresp(Nl)
        p = Ml.noutputs
        for i in range(FREQS.size):
            row = np.hstack([fr_Ml[:, :, i], fr_Nl[:, :, i]])
            assert np.allclose(row @ row.conj().T, np.eye(p), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Youla parametrization
# ---------------------------------------------------------------------------


class TestYoula:
    def test_Q_zero_gives_K0(self):
        G, K0 = _plant_and_K0()
        K = youla(G, K0, Q=control.ss([], [], [], 0))
        assert np.allclose(_freqresp(K), _freqresp(K0), atol=1e-8)

    def test_Q0b_roundtrip(self):
        """youla(G, K0, youla_Q0b(K1, K0, G)) recovers K1."""
        G, K0 = _plant_and_K0()
        K1, _, _ = lqg_regulator(G, 2.0, 0.5, 1.0, 1.0)
        Q = youla_Q0b(K1, K0, G)
        K_rec = youla(G, K0, Q)
        assert np.allclose(_freqresp(K_rec), _freqresp(K1), atol=1e-6)

    def test_Qab_roundtrip(self):
        """youla(G, Ka, youla_Qab(Ka, Kb, Gstab)) recovers Kb."""
        G, Ka = _plant_and_K0(Qx=1.0, Ru=1.0)
        Kb, _, _ = lqg_regulator(G, 5.0, 0.1, 1.0, 1.0)
        Gstab = control.feedback(G, Ka, sign=+1)
        Qab = youla_Qab(Ka, Kb, Gstab)
        K_rec = youla(G, Ka, Qab)
        assert np.allclose(_freqresp(K_rec), _freqresp(Kb), atol=1e-6)

    def test_youla_laguerre_theta_zero_gives_K0(self):
        G, K0 = _plant_and_K0()
        K = youla_laguerre(G, K0, p=1.0, theta=np.zeros(3))
        assert np.allclose(_freqresp(K), _freqresp(K0), atol=1e-8)

    def test_youla_lqg_Q_zero_gives_Klqg(self):
        """youla_lqg with Q=0 should recover the base LQG controller."""
        G = _siso_second_order()
        Qx, Ru, Qw, Rv = 1.0, 1.0, 1.0, 1.0
        Klqg, _, _ = lqg_regulator(G, Qx, Ru, Qw, Rv)
        K = youla_lqg(G, Qx, Ru, Qw, Rv, Q=control.ss([], [], [], 0))
        assert np.allclose(_freqresp(K), _freqresp(Klqg), atol=1e-8)

    def test_youla_laguerre_K00_dcgain_zero(self):
        G, K0 = _plant_and_K0()
        theta = np.array([0.1, -0.05, 0.02])
        K = youla_laguerre_K00(G, K0, p=1.0, theta=theta)
        assert float(control.dcgain(K)) == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize("theta", [np.zeros(3), np.array([0.2, -0.1, 0.05])])
    def test_youla_laguerre_K00_dcgain_zero_parametrized(self, theta):
        G, K0 = _plant_and_K0()
        K = youla_laguerre_K00(G, K0, p=1.0, theta=theta)
        assert float(control.dcgain(K)) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# 5. Laguerre basis
# ---------------------------------------------------------------------------


class TestLaguerreBasis:
    @pytest.mark.parametrize("p", [0.5, 1.0, 3.0])
    def test_canonical_tf_poles(self, p):
        """All poles of each canonical basis TF are at -p (loose tol: tf2ss introduces O(1e-4) noise)."""
        N = 4
        Phi = basis_laguerre_canonical(p, N)
        for phi in Phi:
            poles = control.poles(control.tf2ss(phi))
            assert np.allclose(np.real(poles), -p, atol=1e-3)

    @pytest.mark.parametrize("i,p", [(0, 1.0), (1, 1.0), (2, 2.0), (3, 0.5)])
    def test_canonical_dc_gain(self, i, p):
        """DC gain of phi_i = sqrt(2p)/p * (-1)^i."""
        Phi = basis_laguerre_canonical(p, i + 1)
        expected = np.sqrt(2 * p) / p * (-1) ** i
        assert float(control.dcgain(control.tf2ss(Phi[i]))) == pytest.approx(
            expected, rel=1e-8
        )

    @pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
    def test_tf_vs_ss_equivalence(self, p):
        """basis_laguerre (TF) and basis_laguerre_ss produce identical freq response."""
        theta = np.array([1.0, -0.5, 0.3])
        Q_tf = basis_laguerre(p, theta)
        Q_ss = basis_laguerre_ss(p, theta)
        fr_tf = _freqresp(control.tf2ss(Q_tf))
        fr_ss = _freqresp(Q_ss)
        assert np.allclose(fr_tf, fr_ss, atol=1e-8)

    @pytest.mark.parametrize("p,N", [(1.0, 3), (2.0, 5)])
    def test_canonical_ss_matches_canonical_tf(self, p, N):
        """basis_laguerre_ss with e_i theta extracts the i-th canonical element."""
        Phi_tf = basis_laguerre_canonical(p, N)
        for i in range(N):
            e_i = np.zeros(N)
            e_i[i] = 1.0
            fr_tf = _freqresp(control.tf2ss(basis_laguerre(p, e_i)))
            fr_ss = _freqresp(basis_laguerre_ss(p, e_i))
            assert np.allclose(fr_tf, fr_ss, atol=1e-8), f"Mismatch at element {i}"


# ---------------------------------------------------------------------------
# 6. Balanced reduction
# ---------------------------------------------------------------------------


class TestBalreal:
    @pytest.mark.parametrize("G", [_siso_first_order(), _siso_second_order()])
    def test_gramians_balanced_and_diagonal(self, G):
        """After balancing, Wc == Wo and both are diagonal."""
        Gbal = balreal(G)
        Wc = np.asarray(control.gram(Gbal, "c"))
        Wo = np.asarray(control.gram(Gbal, "o"))
        assert np.allclose(Wc, Wo, atol=1e-8)
        assert np.allclose(Wc, np.diag(np.diag(Wc)), atol=1e-8)

    @pytest.mark.parametrize("G", [_siso_first_order(), _siso_second_order()])
    def test_balreal_preserves_freqresp(self, G):
        """Balancing is a similarity transform; frequency response is unchanged."""
        Gbal = balreal(G)
        assert np.allclose(_freqresp(G), _freqresp(Gbal), atol=1e-8)


class TestSysHsv:
    @pytest.mark.parametrize("G", [_siso_first_order(), _siso_second_order()])
    def test_hsv_positive_and_sorted(self, G):
        hsv = sys_hsv(G)
        assert np.all(hsv > 0)
        assert np.all(np.diff(hsv) <= 0)

    def test_hsv_matches_gramian_computation(self):
        G = _siso_second_order()
        Wc = np.asarray(control.gram(G, "c"))
        Wo = np.asarray(control.gram(G, "o"))
        expected = np.sqrt(
            np.sort(np.real(np.linalg.eigvals(Wc @ Wo)))[::-1]
        )
        hsv = sys_hsv(G)
        assert np.allclose(hsv, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Controller residues
# ---------------------------------------------------------------------------


class TestControllerResidues:
    def test_single_real_pole(self):
        """K(s) = c/(s - p) should match residue formula."""
        c, p = 2.0, -3.0
        K = controller_residues(real_c=[c], real_p=[p])
        fr = _freqresp(K)
        expected = c / (1j * FREQS - p)
        assert np.allclose(fr[0, 0, :], expected, atol=1e-8)

    def test_single_complex_pair(self):
        """Real-valued controller built from a complex-conjugate pair."""
        p = -1.0 + 2j
        c = 0.5 + 0.3j
        K = controller_residues(cplx_c=[c], cplx_p=[p])
        fr = _freqresp(K)
        expected = c / (1j * FREQS - p) + np.conj(c) / (1j * FREQS - np.conj(p))
        assert np.allclose(fr[0, 0, :], expected, atol=1e-8)

    def test_combined_real_and_complex(self):
        """Superposition of real and complex contributions."""
        real_c, real_p = [1.0], [-2.0]
        cplx_c, cplx_p = [0.3 + 0.2j], [-1.0 + 1.5j]
        K = controller_residues(real_c, real_p, cplx_c, cplx_p)
        fr = _freqresp(K)
        c_re, p_re = real_c[0], real_p[0]
        c_cx, p_cx = cplx_c[0], cplx_p[0]
        expected = (
            c_re / (1j * FREQS - p_re)
            + c_cx / (1j * FREQS - p_cx)
            + np.conj(c_cx) / (1j * FREQS - np.conj(p_cx))
        )
        assert np.allclose(fr[0, 0, :], expected, atol=1e-8)

    def test_wrapper_roundtrip(self):
        """controller_residues_wrapper gives same result as direct call."""
        real_c, real_p = [1.0, -2.0], [-1.0, -3.0]
        cplx_c, cplx_p = [0.5 + 0.3j], [-1.0 + 2j]
        K_direct = controller_residues(real_c, real_p, cplx_c, cplx_p)
        theta = np.array([
            *real_c, *real_p,
            np.real(cplx_c[0]), np.imag(cplx_c[0]),
            np.real(cplx_p[0]), np.imag(cplx_p[0]),
        ])
        K_wrapped = controller_residues_wrapper(theta, n_real=2, n_cplx=1)
        assert np.allclose(_freqresp(K_direct), _freqresp(K_wrapped), atol=1e-10)

    def test_wrong_theta_length_raises(self):
        with pytest.raises(ValueError, match="theta length"):
            controller_residues_wrapper(np.zeros(5), n_real=2, n_cplx=1)

    def test_getidx_partitions_all_indices(self):
        """Index slices from getidx cover exactly [0, 2*n_real + 4*n_cplx)."""
        n_real, n_cplx = 3, 2
        idx_groups = controller_residues_getidx(n_real, n_cplx)
        all_idx = np.concatenate(idx_groups)
        assert np.array_equal(np.sort(all_idx), np.arange(2 * n_real + 4 * n_cplx))

    def test_empty_call_returns_zero_system(self):
        K = controller_residues()
        assert float(control.dcgain(K)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Slow-fast decomposition
# ---------------------------------------------------------------------------


class TestSlowfast:
    def test_decomposition_sums_to_G(self):
        G = _siso_second_order()
        Gslow, Gfast = slowfast(G, wlim=1.5)
        fr_G = _freqresp(G)
        fr_sum = _freqresp(Gslow + Gfast)
        assert np.allclose(fr_G, fr_sum, atol=1e-8)

    def test_slow_poles_below_wlim(self):
        G = _siso_second_order()
        wlim = 1.5
        Gslow, _ = slowfast(G, wlim=wlim)
        if Gslow.nstates > 0:
            assert np.all(np.abs(control.poles(Gslow)) < wlim)

    def test_fast_poles_above_wlim(self):
        G = _siso_second_order()
        wlim = 1.5
        _, Gfast = slowfast(G, wlim=wlim)
        if Gfast.nstates > 0:
            assert np.all(np.abs(control.poles(Gfast)) >= wlim)

    def test_mimo_raises(self):
        with pytest.raises(ValueError):
            slowfast(_simo_2x1(), wlim=1.0)


# ---------------------------------------------------------------------------
# 9. I/O round-trip
# ---------------------------------------------------------------------------


class TestWriteReadSs:
    def test_matrices_roundtrip(self):
        G = _siso_second_order()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_sys.mat")
            write_ss(G, path)
            G2 = read_ss(path)
        assert np.allclose(G.A, G2.A, atol=1e-12)
        assert np.allclose(G.B, G2.B, atol=1e-12)
        assert np.allclose(G.C, G2.C, atol=1e-12)
        assert np.allclose(G.D, G2.D, atol=1e-12)
