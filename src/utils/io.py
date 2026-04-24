"""I/O utilities: XDMF read/write, field and matrix export, frequency response save/plot."""

import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any, Optional

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.sparse as spr
from dolfin import dot, inner

from utils.fem import get_subspace_dofs, projectm
from utils.linalg import dolfin_to_scipy, petsc_to_scipy

logger = logging.getLogger(__name__)


def write_xdmf(
    filename: str | Path,
    func: dolfin.Function,
    name: str,
    time_step: float = 0.0,
    append: bool = False,
    write_mesh: bool = True,
) -> None:
    """Write a dolfin.Function to an XDMF checkpoint file."""
    with dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename)) as ff:
        ff.parameters["rewrite_function_mesh"] = write_mesh
        ff.parameters["functions_share_mesh"] = not write_mesh
        ff.write_checkpoint(
            func,
            name,
            time_step=time_step,
            encoding=dolfin.XDMFFile.Encoding.HDF5,
            append=append,
        )


def read_xdmf(
    filename: str | Path,
    func: dolfin.Function,
    name: str,
    counter: int = -1,
) -> None:
    """Read a dolfin.Function from an XDMF checkpoint file."""
    with dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename)) as ff:
        ff.read_checkpoint(func, name=name, counter=counter)


_PART_FUNCS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "re": np.real,
    "im": np.imag,
    "abs": np.abs,
    "arg": np.angle,
}


def export_complex_field(
    cfields: np.ndarray,
    W: dolfin.FunctionSpace,
    V: dolfin.FunctionSpace,
    P: dolfin.FunctionSpace,
    file_prefix: Path,
    time_steps: Optional[list[float]] = None,
    parts: Optional[list[str]] = None,
) -> None:
    """Export complex fields to XDMF files, split into velocity and pressure.

    XDMF cannot store mixed function spaces (velocity + pressure in W), so the
    field is split into V and P components via FunctionAssigner. Each part and
    each component gets its own file; time steps are appended within each file.

    Parameters
    ----------
    cfields:
        Complex array of shape (n, ncols) or (n, nu, ncols). For eigenvectors
        pass a 2D array with one mode per column. For frequency-response fields
        (output of get_field_response) pass the 3D array directly; nu > 1 is
        handled by writing one set of files per input column.
    W, V, P:
        Mixed, velocity, and pressure function spaces. V and P must be the
        sub-spaces of W (as passed to FunctionAssigner).
    file_prefix:
        Base path without extension, e.g. Path("results/cylinder").
    time_steps:
        Values written as the XDMF time axis, one per column. Defaults to
        0, 1, 2, … (suitable for mode indices). Pass ww.tolist() to use
        angular frequencies as the time axis so the Paraview slider browses
        the frequency sweep directly.
    parts:
        Subset of ["re", "im", "abs", "arg"] to write. Defaults to all four.
        For frequency-response fields ["abs", "arg"] is the natural choice
        (abs/arg are invariant to global phase, re/im are not).

    File naming
    -----------
    Each (input, component, part) triplet produces one file:
        <prefix>_v_<part>.xdmf       velocity,  nu == 1
        <prefix>_p_<part>.xdmf       pressure,  nu == 1
        <prefix>_input{iu}_v_<part>.xdmf        nu > 1
        <prefix>_input{iu}_p_<part>.xdmf        nu > 1

    Within each file the time axis follows time_steps, so stepping through
    time in Paraview browses columns (modes or frequencies).
    """
    if parts is None:
        parts = list(_PART_FUNCS)

    invalid = set(parts) - _PART_FUNCS.keys()
    if invalid:
        raise ValueError(f"Unknown parts: {invalid}. Choose from {list(_PART_FUNCS)}.")

    if cfields.ndim == 2:
        cfields = cfields[:, np.newaxis, :]
    n, nu, ncols = cfields.shape

    if time_steps is None:
        time_steps = list(range(ncols))
    if len(time_steps) != ncols:
        raise ValueError(f"time_steps length {len(time_steps)} != ncols {ncols}.")

    w_func = dolfin.Function(W)
    vv = dolfin.Function(V)
    pp = dolfin.Function(P)
    fa = dolfin.FunctionAssigner([V, P], W)
    lr = w_func.vector().local_range()

    def mkpath(iu: int, comp: str, part: str) -> Path:
        inp = f"_input{iu}" if nu > 1 else ""
        return file_prefix.parent / (file_prefix.name + inp + f"_{comp}_{part}.xdmf")

    for iu in range(nu):
        for part_name in parts:
            part_func = _PART_FUNCS[part_name]
            for i in range(ncols):
                w_func.vector().set_local(part_func(cfields[lr[0]:lr[1], iu, i]))
                w_func.vector().apply("insert")
                fa.assign([vv, pp], w_func)
                write_xdmf(
                    mkpath(iu, "v", part_name), vv, f"v_{part_name}",
                    time_step=float(time_steps[i]),
                    append=(i > 0), write_mesh=(i == 0),
                )
                write_xdmf(
                    mkpath(iu, "p", part_name), pp, f"p_{part_name}",
                    time_step=float(time_steps[i]),
                    append=(i > 0), write_mesh=(i == 0),
                )
        logger.info("Exported %d fields (input %d) to %s", ncols, iu, file_prefix)


def export_npz_to_mat(
    infile: str | Path,
    outfile: str | Path,
    matname: str,
) -> None:
    """Load a scipy sparse matrix from infile.npz and save it to outfile.mat."""
    Msp = spr.load_npz(infile)
    sio.savemat(outfile, mdict={matname: Msp.tocsc()})


def export_subdomains(
    mesh: dolfin.Mesh,
    subdomains_list: list[dolfin.SubDomain],
    filename: str | Path = "subdomains.xdmf",
) -> None:
    """Export mesh subdomains to XDMF for visualization.
    Usage: export_subdomains(fs.mesh, fs.boundaries.subdomain, ...)"""
    subd = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.set_all(0)
    for i, subdomain in enumerate(subdomains_list):
        subdomain.mark(subd, i + 1)
        logger.info("Marking subdomain nr: %d", i + 1)
    logger.info("Writing subdomains file: %s", filename)
    with dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename)) as fsubd:
        fsubd.write(subd)


def export_boundary_field(
    mesh: dolfin.Mesh,
    ds: dolfin.Measure,
    n: Any = None,
    filename: str | Path = "boundary_field.xdmf",
    name: str = "boundary_field",
) -> None:
    """Project a vector field defined on ds onto CG1 and write to XDMF.
    If n is None, defaults to the mesh facet normals."""
    if n is None:
        n = dolfin.FacetNormal(mesh)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    A = dolfin.assemble(inner(u, v) * ds, keep_diagonal=True)
    L = dolfin.assemble(inner(n, v) * ds)
    A.ident_zeros()
    nh = dolfin.Function(V, name=name)
    dolfin.solve(A, nh.vector(), L)
    write_xdmf(filename, nh, name)


def export_stress_tensor(
    sigma: Any,
    mesh: dolfin.Mesh,
    filename: str | Path = "stress_tensor.xdmf",
    name: str = "stress_tensor",
) -> None:
    """Project and write a stress tensor field to XDMF."""
    TT = dolfin.TensorFunctionSpace(mesh, "DG", degree=0)
    sigma_ = dolfin.Function(TT, name=name)
    sigma_.assign(projectm(sigma, TT))
    write_xdmf(filename, sigma_, name)


def export_boundary_forces(
    sigma: Any,
    mesh: dolfin.Mesh,
    ds: dolfin.Measure,
    filename: str | Path = "forces.xdmf",
    name: str = "forces",
) -> None:
    """Compute and write boundary forces -dot(sigma, n) to XDMF."""
    n = dolfin.FacetNormal(mesh)
    export_boundary_field(
        mesh=mesh, n=-dot(sigma, n), ds=ds, filename=filename, name=name
    )


def export_square_operators(
    path: str | Path,
    operators: list[dolfin.PETScMatrix],
    operators_names: list[str],
) -> None:
    """Export square operators (dolfin PETScMatrix) as spy PNG and sparse NPZ files."""
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)
    for Mat, Matname in zip(operators, operators_names):
        export_sparse_matrix(Mat, save_path / f"{Matname}.png")
        indptr, indices, data = Mat.mat().getValuesCSR()
        Acsr = spr.csr_matrix((data, indices, indptr))
        spr.save_npz(save_path / f"{Matname}.npz", Acsr)
        spr.save_npz(save_path / f"{Matname}_coo.npz", Acsr.tocoo())


def export_sparse_matrix(
    A: dolfin.cpp.la.PETScMatrix | spr.spmatrix | np.ndarray,
    figname: str | Path | None = None,
) -> None:
    """Export sparse matrix to a spy plot PNG.
    A: dolfin PETScMatrix or scipy.sparse matrix."""

    if spr.issparse(A):
        Acsr = A
    elif isinstance(A, np.ndarray):
        Acsr = spr.csr_matrix(A)
    else:
        Acsr = dolfin_to_scipy(A)

    fig, ax = plt.subplots()
    ax.spy(Acsr, markersize=1)
    ax.set_title("Sparse matrix plot")
    fig.savefig(figname if figname is not None else "spy.png")
    plt.close(fig)


def export_dof_map(
    W: dolfin.FunctionSpace,
    plotsz: Optional[int] = None,
    figname: str | Path = "dofmap.png",
) -> None:
    """Save an image showing DOF distribution by subspace (u, v, p) for W."""
    dofmap = get_subspace_dofs(W)
    if plotsz is None:
        plotsz = W.dim()

    dofim = np.zeros((plotsz, plotsz), dtype=int)
    for i, subs in enumerate(["u", "v", "p"]):
        dofmap_idx = dofmap[subs]
        subs_low = dofmap_idx[dofmap_idx < plotsz]
        dofim[:, subs_low] = i

    fig, ax = plt.subplots()
    im = ax.imshow(dofim, cmap=plt.colormaps["binary"])
    ax.set_title("Distribution of DOFs by index: (u,v,p)")
    fig.colorbar(im)
    fig.savefig(figname)
    plt.close(fig)


def save_Hw(
    H: np.ndarray,
    ww: np.ndarray,
    save_dir: Path,
    save_suffix: str = "",
    input_labels: Optional[list[str]] = None,
    output_labels: Optional[list[str]] = None,
) -> None:
    """Save frequency response data to .mat files.
    Saves one combined file (full H) and one file per (output, input) pair.

    Args:
        H:             Frequency response, shape (ny, nu, nw), complex.
        ww:            Frequency array, shape (nw,).
        save_dir:      Directory to save files into.
        save_suffix:   Optional suffix appended to filenames.
        input_labels:  Names for each input channel, length nu.
        output_labels: Names for each output channel, length ny.
    """
    ny, nu, nw = H.shape

    if input_labels is None:
        input_labels = [f"u{i}" for i in range(nu)]
    if output_labels is None:
        output_labels = [f"y{i}" for i in range(ny)]

    combined_path = save_dir / f"Hw_nw{nw}_ny{ny}_nu{nu}{save_suffix}.mat"
    sio.savemat(
        combined_path,
        {
            "H": H,
            "w": ww,
            "input_labels": input_labels,
            "output_labels": output_labels,
            "comment": "H has shape (ny, nu, nw)",
        },
    )
    logger.info("Saving combined frequency response to: %s", combined_path)

    for iy in range(ny):
        for iu in range(nu):
            suffix_i = f"_{output_labels[iy]}_to_{input_labels[iu]}"
            pair_path = save_dir / f"Hw_nw{nw}{save_suffix}{suffix_i}.mat"
            sio.savemat(
                pair_path,
                {
                    "H": H[iy, iu, :],
                    "w": ww,
                    "input_label": input_labels[iu],
                    "output_label": output_labels[iy],
                },
            )
            logger.info(
                "Saving H[%s -> %s] to: %s",
                input_labels[iu],
                output_labels[iy],
                pair_path,
            )


def plot_Hw(
    H: np.ndarray,
    ww: np.ndarray,
    save_dir: Path,
    save_suffix: str = "",
    input_labels: Optional[list[str]] = None,
    output_labels: Optional[list[str]] = None,
) -> None:
    """Plot and save Bode diagrams for each (output, input) pair.
    Produces one figure per input channel, with ny subplots (one per output).

    Args:
        H:             Frequency response, shape (ny, nu, nw), complex.
        ww:            Frequency array, shape (nw,).
        save_dir:      Directory to save figures into.
        save_suffix:   Optional suffix appended to filenames.
        input_labels:  Names for each input channel, length nu.
        output_labels: Names for each output channel, length ny.
    """
    ny, nu, _ = H.shape

    if input_labels is None:
        input_labels = [f"u{i}" for i in range(nu)]
    if output_labels is None:
        output_labels = [f"y{i}" for i in range(ny)]

    for iu in range(nu):
        fig, axs = plt.subplots(ny, 2, figsize=(10, 3 * ny), squeeze=False)
        fig.suptitle(f"Bode plot — input: {input_labels[iu]}")

        for iy in range(ny):
            H_iy_iu = H[iy, iu, :]
            ax_mag, ax_phase = axs[iy, 0], axs[iy, 1]

            ax_mag.scatter(ww, 20 * np.log10(np.abs(H_iy_iu)), marker=".")
            ax_mag.set_ylabel(f"|H| (dB)\n{output_labels[iy]}")
            ax_mag.set_xscale("log")
            ax_mag.grid(which="both")

            ax_phase.scatter(
                ww, (180 / np.pi) * np.unwrap(np.angle(H_iy_iu)), marker="."
            )
            ax_phase.set_ylabel(f"Phase (deg)\n{output_labels[iy]}")
            ax_phase.set_xscale("log")
            ax_phase.grid(which="both")

        axs[-1, 0].set_xlabel("Frequency")
        axs[-1, 1].set_xlabel("Frequency")

        fig.tight_layout()
        fig_path = save_dir / f"bodeplot_{input_labels[iu]}{save_suffix}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info("Saving Bode plot for input %s to: %s", input_labels[iu], fig_path)
