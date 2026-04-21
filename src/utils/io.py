"""I/O utilities: XDMF read/write, field and matrix export, frequency response save/plot."""

import functools
import logging
from pathlib import Path
from typing import Optional

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.sparse as spr
from dolfin import dot, inner
from matplotlib import cm

logger = logging.getLogger(__name__)

# Shortcut to define projection with MUMPS
_projectm = functools.partial(dolfin.project, solver_type="mumps")


def write_xdmf(filename, func, name, time_step=0.0, append=False, write_mesh=True):
    """Shortcut to write XDMF file with options & context manager"""
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


def read_xdmf(filename, func, name, counter=-1):
    """Shortcut to read XDMF file with context manager"""
    with dolfin.XDMFFile(dolfin.MPI.comm_world, str(filename)) as ff:
        ff.read_checkpoint(func, name=name, counter=counter)


def export_field(cfields, W, V, P, save_dir=None, time_steps=None):
    """Export complex field to files, for visualizing in function space.
    May be used to export any matrix defined on the function spaces W=(V,P),
    such as the actuation and sensing matrices B, C.
    cfields: array with cfields as columns"""
    if save_dir is None:
        save_dir = "/stck/wjussiau/fenics-python/ns/data/export/vec_"
    vec_v_file = Path(str(save_dir) + "_v")
    vec_p_file = Path(str(save_dir) + "_p")
    xdmf = ".xdmf"

    def mkfilename(filename, part):
        return Path(str(filename) + "_" + part + xdmf)

    ww = dolfin.Function(W)
    vv = dolfin.Function(V)
    pp = dolfin.Function(P)
    fa = dolfin.FunctionAssigner([V, P], W)

    if time_steps is None:
        time_steps = list(range(cfields.shape[1]))

    is_append = False
    for i in range(cfields.shape[1]):
        cfield = cfields[:, i]

        cfield_re = np.real(cfield)
        cfield_im = np.imag(cfield)
        cfield_abs = np.abs(cfield)
        cfield_arg = np.angle(cfield)

        for cfield_part, cfield_part_name in zip(
            [cfield_re, cfield_im, cfield_abs, cfield_arg], ["re", "im", "abs", "arg"]
        ):
            ww.vector().set_local(cfield_part)
            fa.assign([vv, pp], ww)
            write_xdmf(
                mkfilename(vec_v_file, cfield_part_name),
                vv,
                "v_eig_" + cfield_part_name,
                time_step=time_steps[i],
                append=is_append,
            )
            write_xdmf(
                mkfilename(vec_p_file, cfield_part_name),
                pp,
                "p_eig_" + cfield_part_name,
                time_step=time_steps[i],
                append=is_append,
            )

        is_append = True
        logger.info("Writing eigenvector: %d" % (i + 1))


def export_to_mat(infile, outfile, matname, option="sparse"):
    """Load sparse matrix from infile.npz and export it to outfile.mat"""
    if option == "sparse":
        Msp = spr.load_npz(infile)
        sio.savemat(outfile, mdict={matname: Msp.tocsc()})
    else:
        sio.savemat(outfile, mdict=np.load(infile))


def export_subdomains(mesh, subdomains_list, filename="subdomains.xdmf"):
    """Export subdomains of FlowSolver object to be displayed.
    Usage: export_subdomains(fs.mesh, fs.boundaries.subdomain, ...)"""
    subd = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subd.set_all(0)
    for i, subdomain in enumerate(subdomains_list):
        subdnr = 10 * (i + 1)
        subdomain.mark(subd, subdnr)
        logger.info("Marking subdomain nr: {0} ({1})".format(i + 1, subdnr))
    logger.info("Writing subdomains file: %s", filename)
    with dolfin.XDMFFile(str(filename)) as fsubd:
        fsubd.write(subd)


def export_facetnormals(
    mesh, ds, n=None, filename="facet_normals.xdmf", name="facet_normals"
):
    """Write mesh facet normals to file.
    Can be used to export forces (n << dot(sigma, n))"""
    if n is None:
        n = dolfin.FacetNormal(mesh)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    a_lhs = inner(u, v) * ds
    l_rhs = inner(n, v) * ds
    A = dolfin.assemble(a_lhs, keep_diagonal=True)
    L = dolfin.assemble(l_rhs)
    A.ident_zeros()
    nh = dolfin.Function(V, name=name)
    dolfin.solve(A, nh.vector(), L)
    write_xdmf(filename, nh, name)


def export_stress_tensor(
    sigma,
    mesh,
    filename="stress_tensor.xdmf",
    export_forces=False,
    ds=None,
    name="stress_tensor",
):
    """Write stress tensor to file"""
    TT = dolfin.TensorFunctionSpace(mesh, "DG", degree=0)
    sigma_ = dolfin.Function(TT, name=name)
    sigma_.assign(_projectm(sigma, TT))
    write_xdmf(filename, sigma_, name)

    if export_forces:
        n = dolfin.FacetNormal(mesh)
        export_facetnormals(
            mesh=mesh, n=-dot(sigma, n), ds=ds, filename="forces.xdmf", name="forces"
        )


def export_sparse_matrix(A, figname=None):
    """Export sparse matrix to spy plot.
    A: dolfin PETScMatrix or scipy.sparse.csr_matrix"""
    from utils.linalg import dense_to_sparse

    if spr.issparse(A):
        Acsr = A
    else:
        Acsr = dense_to_sparse(A)

    fig, ax = plt.subplots()
    ax.spy(Acsr, markersize=1)
    ax.set_title("Sparse matrix plot")
    if figname is None:
        figname = "spy.png"
    fig.savefig(figname)


def export_dof_map(W, plotsz=None):
    """Create an image of size W.dim*W.dim with each column
    being the colour of the underlying function space of the corresponding dof.
    E.G. (column i==red) if (dof i==u)"""
    from utils.fem import get_subspace_dofs

    dofmap = get_subspace_dofs(W)
    sz = W.dim()

    if plotsz is None:
        plotsz = sz

    dofim = np.zeros((plotsz, plotsz), dtype=int)

    for i, subs in enumerate(["u", "v", "p"]):
        dofmap_idx = dofmap[subs]
        subs_low = dofmap_idx[dofmap_idx < plotsz]
        dofim[:, subs_low] = i

    fig, ax = plt.subplots()
    im = ax.imshow(dofim, cmap=cm.get_cmap("binary"))
    ax.set_title("Distribution of DOFs by index: (u,v,p)")
    fig.colorbar(im)
    fig.savefig("dofmap.png")


def save_Hw(
    H: np.ndarray,
    ww: np.ndarray,
    save_dir: str,
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
    save_path = Path(save_dir)

    if input_labels is None:
        input_labels = [f"u{i}" for i in range(nu)]
    if output_labels is None:
        output_labels = [f"y{i}" for i in range(ny)]

    combined_path = save_path / f"Hw_nw{nw}_ny{ny}_nu{nu}{save_suffix}.mat"
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
            pair_path = save_path / f"Hw_nw{nw}{save_suffix}{suffix_i}.mat"
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
    save_dir: str,
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
    ny, nu, nw = H.shape
    save_path = Path(save_dir)

    if input_labels is None:
        input_labels = [f"u{i}" for i in range(nu)]
    if output_labels is None:
        output_labels = [f"y{i}" for i in range(ny)]

    for iu in range(nu):
        fig, axs = plt.subplots(ny, 2, figsize=(10, 3 * ny), squeeze=False)
        fig.suptitle(f"Bode plot — input: {input_labels[iu]}")

        for iy in range(ny):
            H_iy_iu = H[iy, iu, :]

            ax_mag = axs[iy, 0]
            ax_phase = axs[iy, 1]

            ax_mag.scatter(ww, 20 * np.log10(np.abs(H_iy_iu)), marker=".")
            ax_mag.set_ylabel(f"|H| (dB)\n{output_labels[iy]}")
            ax_mag.set_xscale("log")
            ax_mag.grid(which="both")

            ax_phase.scatter(
                ww,
                (180 / np.pi) * np.unwrap(np.angle(H_iy_iu)),
                marker=".",
            )
            ax_phase.set_ylabel(f"Phase (deg)\n{output_labels[iy]}")
            ax_phase.set_xscale("log")
            ax_phase.grid(which="both")

        axs[-1, 0].set_xlabel("Frequency")
        axs[-1, 1].set_xlabel("Frequency")

        fig.tight_layout()
        fig_path = save_path / f"bodeplot_{input_labels[iu]}{save_suffix}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info("Saving Bode plot for input %s to: %s", input_labels[iu], fig_path)
