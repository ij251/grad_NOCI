from pyscf import gto, scf, grad
import numpy as np
# from spatial_CPHF_functions import *


def rhf_to_ghf(g0_rhf, nelec):

    r"""Calculates the GHF MO coefficient matrix from the RHF, organised in
    blocks of occupied then virtual orbitals.

    :param g0_rhf: RHF MO coefficient matrix.
    :param nelec: Number of electrons in the molecule, determines the number
            of occupied orbitals.

    :returns: GHF MO coefficient matrix.
    """

    nbasis = g0_rhf.shape[1]

    # hcore0 =get_hcore0_spatial(mol)
    # pi0 = get_pi0_spatial(mol)
    # f0 = get_f0_spatial(hcore0, pi0, mol)
    # eta0,_ = np.linalg.eig(f0)
    # index = np.argsort(eta0)
    # g0_rhf = g0_rhf[index]

    if nelec % 2 == 0:
        nocc = int(nelec/2)
    else:
        nocc = int((nelec+1)/2)

    g0_ghf = np.block([[g0_rhf[:, 0:nocc],
                        np.zeros_like(g0_rhf[:, 0:nocc]),
                        g0_rhf[:, nocc:nbasis],
                        np.zeros_like(g0_rhf[:, nocc:nbasis])],
                       [np.zeros_like(g0_rhf[:, 0:nocc]),
                        g0_rhf[:, 0:nocc],
                        np.zeros_like(g0_rhf[:, nocc:nbasis]),
                        g0_rhf[:, nocc:nbasis]]])

    return g0_ghf


def uhf_to_ghf(g0_uhf_alpha, g0_uhf_beta, nalpha, nbeta):

    r"""Calculates the GHF MO coefficient matrix from the UHF alpha and beta
    coefficient matrices, organised in blocks of occupied then virtual
    orbitals.

    :param g0_uhf_alpha: UHF MO coefficient matrix for alpha orbitals.
    :param g0_uhf_beta: UHF MO coefficient matrix for beta orbitals.
    :param nalpha: Number of alpha electrons.
    :param nbeta: Number of beta electrons.

    :returns: GHF MO coefficient matrix.
    """

    assert np.allclose(g0_uhf_alpha.shape[0], g0_uhf_beta.shape[0])

    nbasis = g0_uhf_alpha.shape[1]

    g0_ghf = np.block([[g0_uhf_alpha[:, 0:nalpha],
                        np.zeros_like(g0_uhf_beta[:, 0:nbeta]),
                        g0_uhf_alpha[:, nalpha:nbasis],
                        np.zeros_like(g0_uhf_beta[:, nbeta:nbasis])],
                       [np.zeros_like(g0_uhf_alpha[:, 0:nalpha]),
                        g0_uhf_beta[:, 0:nbeta],
                        np.zeros_like(g0_uhf_alpha[:, nalpha:nbasis]),
                        g0_uhf_beta[:, nbeta:nbasis]]])

    return g0_ghf


def get_x_lowdin(mol, thresh: float = 1e-14):

    r"""Calculates canonical basis orthogonalisation matrix x, defined by:

    .. math::

        \mathbf{X}=\mathbf{Us^{-\frac{1}{2}}}}

    where U is the matrix of eigenvectors of s_ao, and
    :math:'s^{-\frac{1}{2}}}' is the diagonal matrix of inverse square root
    eigenvalues of s_ao.

    :param s_ao: atomic orbital overlap matrix
    :param thresh: Threshold to consider an eigenvalue of the AO overlap
            as zero.

    :returns: the orthogonalisation matrix x
    """

    omega = np.identity(2)
    spatial_overlap_s = mol.intor('int1e_ovlp')
    overlap_s = np.kron(omega, spatial_overlap_s)

    assert np.allclose(overlap_s, overlap_s.T.conj(), rtol=0, atol=thresh)
    s_eig, mat_u = np.linalg.eigh(overlap_s)
    overlap_indices = np.where(np.abs(s_eig) > thresh)[0]
    s_eig = s_eig[overlap_indices]
    mat_u = mat_u[:, overlap_indices]
    s_s = np.diag(1.0/s_eig)**0.5
    mat_x = np.dot(mat_u, s_s)

    return mat_x


def get_p0(g0, complexsymmetric: bool, nelec):

    r"""Calculates the zeroth order density matrix from the zeroth order
    coefficient matrix. It is defined by (only over occupied MOs):

    .. math::

        \mathbf{P^{(0)}}=\mathbf{G^{(0)}G^{(0)\dagger\diamond}}

    :param g0: zeroth order GHF coefficient matrix.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The zeroth order density matrix.
    """

    if not complexsymmetric:
        p0 = np.dot(g0[:,0:nelec], g0[:,0:nelec].T.conj())
    else:
        p0 = np.dot(g0[:,0:nelec], g0[:,0:nelec].T)

    return p0


def get_hcore0(mol):

    r"""Calculates The zeroth order core hamiltonian.
    Each element is given by:

    .. math::

        \left(\mathbf{H_{core}^{(0)}}\right)_{\mu\nu}
        =\left(\phi_{\mu}\left|\mathbf{\hat{H}_{core}}\right|\phi_{\nu}\right)

    :param mol: The class for a molecule as defined by PySCF.

    :returns: The zeroth order core hamiltonian matrix.
    """

    hcore0 = (mol.intor('int1e_nuc')
              + mol.intor('int1e_kin'))

    omega = np.identity(2)
    hcore0 =  np.kron(omega, hcore0)

    return hcore0


def get_pi0(mol):

    r"""Calculate the 4 dimensional zeroth order Pi tensor.
    Each element is given by:

    .. math::

        \mathbf{\Pi_{\delta'\mu',\epsilon'\nu',\delta\mu,\epsilon\nu}^{(0)}}
        = \mathbf{\Omega_{\delta'\delta}\Omega_{\epsilon'\epsilon}}
          \left(\mu'\mu|\nu'\nu\right)
        - \mathbf{\Omega_{\delta'\epsilon}\Omega_{\epsilon'\delta}}
          \left(\mu'\nu|\nu'\mu\right)

    :param mol: The class for a molecule as defined by PySCF.

    :returns: The zeroth order Pi tensor.
    """

    spatial_j = mol.intor('int2e')
    phys_spatial_j = np.einsum("abcd->acbd", spatial_j)
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    j = np.kron(spin_j, phys_spatial_j)
    k = np.einsum("ijkl->ijlk", j)
    pi0 = j - k

    return pi0


def get_j0(mol):

    r"""Calculate the 4 dimensional zeroth order j tensor.
    Each element is given by:

    .. math::

        \mathbf{\Pi_{\delta'\mu',\epsilon'\nu',\delta\mu,\epsilon\nu}^{(0)}}
        = \mathbf{\Omega_{\delta'\delta}\Omega_{\epsilon'\epsilon}}
          \left(\mu'\mu|\nu'\nu\right)

    :param mol: The class for a molecule as defined by PySCF.

    :returns: The zeroth order j tensor.
    """

    spatial_j = mol.intor('int2e')
    phys_spatial_j = np.einsum("abcd->acbd", spatial_j)
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    j = np.kron(spin_j, phys_spatial_j)
    j = np.einsum("acbd->abcd", j)

    return j


def get_f0(hcore0, pi0, p0):

    r"""Calculates the zeroth order fock matrix, defined by:

    .. math::

        \mathbf{F^{(0)}}
        =\mathbf{H_{core}^{(0)}}
        +\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(0)}}

    :param hcore0: Zeroth order core hamiltonian matrix.
    :param pi0: Zeroth order 4 dimensional Pi tensor.
    :param p0: Zeroth order density matrix.

    :returns: The zeroth order fock matrix
    """

    f0_1e = hcore0
    f0_2e = np.einsum("ijkl,jl->ik", pi0, p0)
    f0 = f0_1e + f0_2e

    return f0


def get_e0_nuc(mol):

    r"""Calculates the zeroth order nuclear repulsion energy.
    This is given by the following expression, where N is the total number of
    nuclei in the system, A and B are nuclear indices, Z is the atomic number,
    and R_{AB} is the distance between nucei A and B:

    .. math::

        E^{(0)}_{nuc} = \sum\limits_{A>B}^N\frac{Z_AZ_B}{R_{AB}}

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.

    :returns: The zeroth order nuclear repulsion energy.
    """

    e0_nuc = 0

    for a in range(len(mol.atom_charges())):
        for b in range(a+1, len(mol.atom_charges())):

            r_ab2 = np.dot(mol.atom_coord(a) - mol.atom_coord(b),
                           mol.atom_coord(a) - mol.atom_coord(b))
            r_ab = np.sqrt(r_ab2)

            e0_nuc += (mol.atom_charge(a) * mol.atom_charge(b)) / r_ab

    return e0_nuc


def get_e0_elec(mol, g0, complexsymmetric, nelec):

    r"""Calculates the zeroth order electronic energy.
    This is a contraction of the zeroth order density matrix with zeroth order
    core hamiltonian and pi tensors:

    .. math::

        E^{(0)}_{elec} = Tr\left(\mathbf{F'}^{(0)}\mathbf{P}^{(0)}\right)

    where

    .. math::

        \mathbf{F'}^{(0)} = \mathbf{H_{core}^{(0)}}
        + \frac{1}{2}\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(0)}}

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param g0: The zeroth order coefficient matrix.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.

    :returns: The zeroth order Hartree Fock electronic energy
    """

    p0 = get_p0(g0, complexsymmetric, nelec)
    hcore0 = get_hcore0(mol)
    pi0 = get_pi0(mol)

    f0_prime_1e = hcore0
    f0_prime_2e = 0.5 * np.einsum("ijkl,jl->ik", pi0, p0)

    e0_elec_1e = np.trace(np.dot(f0_prime_1e, p0))
    e0_elec_2e = np.trace(np.dot(f0_prime_2e, p0))

    e0_elec = e0_elec_1e + e0_elec_2e

    return e0_elec
