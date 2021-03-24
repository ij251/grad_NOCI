import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *


def lowdin_pairing0(gw0, gx0, sao = None, mol, nelec, complexsymmetric: bool):

    r"""Calculates the zeroth order diagonal matrix of singular values,
    calculated from a singular value decomposition of the molecular orbital
    overlap matrix between determinants :math:'\prescript{w}{}{\Psi}' and
    :math:'\prescript{x}{}{\Psi}'. This is the Löwdin tranformation required
    to ensure the MOs of the determinants are bi-orthogonal.

    .. math::

            \mathbf{\prescript{wx}{}\Lambda}
            = \mathbf{\prescript{wx}{}U}^{\dagger}\
              \mathbf{\prescript{wx}{}S}\ \mathbf{\prescript{wx}{}V}

    where

    .. math::

            \mathbf{\prescript{wx}{}S}
            = \mathbf{\prescript{w}{}G}^{\dagger\diamond}\
              \mathbf{S}_{\mathrm{AO}}\ \mathbf{\prescript{x}{}G}

    The coefficient matrices are then transformed according to:

    .. math::

            \prescript{w}{}{\tilde{\mathbf{G}}}
            = \prescript{w}{}{\mathbf{G}}
            \prescript{wx}{}{\mathbf{U}}^{\diamond}

    and

    .. math::

            \prescript{x}{}{\tilde{\mathbf{G}}}
            = \prescript{x}{}{\mathbf{G}}
            \prescript{wx}{}{\mathbf{V}}

    :param gw0: The zeroth order molecular orbital coefficient matrix of the
            wth determinant.
    :param gx0: The zeroth order molecular orbital coefficient matrix of the
            xth determinant.
    :param sao: Gives option to specify different sao atomic orbital overlap
            matrix, by default set to None in which case it will use the
            standard zeroth order overlap matrix from PySCF.
    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: list: Diagonal matrix of Löwdin overlaps, transformed MO
            coefficient for w determinant, transformed MO coefficient for x
            determinant.
    """

    omega = np.identity(2)
    if sao==None:
        sao0 = mol.intor("int1e_ovlp")
        sao0 = np.kron(omega, sao0)
        sao = sao0

    if not complexsymmetric:
        wxs0 = np.linalg.multi_dot([gw0[:, 0:nelec].T.conj(), sao,
                                    gx0[:, 0:nelec]]) #Only occ orbitals
    else:
        wxs0 = np.linalg.multi_dot([gw0[:, 0:nelec].T, sao,
                                    gx0[:, 0:nelec]])

    wxu,wxlambda0,wxvh = np.linalg.svd(wxs0)
    wxv = wxvh.T.conj()
    wxlambda0 = np.diag(wxlambda0)
    det_wxu = np.linalg.det(wxu)
    det_wxv = np.linalg.det(wxv)

    assert np.allclose(np.dot(wxu.T.conj(), wxu), np.identity(wxu.shape[0]),
                       rtol = 1.e-5, atol = 1.e-8)
    assert np.allclose(np.dot(wxv.T.conj(), wxv), np.identity(wxv.shape[0]),
                       rtol = 1.e-5, atol = 1.e-8)

    if not complexsymmetric:
        gw0_t = np.dot(gw0[:,0:nelec], wxu)
    else:
        gw0_t = np.dot(gw0[:,0:nelec], wxu.conj())

    gx0_t = np.dot(gx0[:,0:nelec], wxv)

    gw0_t[:,0] *= det_wxu.conj() #Removes phase induced by unitary transform
    gx0_t[:,0] *= det_wxv.conj()

    return wxlambda0, gw0_t, gx0_t


def get_sao1_partial(mol, atom, coord):

    """Function gives parts of the sao derivative matrix needed to find the
    Lowdin overlaps in terms B and D of the matrix element derivative.
    """

    sao0 =  mol.intor("int1e_ovlp")
    onee = -mol.intor("int1e_ipovlp") #minus sign due to pyscf definition
    s1_bra = np.zeros_like(sao0)
    s1_ket = np.zeros_like(sao0)

    for i in range(sao0.shape[1]):

        atoms_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(sao0.shape[1]):

            atoms_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            s1_bra[i][j] += onee[coord][i][j]*atoms_i
            s1_ket[i][j] += onee[coord][j][i]*atoms_j

    omega = np.identity(2)
    s1_bra = np.kron(omega, s1_bra)
    s1_ket = np.kron(omega, s1_ket)

    return s1_bra, s1_ket


def lowdin_prod(wxlambda0, rmind):

    r"""Calculates product of Löwdin singular values with option to remove
    certain values, for use in calculating overlap derivatives. For example,
    if j is a removed index the product looks like:

    .. math::

            \prod\limits_{i\not = j}^{N_e}\prescript{wx}{}\lambda_i

    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param rmind: A python list of indices to be removed from the product

    :returns: A value for the singular value product
    """
    wxlambda0_diag = np.diag(wxlambda0)
    ind = list(set(range(len(wxlambda0_diag)))-set(rmind))
    lowdin_prod = np.prod(wxlambda0_diag[ind])

    return lowdin_prod


def get_g1_list(mol, atom, coord, g0_list, nelec, complexsymmetric: bool):

    r"""Converts a list of zeroth order molecular orbital coefficient matrices
    to a list of first order molecular orbital coefficient matrices.

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param g0_list: Python list of molecular orbital coefficient matrices.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Python list of first order MO coefficient matrices.
    """

    g1_list = [g1_iteration(complexsymmetric, mol, atom, coord, nelec,
                            g0_list[i]) for i in range(len(g0_list))]

    return g1_list


def get_xwp0(gw0_t, gx0_t, orbital_num, complexsymmetric: bool):

    r"""Calculates the zeroth order one particle transition density matrix
    from transformed MO coefficient matrices for determinants w and x.

    .. math::

            \prescript{wx}{}P_m
            = \prescript{w}{}{\tilde{G}}_m
            \prescript{x}{}{\tilde{G}}_m^{\dagger\diamond}

    :param gw0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the wth determinant.
    :param gx0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the xth determinant.
    :param orbital_num: The index of the particular MO.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The one particle transition density matrix.
    """

    if not complexsymmetric:
        xwp0 = np.einsum("i,j->ij", gx0_t[:, orbital_num],
                         gw0_t[:, orbital_num].conj())
    else:
        xwp0 = np.einsum("i,j->ij", gx0_t[:, orbital_num],
                         gw0_t[:, orbital_num])

    return xwp0


def get_xwp1(gw0_t, gx0_t, gw1_t, gx1_t, orbital_num, complexsymmetric: bool):

    r"""Calculates the first order one particle transition density matrix
    from transformed MO coefficient matrices for determinants w and x.

    .. math::

            \frac{\partial \prescript{wx}{}P_m}{\partial X_A}
            = \frac{\partial \prescript{w}{}{\tilde{G}}_m}{\partial X_A}
            \prescript{x}{}{\tilde{G}}_m^{\dagger\diamond}
            + \prescript{w}{}{\tilde{G}}_m  \frac{\partial
            \prescript{x}{}{\tilde{G}}_m^{\dagger\diamond}}{\partial X_A}

    :param gw0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the wth determinant.
    :param gx0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the xth determinant.
    :param gw1_t: The Löwdin transformed first order molecular orbital
            coefficient matrix of the wth determinant.
    :param gx1_t: The Löwdin transformed first order molecular orbital
            coefficient matrix of the xth determinant.
    :param orbital_num: The index of the particular MO.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The first order one particle transition density matrix.
    """

    if not complexsymmetric:
        xwp1 = (np.einsum("i,j->ij", gx0_t[:, orbital_num],
                         gw1_t[:, orbital_num].conj())
                + np.einsum("i,j->ij", gx1_t[:, orbital_num],
                            gw0_t[:, orbital_num].conj()))
    else:
        xwp1 = (np.einsum("i,j->ij", gx0_t[:, orbital_num],
                         gw1_t[:, orbital_num])
                + np.einsum("i,j->ij", gx1_t[:, orbital_num],
                            gw0_t[:, orbital_num]))

    return xwp1
