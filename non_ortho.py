import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *


def lowdin_pairing0(gw0, gx0, mol, nelec, complexsymmetric: bool):

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
    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: list: Diagonal matrix of Löwdin overlaps, transformed MO
            coefficient for w determinant, transformed MO coefficient for x
            determinant.
    """

    sao0 = mol.intor("int1e_ovlp")
    omega = np.identity(2)
    sao0 = np.kron(omega, sao0)
    if not complexsymmetric:
        wxs0 = np.linalg.multi_dot([gw0[:, 0:nelec].T.conj(), sao0,
                                    gx0[:, 0:nelec]]) #Only occ orbitals
    else:
        wxs0 = np.linalg.multi_dot([gw0[:, 0:nelec].T, sao0,
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

    gw0_t = np.append(gw0_t, gw0[:,nelec:], axis=1) #Adds virtual block back
    gx0_t = np.append(gx0_t, gw0[:,nelec:], axis=1)

    return wxlambda0, gw0_t, gx0_t


def get_wxlambda1(gw0, gw1, gx0, gx1, mol, atom, coord, nelec,
                  complexsymmetric: bool):

    r"""Calculates the first order diagonal matrix of singular values from
    differentiating the singular value decomposition:

    .. math::

            \frac{\partial \mathbf{\prescript{wx}{}\Lambda}}{\partial X_A}
            = \mathbf{I} \circ \left[\mathbf{\prescript{wx}{}U}^{\dagger}\
            \frac{\partial \mathbf{\prescript{wx}{}S}}{\partial X_A}\
            \mathbf{\prescript{wx}{}V}\right]

    where

    .. math::

             \frac{\partial \mathbf{\prescript{wx}{}S}}{\partial X_A}
             = \frac{\partial \mathbf{\prescript{w}{}G}^{\dagger\diamond}}
             {\partial X_A}\ \mathbf{S}_{\mathrm{AO}}\
             \mathbf{\prescript{x}{}G}
             + \mathbf{\prescript{w}{}G}^{\dagger\diamond}\
             \frac{\partial \mathbf{S}_{\mathrm{AO}}}{\partial X_A}\
             \mathbf{\prescript{x}{}G}
             + \mathbf{\prescript{w}{}G}^{\dagger\diamond}\
             \mathbf{S}_{\mathrm{AO}}\
             \frac{\partial \mathbf{\prescript{x}{}G}}{\partial X_A}

    :param gw0: The zeroth order molecular orbital coefficient matrix of the
            wth determinant.
    :param gw1: The first order molecular orbital coefficient matrix of the
            wth determinant.
    :param gx0: The zeroth order molecular orbital coefficient matrix of the
            xth determinant.
    :param gx1: The first order molecular orbital coefficient matrix of the
            xth determinant.
    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Diagonal matrix of singular value derivatives between
            determinants w and x.
    """

    sao0 = mol.intor("int1e_ovlp")
    omega = np.identity(2)
    sao0 = np.kron(omega, sao0)
    sao1 = get_s1(mol, atom, coord)

    if not complexsymmetric:
        wxs0 = np.linalg.multi_dot([gw0[:,0:nelec].T.conj(), sao0,
                                    gx0[:,0:nelec]])
        wxs1 = (np.linalg.multi_dot([gw1[:,0:nelec].T.conj(), sao0,
                                     gx0[:,0:nelec]])
                + np.linalg.multi_dot([gw0[:,0:nelec].T.conj(), sao1,
                                       gx0[:,0:nelec]])
                + np.linalg.multi_dot([gw0[:,0:nelec].T.conj(), sao0,
                                       gx1[:,0:nelec]]))
    else:
        wxs0 = np.linalg.multi_dot([gw0[:,0:nelec].T, sao0,
                                    gx0[:,0:nelec]])
        wxs1 = (np.linalg.multi_dot([gw1[:,0:nelec].T, sao0,
                                     gx0[:,0:nelec]])
                + np.linalg.multi_dot([gw0[:,0:nelec].T, sao1,
                                       gx0[:,0:nelec]])
                + np.linalg.multi_dot([gw0[:,0:nelec].T, sao0,
                                       gx1[:,0:nelec]]))

    wxu,_,wxvh = np.linalg.svd(wxs0)
    wxv = wxvh.T.conj()

    assert np.allclose(np.dot(wxu.T.conj(), wxu), np.identity(wxu.shape[0]),
                       rtol = 1.e-5, atol = 1.e-8)
    assert np.allclose(np.dot(wxv.T.conj(), wxv), np.identity(wxv.shape[0]),
                       rtol = 1.e-5, atol = 1.e-8)

    wxlambda1 = np.diag(np.diag(np.linalg.multi_dot([wxu.T.conj(),wxs1,wxv])))

    return wxlambda1


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
