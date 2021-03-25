import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *


def lowdin_pairing(w_g, x_g, mol, nelec, complexsymmetric: bool, sao = None):

    r"""Calculates the diagonal matrix of singular values,
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

    :param w_g: The molecular orbital coefficient matrix of the wth
            determinant.
    :param x_g: The molecular orbital coefficient matrix of the xth
            determinant.
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
        wxs0 = np.linalg.multi_dot([w_g[:, 0:nelec].T.conj(), sao,
                                    x_g[:, 0:nelec]]) #Only occ orbitals
    else:
        wxs0 = np.linalg.multi_dot([w_g[:, 0:nelec].T, sao,
                                    x_g[:, 0:nelec]])

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
        w_g_t = np.dot(w_g[:,0:nelec], wxu)
    else:
        w_g_t = np.dot(w_g[:,0:nelec], wxu.conj())

    x_g_t = np.dot(x_g[:,0:nelec], wxv)

    w_g_t[:,0] *= det_wxu.conj() #Removes phase induced by unitary transform
    x_g_t[:,0] *= det_wxv.conj()

    return wxlambda0, w_g_t, x_g_t


def lowdin_prod(wxlambda, rmind):

    r"""Calculates product of Löwdin singular values with option to remove
    certain values, for use in calculating overlap derivatives. For example,
    if j is a removed index the product looks like:

    .. math::

            \prod\limits_{i\not = j}^{N_e}\prescript{wx}{}\lambda_i

    :param wxlambda: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param rmind: A python list of indices to be removed from the product

    :returns: A value for the singular value product
    """

    wxlambda_diag = np.diag(wxlambda)
    ind = list(set(range(len(wxlambda_diag)))-set(rmind))
    lowdin_prod = np.prod(wxlambda_diag[ind])

    return lowdin_prod


def get_xw_p(w_g_t, x_g_t, orbital_num, complexsymmetric: bool):

    r"""Calculates the one particle transition density matrix
    from transformed MO coefficient matrices for determinants w and x.

    .. math::

            \prescript{wx}{}P_m
            = \prescript{w}{}{\tilde{G}}_m
            \prescript{x}{}{\tilde{G}}_m^{\dagger\diamond}

    :param w_g_t: The Löwdin transformed molecular orbital coefficient matrix
            of the wth determinant.
    :param x_g_t: The Löwdin transformed molecular orbital coefficient matrix
            of the xth determinant.
    :param orbital_num: The index of the particular MO.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The one particle transition density matrix.
    """

    if not complexsymmetric:
        xw_p = np.einsum("i,j->ij", x_g_t[:, orbital_num],
                         w_g_t[:, orbital_num].conj())
    else:
        xw_p = np.einsum("i,j->ij", x_g_t[:, orbital_num],
                         w_g_t[:, orbital_num])

    return xw_p
