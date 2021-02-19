import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *


def get_swx0(wxlambda0, nelec):

    r"""Calculates te overlap between deteminants w and x.

    .. math::

            S_{wx} = \prod\limits_i^{N_e}\prescript{wx}{}\lambda_i

    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    """

    swx0 = lowdin_prod(wxlambda0, [])

    return swx0


def get_swx1(wxlambda0, wxlambda1, nelec):

    r"""Calculates the derivative of the overlap between determinants w and x.

    .. math::

            \frac{\partial S_{wx}}{\partial X_A}
            = \sum \limits_j^{N_e} \frac{\partial \prescript{wx}{}\lambda_j}
            {\partial X_A} \prod\limits_{i\not = j}^{N_e}
            \prescript{wx}{}\lambda_i

    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param wxlambda1: Diagonal matrix of first order Löwdin singular values
            for the wth and xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.

    :returns: The numerical value for the overlap derivative.
    """
    wxlambda1_diag = np.diag(wxlambda1)
    swx1 = np.sum(wxlambda1_diag[j,j]*lowdin_prod(wxlambda0, [j])
                  for j in range(nelec))

    return get_swx1


def get_s1mat(mol, atom, coord, g0_list, nelec, complexsymmetric: bool):

    r"""Contructs a matrix of the same dimesions as the noci expansion of the
    overlap derivatives between all determinant combinations.

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

    :returns: Matrix of overlap derivatives.
    """
    g1_list = get_g1_list(mol, atom, coord, g0_list, nelec, complexsymmetric)
    nnoci = g0_list.shape[0]
    s1mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            wxlambda0 = get_wxlambda0(g0_list[w], g0_list[x], mol,
                                      complexsymmetric)
            wxlambda1 = get_wxlambda1(g0_list[w], g1_list[w], g0_list[x],
                                      g1_list[x], mol, atom, coord,
                                      complexsymmetric)

            s1mat[w,x] = get_swx1(wxlambda0, wxlambda1, nelec)

    return s1mat
