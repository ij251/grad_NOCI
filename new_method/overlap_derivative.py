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


def get_s0mat(mol, g0_list, nelec, complexsymmetric):

    r"""Constructs a matrix of the same dimensions as the noci expansion of
    the overlap between all determinant combinations.

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param g0_list: Python list of molecular orbital coefficient matrices.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Matrix of overlaps between determinants. 
    """

    nnoci = len(g0_list)
    s0mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            wxlambda0 = lowdin_pairing0(g0_list[w], g0_list[x], mol,
                                      nelec, complexsymmetric)[0]

            s0mat[w,x] += lowdin_prod(wxlambda0, [])

    return s0mat


def get_s1mat(mol, atom, coord, g0_list, nelec, complexsymmetric: bool):

    r"""Constructs a matrix of the same dimensions as the noci expansion of
    the overlap derivatives between all determinant combinations.

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
    nnoci = len(g0_list)

    sao0 = mol.intor("int1e_ovlp")
    s1_bra, s1_ket = get_sao1_partial(mol, atom, coord)


    s1mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            gw0 = g0_list[w]
            gx0 = g0_list[x]
            gw1 = g1_list[w]
            gx1 = g1_list[x]

            for p in range(nelec):

                gwp01 = gw0
                gwp01[:,p] = gw1[:,p] #Replace pth gw0 column with gw1
                wpxlambda01,_,_ = lowdin_pairing0(gwp01, gx0, mol, nelec,
                                                complexsymmetric)

                a_p = lowdin_prod(wpxlambda01, [])

                wpxsao10 = sao0
                wpxsao10[p,:] = s1_bra[p,:] #Replace pth sao row with (1|0)
                wpxlambda10 = lowdin_pairing0(gw0, gx0, wpxsao10, mol, nelec,
                                              complexsymmetric)

                b_p = lowdin_prod(wpxlambda10, [])

                wxpsao10 = sao0
                wxpsao10[:,p] = s1_ket[:,p] #Replace pth sao column with (0|1)
                wxplambda10 = lowdin_pairing0(gw0, gx0, wxpsao10, mol, nelec,
                                              complexsymmetric)

                d_p = lowdin_prod(wxplambda10, [])

                gxp01 = gx0
                gxp01[:,p] = gx1[:,p] #Replace pth gw0 column with gw1
                wxplambda01,_,_ = lowdin_pairing0(gw0, gxp01, mol, nelec,
                                                complexsymmetric)

                e_p = lowdin_prod(wxplambda01, [])

                s1mat[w,x] += a_p + b_p + d_p + e_p

    return s1mat
