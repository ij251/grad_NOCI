import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *


def get_swx0(wxlambda0):

    r"""Calculates the overlap between deteminants w and x.

    .. math::

            S_{wx} = \prod\limits_i^{N_e}\prescript{wx}{}\lambda_i

    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
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

            wxlambda0 = lowdin_pairing(g0_list[w], g0_list[x], mol,
                                       nelec, complexsymmetric)[0]

            s0mat[w,x] += get_swx0(wxlambda0)

    return s0mat


def get_sao1_partial(mol, atom, coord):

    r"""Function gives the overlap tensors where the bra or the ket contains
    differentiated AO basis functions, as needed to calculate terms B and D of
    the first order overlap between determinants in the theory.

    .. math::

            \left(\mathbf{S}_{\mathrm{bra}}^{(1)}\right)_{\mu\nu}
            = \left(\phi^{(1)}_{\cdot \mu}\Big|\phi^{(0)}_{\cdot \nu}\right)\\

            \left(\mathbf{S}_{\mathrm{ket}}^{(1)}\right)_{\mu\nu}
            = \left(\phi^{(0)}_{\cdot \mu}\Big|\phi^{(1)}_{\cdot \nu}\right)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: Two matrices, the first of which has the bra differentiated and
            the second the ket.
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


def get_swx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
             complexsymmetric)

    r"""Finds the derivative of the overlap between elements w and x by first
    expanding spin orbitals to first order then lowdin pairing.

    Terms A and E are due to the part of the first order spin orbitals which
    are zeroth order AO basis functions expanded using first order MO
    coefficients. We define :math:'^{w_p x}\Psi^{01}' and
    :math:'^{w x_p}\Psi^{01}'determinants where the the pth column
    of the zeroth order MO coefficient matrix has been replaced with the pth
    column of the first order MO coefficient matrix then Lowdin pair.

    Terms B and D are due to the part of the first order spin orbitals which
    are first order AO basis functions expanded using zeroth order MO
    coefficients. This is achieved by defining modified atomic orbital overlap
    matrices in which the pth row or column has been replaced by the
    corresponding overlap where either the bra or ket AO function has been
    differentiated, and using these overlap matrices in Lowdin pairing with
    the zeroth order MO coefficient matrices.

    Term C does not contribute to the overlap.

    In each case once Lowdin pairing is done overlap can be found as:

    .. math::

            S_{wx} = \prod\limits_i^{N_e}\prescript{wx}{}\lambda_i

    for each term and each p in a sum over p from 1 to the number of
    electrons.

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param w_g0: The zeroth order molecular orbital coefficient matrix of the
            wth determinant.
    :param x_g0: The zeroth order molecular orbital coefficient matrix of the
            xth determinant.
    :param w_g1: The first order molecular orbital coefficient matrix of the
            wth determinant.
    :param x_g1: The first order molecular orbital coefficient matrix of the
            xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: single value swx1 for overlap derivative.
    """

    sao0 = mol.intor("int1e_ovlp")
    s1_bra, s1_ket = get_sao1_partial(mol, atom, coord)

    swx1 = 0

    for p in range(nelec):

        wp_g01 = w_g0
        wp_g01[:,p] = w_g1[:,p] #Replace pth w_g0 column with w_g1
        wpxlambda01,_,_ = lowdin_pairing(wp_g01, x_g0, mol, nelec,
                                        complexsymmetric)

        swpx01 = lowdin_prod(wpxlambda01, []) #Term A

        wpxsao10 = sao0
        wpxsao10[p,:] = s1_bra[p,:] #Replace pth sao row with (1|0)
        wpxlambda10,_,_ = lowdin_pairing(w_g0, x_g0, wpxsao10, mol, nelec,
                                      complexsymmetric)

        swpx10 = lowdin_prod(wpxlambda10, []) #Term B

        wxpsao10 = sao0
        wxpsao10[:,p] = s1_ket[:,p] #Replace pth sao column with (0|1)
        wxplambda10,_,_ = lowdin_pairing(w_g0, x_g0, wxpsao10, mol, nelec,
                                      complexsymmetric)

        swxp10 = lowdin_prod(wxplambda10, []) #Term D

        xp_g01 = x_g0
        xp_g01[:,p] = x_g1[:,p] #Replace pth w_g0 column with w_g1
        wxplambda01,_,_ = lowdin_pairing(w_g0, xp_g01, mol, nelec,
                                        complexsymmetric)

        swxp01 = lowdin_prod(wxplambda01, []) #Term E

        swx1 += (swpx01 + swpx10 + swxp10 + swxp01)

    return swx1


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

    s1mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            w_g0 = g0_list[w]
            x_g0 = g0_list[x]
            w_g1 = g1_list[w]
            x_g1 = g1_list[x]

            s1mat[w,x] = get_swx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1,
                                  nelec complexsymmetric)

    return s1mat
