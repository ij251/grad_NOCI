import numpy as np
from pyscf import gto, scf, grad
from cphf.first_order_ghf import g1_iteration, get_s1
from non_ortho import lowdin_prod, lowdin_pairing


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
                            g0) for g0 in g0_list]

    return g1_list


def get_swx0(wxlambda0):

    r"""Calculates the overlap between deteminants w and x.

    .. math::

            S_{wx} = \prod\limits_i^{N_e}\prescript{wx}{}\lambda_i

    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.

    :returns: Numerical value for overlap.
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

    s_py = -mol.intor("int1e_ipovlp")[coord]
    #minus sign due to pyscf definition
    sao1_bra = np.zeros_like(s_py)
    sao1_ket = np.zeros_like(s_py)

    for i in range(s_py.shape[0]):

        atoms_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(s_py.shape[0]):

            atoms_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            sao1_bra[i][j] += s_py[i][j]*atoms_i
            sao1_ket[i][j] += s_py[j][i]*atoms_j

    omega = np.identity(2)
    sao1_bra = np.kron(omega, sao1_bra)
    sao1_ket = np.kron(omega, sao1_ket)

    return sao1_bra, sao1_ket


def get_swx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
             complexsymmetric: bool):

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

    omega = np.identity(2)
    sao = mol.intor("int1e_ovlp")
    sao = np.kron(omega, sao)

    sao1_bra, sao1_ket = get_sao1_partial(mol, atom, coord)

    # print("sao0:\n", sao0)
    # print("sao1_bra:\n", sao1_bra)
    # print("sao1_ket:\n", sao1_ket)
    # print("wxsmo0:\n", wxsmo0)
    # print("wxsmo1_bra:\n", wxsmo1_bra)
    # print("wxsmo1_ket:\n", wxsmo1_ket)
    swx1 = 0

    for p in range(nelec):

        print("######################\np =", p, "\n######################")
        #Term A
        wp_g01 = np.copy(w_g0)
        wp_g01[:, p] = w_g1[:, p] #Replace pth w_g0 column with w_g1
        wpxlambda01_A, wp_g01_t_A, x_g0_t_A\
                = lowdin_pairing(wp_g01, x_g0, nelec, sao,
                                 complexsymmetric)
        swpx01 = lowdin_prod(wpxlambda01_A, []) #Term A

        #Term B
        wpxlambda10_B, w_g0_t_B, x_g0_t_B,_\
                = lowdin_pairing(w_g0, x_g0, nelec, sao, complexsymmetric,
                                 sao1_bra, (p, 0))

        swpx10 = lowdin_prod(wpxlambda10_B, []) #Term B

        #Term D
        wxplambda10_D, w_g0_t_D, x_g0_t_D,_\
                = lowdin_pairing(w_g0, x_g0, nelec, sao, complexsymmetric,
                                 sao1_ket, (p, 1))

        swxp10 = lowdin_prod(wxplambda10_D, []) #Term D

        #Term E
        xp_g01 = np.copy(x_g0)
        xp_g01[:, p] = x_g1[:, p] #Replace pth x_g0 column with x_g1
        wxplambda01_E, w_g0_t_E, xp_g01_t_E\
                = lowdin_pairing(w_g0, xp_g01, nelec, sao,
                                 complexsymmetric)

        swxp01 = lowdin_prod(wxplambda01_E, []) #Term E

        swx1 += (swpx01 #A
                 + swpx10 #B
                 + swxp10 #D
                 + swxp01) #E

    # print("B all contributions:", B)
    # print("D all contributions:", D)
    # print("E all contributions:", E)
    # print("total swx1", swx1)

    return swx1


def get_s1mat(mol, atom, coord, g0_list, g1_list, nelec,
              complexsymmetric: bool):

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
    :param g1_list: Python list of molecular orbital coefficient matrix
            derivatives.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Matrix of overlap derivatives.
    """

    nnoci = len(g0_list)

    s1mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            # if w == x:
            #     continue
            w_g0 = g0_list[w]
            x_g0 = g0_list[x]
            w_g1 = g1_list[w]
            x_g1 = g1_list[x]
            print("##################\n(w,x) =", w,x, "\n##################")

            s1mat[w,x] += get_swx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1,
                                   nelec, complexsymmetric)


    return s1mat
