import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *


def lowdin_pairing(w_g, x_g, sao = None, mol, nelec, complexsymmetric: bool):

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
        wxs0 = np.linalg.multi_dot([w_g0[:, 0:nelec].T.conj(), sao,
                                    x_g0[:, 0:nelec]]) #Only occ orbitals
    else:
        wxs0 = np.linalg.multi_dot([w_g0[:, 0:nelec].T, sao,
                                    x_g0[:, 0:nelec]])

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
        w_g0_t = np.dot(w_g0[:,0:nelec], wxu)
    else:
        w_g0_t = np.dot(w_g0[:,0:nelec], wxu.conj())

    x_g0_t = np.dot(x_g0[:,0:nelec], wxv)

    w_g0_t[:,0] *= det_wxu.conj() #Removes phase induced by unitary transform
    x_g0_t[:,0] *= det_wxv.conj()

    return wxlambda0, w_g0_t, x_g0_t


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


def get_j1_partial(mol, atom coord):

    r"""Function gets the 4th order two electron integral tensors where one
    of each of the 4 positions (bra0,bra1|ket0,ket1) contains differentiated
    AO basis functions, as needed to calculate terms B and D of the first
    order two electron contribution to the hamiltonian matrix element.

    .. math::

            \left(\mathbf{j}_{\mathrm{bra}_0}^{(1)}\right)_{\mu\mu'\nu\nu'}
            = \left(\phi^{(1)}_{\mu}\phi^{(0)}_{\mu'}|
              \phi^{(0)}_{\nu}\phi^{(0)}_{\nu'}\right)\\

            \left(\mathbf{j}_{\mathrm{bra}_1}^{(1)}\right)_{\mu\mu'\nu\nu'}
            = \left(\phi^{(0)}_{\mu}\phi^{(1)}_{\mu'}|
              \phi^{(0)}_{\nu}\phi^{(0)}_{\nu'}\right)

             \left(\mathbf{j}_{\mathrm{ket}_0}^{(1)}\right)_{\mu\mu'\nu\nu'}
             = \left(\phi^{(0)}_{\mu}\phi^{(0)}_{\mu'}|
               \phi^{(1)}_{\nu}\phi^{(0)}_{\nu'}\right)

             \left(\mathbf{j}_{\mathrm{ket}_1}^{(1)}\right)_{\mu\mu'\nu\nu'}
             = \left(\phi^{(0)}_{\mu}\phi^{(0)}_{\mu'}|
               \phi^{(0)}_{\nu}\phi^{(1)}_{\nu'}\right)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: 4 tensors, the first of which has bra0 differentiated, the
            second bra1, the third ket0 and the fourth ket1.
    """

    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)

    twoe = -mol.intor("int2e_ip1")[coord] #minus sign due to pyscf definition

    j1_bra0 = np.zeros_like(twoe)
    j1_bra1 = np.zeros_like(twoe)
    j1_ket0 = np.zeros_like(twoe)
    j1_ket1 = np.zeros_like(twoe)

    for i in range(twoe.shape[0]):

        atoms_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(twoe.shape[0]):

            atoms_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            for k in range(twoe.shape[0]):

                atoms_k = int(k in range(mol.aoslice_by_atom()[atom][2],
                                          mol.aoslice_by_atom()[atom][3]))

                for l in range(twoe.shape[0]):

                    atoms_l = int(l in range(mol.aoslice_by_atom()[atom][2],
                                              mol.aoslice_by_atom()[atom][3]))

                    j1_bra0[i][j][k][l] += twoe[i][j][k][l] * atoms_i
                    j1_bra1[i][j][k][l] += twoe[j][i][k][l] * atoms_j
                    j1_ket0[i][j][k][l] += twoe[k][l][i][j] * atoms_k
                    j1_ket1[i][j][k][l] += twoe[l][k][i][j] * atoms_l


    j1_bra0 = np.einsum("abcd->acbd", j1_bra0,
                        optimize='optimal') #convert to physicists
    j1_bra1 = np.einsum("abcd->acbd", j1_bra1,
                        optimize='optimal') #convert to physicists
    j1_ket0 = np.einsum("abcd->acbd", j1_ket0,
                        optimize='optimal') #convert to physicists
    j1_ket1 = np.einsum("abcd->acbd", j1_ket1,
                        optimize='optimal') #convert to physicists

    j1_bra0 = np.kron(spin_j, j1_bra0)
    j1_bra1 = np.kron(spin_j, j1_bra1)
    j1_ket0 = np.kron(spin_j, j1_ket0)
    j1_ket1 = np.kron(spin_j, j1_ket1)

    return j1_bra0, j1_bra1, j1_ket0, j1_ket1


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
