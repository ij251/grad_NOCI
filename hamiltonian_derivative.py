import numpy as np
from pyscf import gto, scf, grad
from cphf.zeroth_order_ghf import get_hcore0, get_j0, get_e0_nuc
from cphf.first_order_ghf import get_hcore1, get_s1
from overlap_derivative import get_g1_list, get_swx0, get_sao1_partial,\
        get_swx1
from non_ortho import lowdin_pairing, lowdin_prod, get_xw_p


def get_hcore1_op(mol, atom, coord):

    r"""Function to get the matrix element of the hcore derivative containing
    just the contribution from the operator derivative and not that of the
    basis functions.

    .. math::

            \left(\mathbf{h}_{\mathrm{op}}^{(1)}\right)_{\mu\nu}
            = \left(\phi^{(0)}_{\cdot \mu}\left|\hat{\mathbf{h}}^{(1)}\right|
            \phi^{(0)}_{\cdot \nu}\right)

    Based on a function from PySCF that calculates the full hcore derivative,
    so to achieve what we want i have set the zeroth order hcore to be zero.
    """

    with_ecp = mol.has_ecp()
    if with_ecp:
        ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
    else:
        ecp_atoms = ()

    aoslices = mol.aoslice_by_atom()

    def hcore_deriv(atm_id):

        shl0, shl1, p0, p1 = aoslices[atm_id]

        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= -mol.atom_charge(atm_id)
            if with_ecp and atm_id in ecp_atoms:
                vrinv += mol.intor('ECPscalar_iprinv', comp=3)

        return vrinv + vrinv.transpose(0,2,1)

    hcore1_op = hcore_deriv(atom)[coord]
    omega = np.identity(2)

    hcore1_op = np.kron(omega, hcore1_op)

    return hcore1_op


def get_hcore1_partial(mol, atom, coord):

    r"""Function gives the core hamiltonian tensors where the bra or the ket
    contains differentiated AO basis functions, as needed to calculate terms
    B and D of the first order one electron hamiltonian element between
    determinants in the theory.

    .. math::

            \left(\mathbf{h}_{\mathrm{bra}}^{(1)}\right)_{\mu\nu}
            = \left(\phi^{(1)}_{\cdot \mu}\left|\hat{\mathbf{h}}^{(0)}\right|
            \phi^{(0)}_{\cdot \nu}\right)

            \left(\mathbf{h}_{\mathrm{ket}}^{(1)}\right)_{\mu\nu}
            = \left(\phi^{(0)}_{\cdot \mu}\left|\hat{\mathbf{h}}^{(0)}\right|
            \phi^{(1)}_{\cdot \nu}\right)

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

    hcore_py = -(mol.intor('int1e_ipnuc')
                 + mol.intor('int1e_ipkin'))[coord]
    #minus sign due to pyscf definition

    hcore1_bra = np.zeros_like(hcore_py)
    hcore1_ket = np.zeros_like(hcore_py)

    for i in range(hcore_py.shape[0]):

        atoms_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(hcore_py.shape[0]):

            atoms_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            hcore1_bra[i][j] += hcore_py[i][j]*atoms_i
            hcore1_ket[i][j] += hcore_py[j][i]*atoms_j

    omega = np.identity(2)
    hcore1_bra = np.kron(omega, hcore1_bra)
    hcore1_ket = np.kron(omega, hcore1_ket)

    return hcore1_bra, hcore1_ket


def get_j1_partial(mol, atom, coord):

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

    pi_py = -mol.intor("int2e_ip1")[coord] #minus sign due to pyscf definition

    j1_bra0 = np.zeros_like(pi_py)
    j1_bra1 = np.zeros_like(pi_py)
    j1_ket0 = np.zeros_like(pi_py)
    j1_ket1 = np.zeros_like(pi_py)

    for i in range(pi_py.shape[0]):

        atoms_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(pi_py.shape[0]):

            atoms_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            for k in range(pi_py.shape[0]):

                atoms_k = int(k in range(mol.aoslice_by_atom()[atom][2],
                                          mol.aoslice_by_atom()[atom][3]))

                for l in range(pi_py.shape[0]):

                    atoms_l = int(l in range(mol.aoslice_by_atom()[atom][2],
                                              mol.aoslice_by_atom()[atom][3]))

                    j1_bra0[i][j][k][l] += pi_py[i][j][k][l] * atoms_i
                    j1_bra1[i][j][k][l] += pi_py[j][i][k][l] * atoms_j
                    j1_ket0[i][j][k][l] += pi_py[k][l][i][j] * atoms_k
                    j1_ket1[i][j][k][l] += pi_py[l][k][i][j] * atoms_l


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


def get_onewx0(mol, w_g0_t, x_g0_t, wxlambda0, nelec,
               complexsymmetric: bool):

    r"""Calculates the one electron contribution to the hamiltonian element
    between determinants w and x.

    .. math::

            \sum\limits_{m=1}^{N_e}\langle\prescript{w}{}\Psi
            |\hat{h}(\mathbf{r}_m)|\prescript{x}{}\Psi\rangle
            =  h_{\delta\mu, \gamma\nu}\sum\limits_{m=1}^{N_e}
            \left(\prod\limits_{i\not =m}^{N_e}
            \prescript{wx}{}\lambda_{i}\right)
            \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param w_g0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the wth determinant.
    :param x_g0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the xth determinant.
    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Value for one electron contribution to hamiltonian element.
    """

    hcore0 = get_hcore0(mol)

    onewx0 = 0

    for m in range(nelec):

        lowdin_prod0 = lowdin_prod(wxlambda0, [m])
        xw_p0 = get_xw_p(w_g0_t, x_g0_t, m, complexsymmetric)

        onewx0 += lowdin_prod0 * np.einsum("ij,ji->", hcore0, xw_p0)

    return onewx0


def get_onewx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
               complexsymmetric: bool):

    r"""Calculates the first order one electron contribution to the
    hamiltonian element between determinants w and x by first expanding
    spin orbitals to first order then lowdin pairing to simplify and
    calculate matrix element.

    .. math::

            \sum\limits_{m=1}^{N_e}\langle\prescript{w}{}\Psi
            |\hat{h}(\mathbf{r}_m)|\prescript{x}{}\Psi\rangle
            =  h_{\delta\mu, \gamma\nu}\sum\limits_{m=1}^{N_e}
            \left(\prod\limits_{i\not =m}^{N_e}
            \prescript{wx}{}\lambda_{i}\right)
            \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}

    Terms A and E are due to the part of the first order spin orbitals which
    are zeroth order AO basis functions expanded using first order MO
    coefficients. We define :math:'^{w_p x}\Psi^{01}' and
    :math:'^{w x_p}\Psi^{01}'determinants where the the pth column
    of the zeroth order MO coefficient matrix has been replaced with the pth
    column of the first order MO coefficient matrix then Lowdin pair.
    Once pairing is done the matrix element is calculated as for the zeroth
    order case above for each p in a sum over p from 1 to the number of
    electrons.

    Terms B and D are due to the part of the first order spin orbitals which
    are first order AO basis functions expanded using zeroth order MO
    coefficients. This is achieved by defining modified atomic orbital overlap
    matrices in which a row or column has been replaced by the corresponding
    overlap where either the bra or ket AO function has been differentiated,
    and using these overlap matrices in Lowdin pairing with the zeroth order
    MO coefficient matrices.
    Once pairing is done the matrix element is calculated as for the zeroth
    order case above for each p, except we must now consider that the pth spin
    orbital is expanded in first order AO basis functions, so for the m=p term
    in the sum over m the zeroth order core hamiltonian matrix is replaced
    with the matrix in which either the bra or ket of each element has been
    differentiated.

    Term C does contribute here because the operator is a function of nuclear
    coordinate, and this is found as above (no sum over p) but replacing the
    zeroth order core hamiltonian matrix with the matrix in which the operator
    has been differentiated.

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

    :returns: Value for one electron contribution to hamiltonian element
            derivative.
    """

    omega = np.identity(2)
    sao0 = mol.intor("int1e_ovlp")
    sao0 = np.kron(omega, sao0)

    sao1_bra, sao1_ket = get_sao1_partial(mol, atom, coord)

    # if not complexsymmetric:
    #     wxsmo0 = np.linalg.multi_dot([w_g0[:, 0:nelec].T.conj(), sao0,
    #                                   x_g0[:, 0:nelec]]) #Only occ orbitals
    #     wxsmo1_bra = np.linalg.multi_dot([w_g0[:, 0:nelec].T.conj(), sao1_bra,
    #                                      x_g0[:, 0:nelec]])
    #     wxsmo1_ket = np.linalg.multi_dot([w_g0[:, 0:nelec].T.conj(), sao1_ket,
    #                                       x_g0[:, 0:nelec]])
    # else:
    #     wxsmo0 = np.linalg.multi_dot([w_g0[:, 0:nelec].T, sao0,
    #                                   x_g0[:, 0:nelec]]) #Only occ orbitals
    #     wxsmo1_bra = np.linalg.multi_dot([w_g0[:, 0:nelec].T, sao1_bra,
    #                                       x_g0[:, 0:nelec]])
    #     wxsmo1_ket = np.linalg.multi_dot([w_g0[:, 0:nelec].T, sao1_ket,
    #                                       x_g0[:, 0:nelec]])

    # print("sao0:\n", sao0)
    # print("sao1_bra:\n", sao1_bra)
    # print("sao1_ket:\n", sao1_ket)
    # print("wxsmo0:\n", wxsmo0)
    # print("wxsmo1_bra:\n", wxsmo1_bra)
    # print("wxsmo1_ket:\n", wxsmo1_ket)
    hcore0 = get_hcore0(mol)
    hcore1_bra, hcore1_ket = get_hcore1_partial(mol, atom, coord)
    hcore1_op = get_hcore1_op(mol, atom, coord)

    onewx1 = 0

    for p in range(nelec):

        print("\n")
        print("#######################\np =", p, "\n#######################")
        onewpx01 = 0 #Term A
        onewpx10 = 0 #Term B
        onewxp10 = 0 #Term D
        onewxp01 = 0 #Term E

        # Lowdin pairing for the various terms
        omega = np.identity(2)
        sao = mol.intor("int1e_ovlp")
        sao = np.kron(omega, sao)

        #Term A
        wp_g01 = np.copy(w_g0)
        wp_g01[:, p] = w_g1[:, p] #Replace pth w_g0 column with w_g1
        print("Lowdin A")
        wpxlambda01_A, wp_g01_t_A, x_g0_t_A\
                = lowdin_pairing(wp_g01, x_g0, nelec, sao,
                                 complexsymmetric)

        #Term B
        print("Lowdin B")
        wpxlambda10_B, w_g0_t_B, x_g0_t_B, w_g_t_p_B\
                = lowdin_pairing(w_g0, x_g0, nelec, sao, complexsymmetric,
                                 sao1_bra, (p, 0))
        print("wpxlambda10_B:\n", wpxlambda10_B)
        print("w_g0_t_B:\n", w_g0_t_B)
        print("x_g0_t_B:\n", x_g0_t_B)

        #Term D
        print("Lowdin D")
        wxplambda10_D, w_g0_t_D, x_g0_t_D, x_g_t_p_D\
                = lowdin_pairing(w_g0, x_g0, nelec, sao, complexsymmetric,
                                 sao1_ket, (p, 1))
        print("wxplambda10_D:\n", wpxlambda10_B)
        print("w_g0_t_D:\n", w_g0_t_D)
        print("x_g0_t_D:\n", x_g0_t_D)
        print("x_g_t_p_D:\n", x_g_t_p_D)

        #Term E
        print("Lowdin E")
        xp_g01 = np.copy(x_g0)
        xp_g01[:, p] = x_g1[:, p] #Replace pth x_g0 column with x_g1
        wxplambda01_E, w_g0_t_E, xp_g01_t_E\
                = lowdin_pairing(w_g0, xp_g01, nelec, sao,
                                 complexsymmetric)

        for m in range(nelec):

            print("==================\nm =", m, "\n==================")

            # Term A
            wpxlowdin_prod01 = lowdin_prod(wpxlambda01_A, [m])
            xwp_p01_A = get_xw_p(wp_g01_t_A, x_g0_t_A, m, complexsymmetric)

            onewpx01 += wpxlowdin_prod01 * np.einsum("ij,ji->", hcore0,
                                                     xwp_p01_A)

            # Term B
            wpxlowdin_prod10 = lowdin_prod(wpxlambda10_B, [m])
            print(f"Reduced overlap B term for m = {m}:", wpxlowdin_prod10)
            xwp_p10_m_B = get_xw_p(w_g0_t_B, x_g0_t_B, m, complexsymmetric)
            xwp_p10_p_B = get_xw_p(w_g_t_p_B, x_g0_t_B, m, complexsymmetric)

            hom_term = np.einsum("ij,ji->", hcore0, xwp_p10_m_B)
            het_term = np.einsum("ij,ji->", (hcore1_bra - hcore0), xwp_p10_p_B)
            onewpx10 += wpxlowdin_prod10 * (hom_term + het_term)
            print(f"Contribution to B from m = {m}:", wpxlowdin_prod10 * (hom_term + het_term))

            # Term D
            wxplowdin_prod10 = lowdin_prod(wxplambda10_D, [m])
            print(f"Reduced overlap D term for m = {m}:", wxplowdin_prod10)
            xpw_p10_m_D = get_xw_p(w_g0_t_D, x_g0_t_D, m, complexsymmetric)
            xpw_p10_p_D = get_xw_p(w_g0_t_D, x_g_t_p_D, m, complexsymmetric)

            hom_term = np.einsum("ij,ji->", hcore0, xpw_p10_m_D)
            het_term = np.einsum("ij,ji->", (hcore1_ket - hcore0), xpw_p10_p_D)
            onewxp10 += wxplowdin_prod10 * (hom_term + het_term)
            print(f"Contribution to D from m = {m}:", wxplowdin_prod10 * (hom_term + het_term))

            # if m == p:
            #     onewxp10 += wxplowdin_prod10 * np.einsum("ij,ji->",
            #                                              hcore1_ket,
            #                                              xpw_p10_D)
            # else:
            #     onewxp10 += wxplowdin_prod10 * np.einsum("ij,ji->",
            #                                              hcore0,
            #                                              xpw_p10_D)

            # Term E
            wxplowdin_prod01 = lowdin_prod(wxplambda01_E, [m])
            xpw_p01_E = get_xw_p(w_g0_t_E, xp_g01_t_E, m, complexsymmetric)
            onewxp01 += wxplowdin_prod01 * np.einsum("ij,ji->", hcore0,
                                                     xpw_p01_E)


            # print("A lowdin:\n", wpxlambda01_A)
            # print("D lowdin:\n", wxplambda10_D)
            # print("E lowdin:\n", wxplambda01_E)

        onewx1 += onewpx01 + onewpx10 + onewxp10 + onewxp01

        print("")
        print(f"A contribution 1e from p = {p}:", onewpx01)
        print(f"B contribution 1e from p = {p}:", onewpx10)
        print(f"D contribution 1e from p = {p}:", onewxp10)
        print(f"E contribution 1e from p = {p}:", onewxp01)


    # Term C
    wxlambda0, w_g0_t, x_g0_t = lowdin_pairing(w_g0, x_g0, nelec, sao,
                                               complexsymmetric)
    onewx010 = 0

    for m in range(nelec):

        lowdin_prod0 = lowdin_prod(wxlambda0, [m])
        xw_p0 = get_xw_p(w_g0_t, x_g0_t, m, complexsymmetric)

        onewx010 += lowdin_prod0 * np.einsum("ij,ji->", hcore1_op, xw_p0)

    onewx1 += onewx010
    print("C contribution:", onewx010)
    print("onewx1:", onewx1)
    return onewx1


def get_twowx0(mol, w_g0_t, x_g0_t, wxlambda0, nelec, complexsymmetric: bool):

    r"""Calculates the two electron contribution to the hamiltonian element
    between determinants w and x.

    .. math::

            \frac{1}{2}(\delta\mu,\gamma\nu|\delta'\mu',\gamma'\nu')
            \sum\limits_m^{N_e}\sum\limits_{n}^{N_e} \left(
            \prod\limits_{i\not =m,n}^{N_e}\prescript{wx}{}\lambda_{i}\right)
            \left[ \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}
            \prescript{xw}{}P_n^{\gamma'\nu', \delta'\mu'} -  \prescript{xw}{}
            P_m^{\gamma'\nu', \delta\mu} \prescript{xw}{}
            P_n^{\gamma\nu, \delta'\mu'} \right]

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param w_g0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the wth determinant.
    :param x_g0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the xth determinant.
    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Value for two electron contribution to hamiltonian element.
    """

    j0 = get_j0(mol)

    twowx0 = 0
    for m in range(nelec):
        for n in range(nelec):

            if n == m:
                continue

            lowdin_prod0 = lowdin_prod(wxlambda0, [m,n])
            xw_p0_m = get_xw_p(w_g0_t, x_g0_t, m, complexsymmetric)
            xw_p0_n = get_xw_p(w_g0_t, x_g0_t, n, complexsymmetric)

            twowx0 += lowdin_prod0 * (np.einsum("ijkl,ji,lk->",
                                                j0, xw_p0_m, xw_p0_n)
                                      - np.einsum("ijkl,li,jk->",
                                                  j0, xw_p0_m, xw_p0_n)) * 0.5

    return twowx0


def get_twowx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
               complexsymmetric: bool):

    r"""Calculates the first order two electron contribution to the
    hamiltonian element between determinants w and x by first expanding
    spin orbitals to first order then lowdin pairing to simplify and
    calculate matrix element.

    .. math::

            \frac{1}{2}(\delta\mu,\gamma\nu|\delta'\mu',\gamma'\nu')
            \sum\limits_m^{N_e}\sum\limits_{n}^{N_e} \left(
            \prod\limits_{i\not =m,n}^{N_e}\prescript{wx}{}\lambda_{i}\right)
            \left[ \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}
            \prescript{xw}{}P_n^{\gamma'\nu', \delta'\mu'} -  \prescript{xw}{}
            P_m^{\gamma'\nu', \delta\mu} \prescript{xw}{}
            P_n^{\gamma\nu, \delta'\mu'} \right]

    Terms A and E are due to the part of the first order spin orbitals which
    are zeroth order AO basis functions expanded using first order MO
    coefficients. We define :math:'^{w_p x}\Psi^{01}' and
    :math:'^{w x_p}\Psi^{01}'determinants where the the pth column
    of the zeroth order MO coefficient matrix has been replaced with the pth
    column of the first order MO coefficient matrix then Lowdin pair.
    Once pairing is done the matrix element is calculated as for the zeroth
    order case above for each p in a sum over p from 1 to the number of
    electrons.

    Terms B and D are due to the part of the first order spin orbitals which
    are first order AO basis functions expanded using zeroth order MO
    coefficients. This is achieved by defining modified atomic orbital overlap
    matrices in which a row or column has been replaced by the corresponding
    overlap where either the bra or ket AO function has been differentiated,
    and using these overlap matrices in Lowdin pairing with the zeroth order
    MO coefficient matrices.
    Once pairing is done the matrix element is calculated as for the zeroth
    order case above for each p, except we must now consider that the pth spin
    orbital is expanded in first order AO basis functions, so for the m=p and
    n=p terms in the sums over m and n the zeroth order two electron integral
    coulomb tensor has been replaced by the tensor in which either of the
    two AO functions in the bra or either of the two in the ket has been
    differentiated.

    Term C does not contribute to the two electron term.

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

    :returns: Value for two electron contribution to hamiltonian element
            derivative.
    """

    omega = np.identity(2)
    sao0 = mol.intor("int1e_ovlp")
    sao0 = np.kron(omega, sao0)

    sao1_bra, sao1_ket = get_sao1_partial(mol, atom, coord)

    if not complexsymmetric:
        wxsmo0 = np.linalg.multi_dot([w_g0[:, 0:nelec].T.conj(), sao0,
                                      x_g0[:, 0:nelec]]) #Only occ orbitals
        wxsmo1_bra = np.linalg.multi_dot([w_g0[:, 0:nelec].T.conj(), sao1_bra,
                                          x_g0[:, 0:nelec]])
        wxsmo1_ket = np.linalg.multi_dot([w_g0[:, 0:nelec].T.conj(), sao1_ket,
                                          x_g0[:, 0:nelec]])
    else:
        wxsmo0 = np.linalg.multi_dot([w_g0[:, 0:nelec].T, sao0,
                                      x_g0[:, 0:nelec]]) #Only occ orbitals
        wxsmo1_bra = np.linalg.multi_dot([w_g0[:, 0:nelec].T, sao1_bra,
                                          x_g0[:, 0:nelec]])
        wxsmo1_ket = np.linalg.multi_dot([w_g0[:, 0:nelec].T, sao1_ket,
                                          x_g0[:, 0:nelec]])

    j0 = get_j0(mol)
    j1_bra0, j1_bra1, j1_ket0, j1_ket1 = get_j1_partial(mol, atom, coord)

    twowx1 = 0

    for p in range(nelec):

        twowpx01 = 0 #Term A
        twowpx10 = 0 #Term B
        twowxp10 = 0 #Term D
        twowxp01 = 0 #Term E

        # Lowdin pairing for the various terms
        omega = np.identity(2)
        sao = mol.intor("int1e_ovlp")
        sao = np.kron(omega, sao)

        #Term A
        wp_g01 = np.copy(w_g0)
        wp_g01[:, p] = w_g1[:, p] #Replace pth w_g0 column with w_g1
        wpxlambda01_A, wp_g01_t_A, x_g0_t_A\
                = lowdin_pairing(wp_g01, x_g0, nelec, sao,
                                 complexsymmetric)

        #Term B
        wpxlambda10_B, w_g0_t_B, x_g0_t_B\
                = lowdin_pairing(w_g0, x_g0, nelec, sao, complexsymmetric,
                                 sao1_bra, (p, 0))

        #Term D
        wxplambda10_D, w_g0_t_D, x_g0_t_D\
                = lowdin_pairing(w_g0, x_g0, nelec, sao, complexsymmetric,
                                 sao1_ket, (p, 1))

        #Term E
        xp_g01 = np.copy(x_g0)
        xp_g01[:, p] = x_g1[:, p] #Replace pth x_g0 column with x_g1
        wxplambda01_E, w_g0_t_E, xp_g01_t_E\
                = lowdin_pairing(w_g0, xp_g01, nelec, sao,
                                 complexsymmetric)

        for m in range(nelec):
            for n in range(nelec):

                if n == m:
                    continue

                #Term A
                wpxlowdin_prod01 = lowdin_prod(wpxlambda01_A, [m,n])
                xwp_p01_A_m = get_xw_p(wp_g01_t_A, x_g0_t_A, m, complexsymmetric)
                xwp_p01_A_n = get_xw_p(wp_g01_t_A, x_g0_t_A, n, complexsymmetric)

                twowpx01 += (wpxlowdin_prod01*0.5*
                             (np.einsum("ijkl,ji,lk->",
                                        j0, xwp_p01_A_m, xwp_p01_A_n)
                              - np.einsum("ijkl,li,jk->",
                                          j0, xwp_p01_A_m, xwp_p01_A_n)))

                #Term B
                wpxlowdin_prod10 = lowdin_prod(wpxlambda10_B, [m,n])
                xwp_p10_B_m = get_xw_p(w_g0_t_B, x_g0_t_B, m, complexsymmetric)
                xwp_p10_B_n = get_xw_p(w_g0_t_B, x_g0_t_B, n, complexsymmetric)

                if m == p:
                    twowpx10 += (wpxlowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_bra0, xwp_p10_B_m, xwp_p10_B_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_bra0, xwp_p10_B_m, xwp_p10_B_n)))
                elif n == p:
                    twowpx10 += (wpxlowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_bra1, xwp_p10_B_m, xwp_p10_B_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_bra1, xwp_p10_B_m, xwp_p10_B_n)))
                else:
                    twowpx10 += (wpxlowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j0, xwp_p10_B_m, xwp_p10_B_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j0, xwp_p10_B_m, xwp_p10_B_n)))

                #Term D
                wxplowdin_prod10 = lowdin_prod(wxplambda10_D, [m,n])
                xpw_p10_D_m = get_xw_p(w_g0_t_D, x_g0_t_D, m, complexsymmetric)
                xpw_p10_D_n = get_xw_p(w_g0_t_D, x_g0_t_D, n, complexsymmetric)

                if m == p:
                    twowxp10 += (wxplowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_ket0, xpw_p10_D_m, xpw_p10_D_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_ket0, xpw_p10_D_m, xpw_p10_D_n)))
                elif n == p:
                    twowxp10 += (wxplowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_ket1, xpw_p10_D_m, xpw_p10_D_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_ket1, xpw_p10_D_m, xpw_p10_D_n)))
                else:
                    twowxp10 += (wxplowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j0, xpw_p10_D_m, xpw_p10_D_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j0, xpw_p10_D_m, xpw_p10_D_n)))

                #Term E
                wxplowdin_prod01 = lowdin_prod(wxplambda01_E, [m,n])
                xpw_p01_E_m = get_xw_p(w_g0_t_E, xp_g01_t_E, m, complexsymmetric)
                xpw_p01_E_n = get_xw_p(w_g0_t_E, xp_g01_t_E, n, complexsymmetric)

                twowxp01 += (wxplowdin_prod01*0.5*
                             (np.einsum("ijkl,ji,lk->",
                                        j0, xpw_p01_E_m, xpw_p01_E_n)
                              - np.einsum("ijkl,li,jk->",
                                          j0, xpw_p01_E_m, xpw_p01_E_n)))

        twowx1 += twowpx01 + twowpx10 + twowxp10 + twowxp01
        # print("")
        # print(f"A contribution 2e from p = {p}:", twowpx01)
        # print(f"B contribution 2e from p = {p}:", twowpx10)
        # print(f"D contribution 2e from p = {p}:", twowxp10)
        # print(f"E contribution 2e from p = {p}:", twowxp01)

    return twowx1


def get_nucwx0(mol, wxlambda0, complexsymmetric: bool):

    r"""Calculate the nuclear repulsion contribution to the hamiltonaian
    element between determinants w and x.

    .. math::

            \langle\prescript{w}{}\Psi|\prescript{x}{}\Psi\rangle
            \sum\limits_{A>B}^N\frac{Z_AZ_B}{R_{AB}}
            = S_{wx}\sum\limits_{A>B}^N\frac{Z_AZ_B}{R_{AB}}

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The nuclear repulsion contribution to the hamiltonian element.
    """

    swx0 = get_swx0(wxlambda0)
    e0_nuc = get_e0_nuc(mol)

    nucwx0 = swx0 * e0_nuc

    return nucwx0


def get_nucwx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
               complexsymmetric: bool):

    r"""Calculates the first order nuclear repulsion contribution to the
    hamiltonian element between determinants w and x. Requires the first order
    overlap and the first order nuclear repulsion operator.

    .. math::

            \frac{\partial S_{wx}}{\partial X_A}\sum\limits_{A>B}^N
            \frac{Z_AZ_B}{R_{AB}} + S_{wx}\sum\limits_{B \neq A}^N
            \left(X_B-X_A\right)\frac{Z_AZ_B}{R^3_{AB}}

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

    :returns: Value for nuclear repulsion contribution to hamiltonian element
            derivative.
    """

    swx0 = get_swx0(wxlambda0)
    swx1 = get_swx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
                    complexsymmetric)

    e0_nuc = get_e0_nuc(mol)
    e1_nuc = get_e1_nuc(mol, atom, coord)

    nucwx1 = swx1 * e0_nuc + swx0 * e1_nuc

    return nucwx1


def get_h0mat(mol, g0_list, nelec, complexsymmetric: bool):

    r"""Constructs a matrix of the same dimensions as the noci expansion of
    the hamiltonian elements between all determinant combinations.

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param g0_list: Python list of molecular orbital coefficient matrices.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Matrix of hamiltonian elements.
    """
    nnoci = len(g0_list)
    h0mat = np.zeros((nnoci,nnoci))
    onemat = np.zeros((nnoci,nnoci))
    twomat = np.zeros((nnoci,nnoci))
    nucmat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            w_g0 = g0_list[w]
            x_g0 = g0_list[x]

            wxlambda0, w_g0_t, x_g0_t = lowdin_pairing(w_g0, x_g0, mol, nelec,
                                                       complexsymmetric)

            onewx0 = get_onewx0(mol, w_g0_t, x_g0_t, wxlambda0, nelec,
                                complexsymmetric)
            twowx0 = get_twowx0(mol, w_g0_t, x_g0_t, wxlambda0, nelec,
                                complexsymmetric)
            nucwx0 = get_nucwx0(mol, wxlambda0, complexsymmetric)

            h0mat[w, x] = onewx0 + twowx0 + nucwx0

            onemat[w, x] = onewx0
            twomat[w, x] = twowx0
            nucmat[w, x] = nucwx0

    return h0mat


def get_h1mat(mol, atom, coord, g0_list, g1_list, nelec,
              complexsymmetric: bool):

    r"""Constructs a matrix of the same dimensions as the noci expansion of
    the hamiltonian element derivatives between all determinant combinations.

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

    :returns: Matrix of hamiltonian element derivatives.
    """
    nnoci = len(g0_list)
    h1mat = np.zeros((nnoci,nnoci))

    # for w in range(nnoci):
    #     for x in range(nnoci):
    for w in [0]:
        for x in [1]:

            w_g0 = g0_list[w]
            x_g0 = g0_list[x]
            w_g1 = g1_list[w]
            x_g1 = g1_list[x]
            print("###################\n(w, x)=", w, x, "\n###################")
            onewx1 = get_onewx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1,
                                nelec, complexsymmetric)
            # twowx1 = get_twowx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1,
            #                     nelec, complexsymmetric)

            # BCH: nucwx1 = get_twowx1???
            # nucwx1 = get_twowx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1,
            #                     nelec, complexsymmetric)

            # print("twowx1:", twowx1)
            h1mat[w, x] = onewx1
            # h1mat[w, x] = onewx1 + twowx1 + nucwx1

    return h1mat
