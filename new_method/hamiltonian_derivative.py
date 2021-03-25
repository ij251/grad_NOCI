import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *
from overlap_derivative import *

def get_hcore1_operator(mf, mol=None):

    """Function to get the matrix element of the hcore derivative containing
    just the contribution from the operator derivative and not that of the
    basis functions.

    Based on a function from PySCF that calculates the full hcore derivative,
    so to achieve what we want i have set the zeroth order hcore to be zero.
    """

    if mol is None: mol = mf.mol
    with_x2c = getattr(mf.base, 'with_x2c', None)

    if with_x2c:

        hcore_deriv = with_x2c.hcore_deriv_generator(deriv=1)

    else:

        with_ecp = mol.has_ecp()
        if with_ecp:
            ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()

        aoslices = mol.aoslice_by_atom()
        h1 = mf.get_hcore(mol)

        def hcore_deriv(atm_id):

            shl0, shl1, p0, p1 = aoslices[atm_id]

            with mol.with_rinv_at_nucleus(atm_id):
                vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                vrinv *= -mol.atom_charge(atm_id)
                if with_ecp and atm_id in ecp_atoms:
                    vrinv += mol.intor('ECPscalar_iprinv', comp=3)

            vrinv[:,p0:p1] += h1[:,p0:p1]

            return vrinv + vrinv.transpose(0,2,1)

    return hcore_deriv


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
        xw_p0 = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)

        onewx0 += lowdin_prod0 * np.einsum("ij,ji->", hcore0, xw_p0)

    return onewx0


def get_onewx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
               complexsymmetric: bool):

    r"""Calculates the first order one electron contribution to the
    hamiltonian element between determinants w and x using the new
    formulation. Terms A-E here are defined as in the theory.

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

    sao0 = mol.intor("int1e_ovlp")
    s1_bra, s1_ket = get_sao1_partial(mol, atom, coord)
    hcore0 = get_hcore0(mol)

    onewx1 = 0

    for p in range(nelec):

        onewpx01 = 0 #Term A
        onewpx10 = 0 #Term B
        onewxp10 = 0 #Term D
        onewxp01 = 0 #Term E

        for m in range(nelec):

            #Term A
            wp_g01 = w_g0
            wp_g01[:,p] = w_g1[:,p] #Replace pth w_g0 column with w_g1
            wpxlambda01, wp_g01_t, x_g0_t = lowdin_pairing(wp_g01, x_g0, mol,
                                                           nelec,
                                                           complexsymmetric
                                                           )
            wpxlowdin_prod01 = lowdin_prod(wpxlambda01, [m])
            xwp_p01 = get_xw_p0(wp_g01_t, x_g0_t, m, complexsymmetric)

            onewpx01 += wpxlowdin_prod01 * np.einsum("ij,ji->", hcore0,
                                                     xwp_p01)

            #Term B
            wpxsao10 = sao0
            wpxsao10[p,:] = s1_bra[p,:] #Replace pth sao row with (1|0)
            wpxlambda10, w_g0_t, x_g0_t = lowdin_pairing(w_g0, x_g0,
                                                        wpxsao10, mol,
                                                        nelec,
                                                        complexsymmetric)
            wpxlowdin_prod10 = lowdin_prod(wpxlambda10, [m])
            xwp_p10 = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)

            if m == p:
                onewpx10 += wpxlowdin_prod10 * np.einsum("ij,ji->",
                                                         hcore1_bra,
                                                         xwp_p10)
            else:
                onewpx10 += wpxlowdin_prod10 * np.einsum("ij,ji->",
                                                         hcore0,
                                                         xwp_p10)

            #Term D
            wxpsao10 = sao0
            wxpsao10[:,p] = s1_ket[:,p] #Replace pth sao column with (0|1)
            wxplambda10, w_g0_t, x_g0_t = lowdin_pairing(w_g0, x_g0,
                                                        wxpsao10, mol,
                                                        nelec,
                                                        complexsymmetric)
            wxplowdin_prod10 = lowdin_prod(wxplambda10, [m])
            xpw_p10 = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)

            if m == p:
                onewxp10 += wxplowdin_prod10 * np.einsum("ij,ji->",
                                                         hcore1_ket,
                                                         xpw_p10)
            else:
                onewxp10 += wxplowdin_prod10 * np.einsum("ij,ji->",
                                                         hcore0,
                                                         xpw_p10)

            #Term E
            xp_g01 = x_g0
            xp_g01[:,p] = x_g1[:,p] #Replace pth x_g0 column with x_g1
            wxplambda01, w_g0_t, xp_g01_t = lowdin_pairing(w_g0, xp_g01, mol,
                                                          nelec,
                                                          complexsymmetric
                                                          )
            wxplowdin_prod01 = lowdin_prod(wxplambda01, [m])
            xpw_p01 = get_xw_p0(w_g0_t, xp_g01_t, m, complexsymmetric)

            onewxp01 += wxplowdin_prod01 * np.einsum("ij,ji->", hcore0,
                                                     xpw_p01)

        onewx1 += onewpx01 + onewpx10 + onewxp10 + onewxp01

    #Term C
    wxlambda0 = lowdin_pairing(w_g0, x_g0, mol, nelec, complexsymmetric)
    onewx010 = 0

    for m in range(nelec):

        lowdin_prod0 = lowdin_prod(wxlambda0, [m])
        xw_p0 = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)

        onewx010 += lowdin_prod0 * np.einsum("ij,ji->", hcore1_op, xw_p0)

    onewx1 += onewx010

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
            xw_p0_m = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)
            xw_p0_n = get_xw_p0(w_g0_t, x_g0_t, n, complexsymmetric)

            twowx0 += lowdin_prod0 * (np.einsum("ijkl,ji,lk->",
                                                j0,xw_p0_m,xw_p0_n)
                                      - np.einsum("ijkl,li,jk->",
                                                  j0,xw_p0_m,xw_p0_n)) * 0.5

    return twowx0


def get_twowx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
               complexsymmetric: bool):

    r"""Calculates the first order two electron contribution to the
    hamiltonian element between determinants w and x using the new
    formulation. Terms A-E here are defined as in the theory. There
    is no term C contribution for the two electron operator.

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

    sao0 = mol.intor("int1e_ovlp")
    s1_bra, s1_ket = get_sao1_partial(mol, atom, coord)
    j0 = get_j0(mol)
    j1_bra0, j1_bra1, j1_ket0, j1_ket1 = get_j1_partial(mol, atom, coord)

    twowx1 = 0

            lowdin_prod0 = lowdin_prod(wxlambda0, [m,n])
            xw_p0_m = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)
            xw_p0_n = get_xw_p0(w_g0_t, x_g0_t, n, complexsymmetric)

            twowx0 += lowdin_prod0 * (np.einsum("ijkl,ji,lk->",
                                                j0,xw_p0_m,xw_p0_n)
                                      - np.einsum("ijkl,li,jk->",
                                                  j0,xw_p0_m,xw_p0_n)) * 0.5

    for p in range(nelec):

        twowpx01 = 0 #Term A
        twowpx10 = 0 #Term B
        twowxp10 = 0 #Term D
        twowxp01 = 0 #Term E

        for m in range(nelec):
            for n in range(nelec):

                if n == m:
                    continue

                #Term A
                wp_g01 = w_g0
                wp_g01[:,p] = w_g1[:,p] #Replace pth w_g0 column with w_g1
                wpxlambda01,wp_g01_t,x_g0_t = lowdin_pairing(wp_g01, x_g0,
                                                             mol, nelec,
                                                             complexsymmetric)
                wpxlowdin_prod01 = lowdin_prod(wpxlambda01, [m,n])
                xwp_p01_m = get_xw_p0(wp_g01_t, x_g0_t, m, complexsymmetric)
                xwp_p01_n = get_xw_p0(wp_g01_t, x_g0_t, n, complexsymmetric)

                twowpx01 += (wpxlowdin_prod01*0.5*
                             (np.einsum("ijkl,ji,lk->",
                                        j0,xwp_p01_m,xwp_p01_n)
                              - np.einsum("ijkl,li,jk->",
                                          j0,xwp_p01_m,xwp_p01_n)))

                #Term B
                wpxsao10 = sao0
                wpxsao10[p,:] = s1_bra[p,:] #Replace pth sao row with (1|0)
                wpxlambda10, w_g0_t, x_g0_t = lowdin_pairing(w_g0, x_g0,
                                                            wpxsao10, mol,
                                                            nelec,
                                                            complexsymmetric)
                wpxlowdin_prod10 = lowdin_prod(wpxlambda10, [m,n])
                xwp_p10_m = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)
                xwp_p10_n = get_xw_p0(w_g0_t, x_g0_t, n, complexsymmetric)

                if m == p:
                    twowpx10 += (wpxlowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_bra0,xwp_p10_m,xwp_p10_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_bra0,xwp_p10_m,xwp_p10_n)))
                elif n == p:
                    twowpx10 += (wpxlowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_bra1,xwp_p10_m,xwp_p10_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_bra1,xwp_p10_m,xwp_p10_n)))
                else:
                    twowpx10 += (wpxlowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j0,xwp_p10_m,xwp_p10_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j0,xwp_p10_m,xwp_p10_n)))

                #Term D
                wxpsao10 = sao0
                wxpsao10[:,p] = s1_ket[:,p] #Replace pth sao column with (0|1)
                wxplambda10, w_g0_t, x_g0_t = lowdin_pairing(w_g0, x_g0,
                                                            wxpsao10, mol,
                                                            nelec,
                                                            complexsymmetric)
                wxplowdin_prod10 = lowdin_prod(wxplambda10, [m,n])
                xpw_p10_m = get_xw_p0(w_g0_t, x_g0_t, m, complexsymmetric)
                xpw_p10_n = get_xw_p0(w_g0_t, x_g0_t, n, complexsymmetric)

                if m == p:
                    twowxp10 += (wxplowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_ket0,xpw_p10_m,xpw_p10_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_ket0,xpw_p10_m,xpw_p10_n)))
                elif n == p:
                    twowxp10 += (wxplowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j1_ket1,xpw_p10_m,xpw_p10_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j1_ket1,xpw_p10_m,xpw_p10_n)))
                else:
                    twowxp10 += (wxplowdin_prod10*0.5*
                                 (np.einsum("ijkl,ji,lk->",
                                            j0,xpw_p10_m,xpw_p10_n)
                                  - np.einsum("ijkl,li,jk->",
                                              j0,xpw_p10_m,xpw_p10_n)))

                #Term E
                xp_g01 = x_g0
                xp_g01[:,p] = x_g1[:,p] #Replace pth w_g0 column with w_g1
                wxplambda01,w_g0_t,xp_g01_t = lowdin_pairing(w_g0, xp_g01,
                                                             mol, nelec,
                                                             complexsymmetric)
                wxplowdin_prod01 = lowdin_prod(wxplambda01, [m,n])
                xpw_p01_m = get_xw_p0(w_g0_t, xp_g01_t, m, complexsymmetric)
                xpw_p01_n = get_xw_p0(w_g0_t, xp_g01_t, n, complexsymmetric)

                twowxp01 += (wxplowdin_prod01*0.5*
                             (np.einsum("ijkl,ji,lk->",
                                        j0,xpw_p01_m,xpw_p01_n)
                              - np.einsum("ijkl,li,jk->",
                                          j0,xpw_p01_m,xpw_p01_n)))

        twowx1 += twowpx01 + twowpx10 + twowxp10 + twowxp01

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
    swx0 = lowdin_prod(wxlambda0, [])
    e0_nuc = get_e0_nuc(mol)

    nucwx0 = swx0 * e0_nuc

    return nucwx0


def get_nucwx1(mol, atom, coord, wxlambda0, wxlambda1, nelec,
               complexsymmetric: bool):

    r"""Calculates the first order nuclear repulsion contribution to the
    hamiltonian element between determinants w and x.

    .. math::

            \frac{\partial}{\partial X_A}\left[\langle\prescript{w}{}\Psi|
            \prescript{x}{}\Psi\rangle\sum\limits_{A>B}^N\frac{Z_AZ_B}{R_{AB}}
            \right]
            = \frac{\partial S_{wx}}{\partial X_A}\sum\limits_{A>B}^N
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
    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param wxlambda1: Diagonal matrix of first order Löwdin singular values
            for the wth and xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Value for nuclear repulsion contribution to hamiltonian element
            derivative.
    """

    swx0 = lowdin_prod(wxlambda0, [])
    swx1 = get_swx1(wxlambda0, wxlambda1, nelec)

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

            h0mat[w,x] = onewx0 + twowx0 + nucwx0

            # print(w, x)
            # print("two electron:", twowx0)

    return h0mat


def get_h1mat(mol, atom, coord, g0_list, nelec, complexsymmetric: bool):

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
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Matrix of hamiltonian element derivatives.
    """
    nnoci = len(g0_list)
    h1mat = np.zeros((nnoci,nnoci))
    g1_list = get_g1_list(mol, atom, coord, g0_list, nelec, complexsymmetric)

    for w in range(nnoci):
        for x in range(nnoci):

            w_g0 = g0_list[w]
            x_g0 = g0_list[x]
            w_g1 = g1_list[w]
            x_g1 = g1_list[x]

            onewx1 = get_onewx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
                                complexsymmetric)
            twowx1 = get_wx1(mol, atom, coord, w_g0, x_g0, w_g1, x_g1, nelec,
                                complexsymmetric)
            nucwx1 = get_nucwx1(mol, atom, coord, wxlambda0, wxlambda1, nelec,
                                complexsymmetric)

            h1mat[w,x] = onewx1 + twowx1 + nucwx1

    return h1mat
