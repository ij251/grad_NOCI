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


def get_onewx0(mol, gw0_t, gx0_t, wxlambda0, nelec, complexsymmetric: bool):

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
    :param gw0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the wth determinant.
    :param gx0_t: The Löwdin transformed zeroth order molecular orbital
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
        xwp0 = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)

        onewx0 += lowdin_prod0 * np.einsum("ij,ji->", hcore0, xwp0)

    return onewx0


def get_onewx1(mol, atom, coord, gw0, gx0, gw1, gx1, nelec,
               complexsymmetric: bool):

    r"""Calculates the first order one electron contribution to the
    hamiltonian element between determinants w and x using the new
    formulation. Terms a-e here are defined as in the theory.

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param gw0: The zeroth order molecular orbital coefficient matrix of the
            wth determinant.
    :param gx0: The zeroth order molecular orbital coefficient matrix of the
            xth determinant.
    :param gw1: The first order molecular orbital coefficient matrix of the
            wth determinant.
    :param gx1: The first order molecular orbital coefficient matrix of the
            xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Value for one electron contribution to hamiltonian element
            derivative.
    """

    hcore0 = get_hcore0(mol)
    hcore1 = get_hcore1(mol, atom, coord)

    onewx1 = 0

    for m in range(nelec):
        for p in range(nelec):

                gwp01 = gw0
                gwp01[:,p] = gw1[:,p]
                wpxlambda,gwp01_t,gx_t = lowdin_pairing0(gwp01,gx0,mol,nelec,
                                                         complexsymmetric)

                a_p = get_onewx0(mol, gwp01_t, gx_t, wpxlambda, nelec,
                                 complexsymmetric)

                gxp01 = gx0
                gxp01[:,p] = gx1[:,p]
                wxplambda,gw_t,gxp01_t = lowdin_pairing0(gw0,gxp01,mol,nelec,
                                                         complexsymmetric)

                e_p = get_onewx0(mol, gw_t, gxp01_t, wxplambda, nelec,
                                 complexsymmetric))

                onewx1 += a_p + e_p

                #still need information about the operator derivative

    return onewx1


def get_twowx0(mol, gw0_t, gx0_t, wxlambda0, nelec, complexsymmetric: bool):

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
    :param gw0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the wth determinant.
    :param gx0_t: The Löwdin transformed zeroth order molecular orbital
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

            lowdin_prod0 = lowdin_prod(wxlambda0, [m,n])
            xwp0_m = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)
            xwp0_n = get_xwp0(gw0_t, gx0_t, n, complexsymmetric)

            twowx0 += lowdin_prod0 * (np.einsum("ijkl,ji,lk->",
                                                j0,xwp0_m,xwp0_n)
                                      - np.einsum("ijkl,li,jk->",
                                                  j0,xwp0_m,xwp0_n)) * 0.5

    return twowx0


def get_twowx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t, wxlambda0,
               wxlambda1, nelec, complexsymmetric: bool):

    r"""Calculates the first order two electron contribution to the
    hamiltonian element between determinants w and x.

    .. math::

            \frac{1}{2}\frac{\partial}{\partial X_A}\sum\limits_m^{N_e}
            \sum\limits_{n \not =m}^{N_e}\langle\prescript{w}{}\Psi
            |\hat{r}_{mn}^{-1}|\prescript{x}{}\Psi\rangle
            = \frac{1}{2}\frac{\partial (\delta\mu,\gamma\nu|\delta'\mu',\
            gamma'\nu')}{\partial X_A}\sum\limits_m^{N_e}\sum\limits_{n}^{N_e}
            \left(\prod\limits_{i\not =m,n}^{N_e}\prescript{wx}{}\lambda_{i}
            \right) \left[ \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}
            \prescript{xw}{}P_n^{\gamma'\nu', \delta'\mu'} -  \prescript{xw}{}
            P_m^{\gamma'\nu', \delta\mu} \prescript{xw}{}P_n^{\gamma\nu,
            \delta'\mu'} \right]
            + \frac{1}{2}(\delta\mu,\gamma\nu|\delta'\mu',\gamma'\nu')
            \sum\limits_m^{N_e}\sum\limits_{n}^{N_e} \left(
            \sum \limits_{j \not =m,n}^{N_e} \frac{\partial \prescript{wx}{}
            \lambda_j}{\partial X_A} \prod\limits_{i\not = j,m,n}^{N_e}
            \prescript{wx}{}\lambda_i \right) \left[ \prescript{xw}{}
            P_m^{\gamma\nu, \delta\mu} \prescript{xw}{}P_n^{\gamma'\nu',
            \delta'\mu'} - \prescript{xw}{}P_m^{\gamma'\nu', \delta\mu}
            \prescript{xw}{}P_n^{\gamma\nu, \delta'\mu'} \right]
            + \frac{1}{2}(\delta\mu,\gamma\nu|\delta'\mu',\gamma'\nu')
            \sum\limits_m^{N_e}\sum\limits_{n}^{N_e} \left(
            \prod\limits_{i\not =m,n}^{N_e}\prescript{wx}{}\lambda_{i}\right)
            \Bigg[ \frac{\partial \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}}
            {\partial X_A}\prescript{xw}{}P_n^{\gamma'\nu', \delta'\mu'}
            + \prescript{xw}{}P_m^{\gamma\nu, \delta\mu} \frac{\partial
            \prescript{xw}{}P_n^{\gamma'\nu', \delta'\mu'}}{\partial X_A}
            - \frac{\partial \prescript{xw}{}P_m^{\gamma'\nu', \delta\mu}}
            {\partial X_A}\prescript{xw}{}P_n^{\gamma\nu, \delta'\mu'}
            - \prescript{xw}{}P_m^{\gamma'\nu', \delta\mu} \frac{\partial
            \prescript{xw}{}P_n^{\gamma\nu, \delta'\mu'}}{\partial X_A}\Bigg]

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param gw0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the wth determinant.
    :param gx0_t: The Löwdin transformed zeroth order molecular orbital
            coefficient matrix of the xth determinant.
    :param gw1: The first order molecular orbital coefficient matrix of the
            wth determinant.
    :param gx1: The first order molecular orbital coefficient matrix of the
            xth determinant.
    :param wxlambda0: Diagonal matrix of Löwdin singular values for the wth
            and xth determinant.
    :param wxlambda1: Diagonal matrix of first order Löwdin singular values
            for the wth and xth determinant.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: Value for two electron contribution to hamiltonian element
            derivative.
    """

    j0 = get_j0(mol)
    j1 = get_j1(mol, atom, coord)

    twowx1 = 0
    for m in range(nelec):
        for n in range(nelec):


            lowdin_prod0 = lowdin_prod(wxlambda0, [m,n])
            lowdin_prod1 = (np.sum(wxlambda1[j,j]*lowdin_prod(wxlambda0,
                                                                   [j,m,n])
                                  for j in range(nelec))
                            - wxlambda1[m,m]*lowdin_prod(wxlambda0,[m])
                            - wxlambda1[n,n]*lowdin_prod(wxlambda0,[n]))
            xwp0_m = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)
            xwp1_m = get_xwp1(gw0_t, gx0_t, gw0_t, gx1_t, m, complexsymmetric)
            xwp0_n = get_xwp0(gw0_t, gx0_t, n, complexsymmetric)
            xwp1_n = get_xwp1(gw0_t, gx0_t, gw0_t, gx1_t, n, complexsymmetric)

            a = lowdin_prod0 * (np.einsum("ijkl,ji,lk->",j1,xwp0_m,xwp0_n)
                                - np.einsum("ijkl,li,jk->",j1,xwp0_m,xwp0_n))
            b = lowdin_prod1 * (np.einsum("ijkl,ji,lk->",j0,xwp0_m,xwp0_n)
                                - np.einsum("ijkl,li,jk->",j0,xwp0_m,xwp0_n))
            c = lowdin_prod0 * (np.einsum("ijkl,ji,lk->",j0,xwp1_m,xwp0_n)
                                + np.einsum("ijkl,ji,lk->",j0,xwp0_m,xwp1_n)
                                - np.einsum("ijkl,li,jk->",j0,xwp1_m,xwp0_n)
                                - np.einsum("ijkl,li,jk->",j0,xwp0_m,xwp1_n))

            twowx1 += 0.5 * (a + b + c)

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

            gw0 = g0_list[w]
            gx0 = g0_list[x]

            wxlambda0, gw0_t, gx0_t = lowdin_pairing0(gw0, gx0, mol, nelec,
                                                      complexsymmetric)

            onewx0 = get_onewx0(mol, gw0_t, gx0_t, wxlambda0, nelec,
                                complexsymmetric)
            twowx0 = get_twowx0(mol, gw0_t, gx0_t, wxlambda0, nelec,
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

            gw0 = g0_list[w]
            gx0 = g0_list[x]
            gw1 = g1_list[w]
            gx1 = g1_list[x]

            onewx1 = get_onewx1(mol, atom, coord, gw0, gx0, gw1, gx1, nelec,
                                complexsymmetric)
            twowx1 = get_wx1(mol, atom, coord, gw0, gx0, gw1, gx1, nelec,
                                complexsymmetric)
            nucwx1 = get_nucwx1(mol, atom, coord, wxlambda0, wxlambda1, nelec,
                                complexsymmetric)

            h1mat[w,x] = onewx1 + twowx1 + nucwx1

    return h1mat
