import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *


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
    # onewx0 = np.sum(lowdin_prod(wxlambda0, [m])
    #                 * np.einsum("ij,ji->",
    #                             hcore0,
    #                             get_xwp0(gw0_t, gx0_t, m, complexsymmetric))
    #                 for m in range(nelec))

    for m in range(nelec):

        lowdin_prod0 = lowdin_prod(wxlambda0, [m])
        xwp0 = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)

        onewx0 += lowdin_prod0 * np.einsum("ij,ji->", hcore0, xwp0)

    return onewx0


def get_onewx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t, wxlambda0,
               wxlambda1, nelec, complexsymmetric: bool):

    r"""Calculates the first order one electron contribution to the
    hamiltonian element between determinants w and x.

    .. math::

            \frac{\partial}{\partial X_A}\sum\limits_{m=1}^{N_e}\langle
            \prescript{w}{}\Psi|\hat{h}(\mathbf{r}_m)
            |\prescript{x}{}\Psi \rangle
            = \frac{\partial h_{\delta\mu, \gamma\nu}}{\partial X_A}
            \sum\limits_{m=1}^{N_e}\left(\prod\limits_{i\not =m}^{N_e}
            \prescript{wx}{}\lambda_{i}\right)
            \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}
            + h_{\delta\mu, \gamma\nu}\sum\limits_{m=1}^{N_e}
            \left(\sum \limits_{j \not =m}^{N_e} \frac{\partial \prescript{wx}
            {}\lambda_j}{\partial X_A} \prod\limits_{i\not = j,m}^{N_e}
            \prescript{wx}{}\lambda_i \right)\prescript{xw}{}
            P_m^{\gamma\nu, \delta\mu}
            + h_{\delta\mu, \gamma\nu}\sum\limits_{m=1}^{N_e}
            \left(\prod\limits_{i\not =m}^{N_e}\prescript{wx}{}\lambda_{i}
            \right)\frac{\partial \prescript{xw}{}P_m^{\gamma\nu, \delta\mu}}
            {\partial X_A}

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

    :returns: Value for one electron contribution to hamiltonian element
            derivative.
    """

    hcore0 = get_hcore0(mol)
    hcore1 = get_hcore1(mol, atom, coord)
    wxlambda1_diag = np.diag(wxlambda1)

    onewx1 = 0

    for m in range(nelec):

        lowdin_prod0 = lowdin_prod(wxlambda0, [m])
        lowdin_prod1 = (np.sum(wxlambda1_diag[j,j]*lowdin_prod(wxlambda0,
                                                               [j,m])
                              for j in range(nelec))
                        - wxlambda1_diag[m,m]*lowdin_prod(wxlambda0,[m]))
        xwp0 = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)
        xwp1 = get_xwp1(gw0_t, gx0_t, gw0_t, gx1_t, m, complexsymmetric)

        a = lowdin_prod0 * np.einsum("ij,ji->", hcore1, xwp0)
        b = lowdin_prod1 * np.einsum("ij,ji->", hcore0, xwp0)
        c = lowdin_prod0 * np.einsum("ij,ji->", hcore0, xwp1)

        onewx1 += a + b + c

    return onexw1


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
    wxlambda1_diag = np.diag(wxlambda1)

    twowx1 = 0
    for m in range(nelec):
        for n in range(nelec):


            lowdin_prod0 = lowdin_prod(wxlambda0, [m,n])
            lowdin_prod1 = (np.sum(wxlambda1_diag[j,j]*lowdin_prod(wxlambda0,
                                                                   [j,m,n])
                                  for j in range(nelec))
                            - wxlambda1_diag[m,m]*lowdin_prod(wxlambda0,[m])
                            - wxlambda1_diag[n,n]*lowdin_prod(wxlambda0,[n]))
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

    return twoxw1


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
    nnoci = g0_list.shape[0]
    h0mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            gw0 = g0_list[w]
            gx0 = g0_list[x]
            gw0_t, gx0_t = transform_g(gw0, gx0, mol, complexsymmetric)

            wxlambda0 = get_wxlambda0(gw0, gx0, mol, complexsymmetric)

            onewx0 = get_onewx0(mol, gw0_t, gx0_t, wxlambda0, nelec,
                                complexsymmetric)
            twowx0 = get_twowx0(mol, gw0_t, gx0_t, wxlambda0, nelec,
                                complexsymmetric)
            nucwx0 = get_nucwx0(mol, wxlambda0, complexsymmetric)

            h0mat[w,x] = onewx0 + twowx0 + nucwx0

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
    nnoci = g0_list.shape[0]
    h1mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            gw0 = g0_list[w]
            gx0 = g0_list[x]
            gw0_t, gx0_t = transform_g(gw0, gx0, mol, complexsymmetric)
            gw1_t = g1_iteration(complexsymmetric, mol, atom, coord, nelec,
                                 gw0_t)
            gx1_t = g1_iteration(complexsymmetric, mol, atom, coord, nelec,
                                 gx0_t)


            wxlambda0 = get_wxlambda0(gw0, gx0, mol, complexsymmetric)
            wxlambda1 = get_wxlambda1(gw0, gw1, gx0, gx1, mol, atom, coord,
                                      complexsymmetric)

            onewx1 = get_onewx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t,
                                wxlambda0, wxlambda1, nelec, complexsymmetric)
            twowx1 = get_twowx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t,
                                wxlambda0, wxlambda1, nelec, complexsymmetric)
            nucwx1 = get_nucwx1(mol, atom, coord, wxlambda0, wxlambda1, nelec,
                                complexsymmetric)

            h1mat[w,x] = onewx1 + twowx1 + nucwx1

    return h1mat
