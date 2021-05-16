from pyscf import gto, scf, grad
import time
import numpy as np
import scipy
from .zeroth_order_ghf import rhf_to_ghf, get_p0, get_hcore0, get_pi0, get_f0,\
get_e0_nuc, get_e0_elec

def get_s1(mol, atom, coord):

    r"""Calculates first order pertubation to the orbital overlap matrix

    .. math::

        \mathbf{S}^{(1)}_{\mu\nu}
        = \left(\frac{\partial\phi_{\mu}}{\partial a}\bigg|\phi_{\nu}\right)
        + \left(\phi_{\mu}\bigg|\frac{\partial\phi_{\nu}}{\partial a}\right)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: First order overlap matrix.
    """

    s_py = -mol.intor("int1e_ipovlp")[coord]
    #minus sign due to pyscf definition
    s1 = np.zeros_like(s_py)

    for i in range(s_py.shape[1]):

        lambda_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(s_py.shape[1]):

            lambda_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            s1[i][j] += s_py[i][j]*lambda_i+s_py[j][i]*lambda_j

    omega = np.identity(2)
    s1 = np.kron(omega, s1)
    np.set_printoptions(precision=3)

    return s1


def get_p1(g0, g1, complexsymmetric, nelec):

    r"""Calculates the first order density matrix from the zeroth and first
    order coefficient matrices. It is defined by (only over occupied MOs):

    .. math::

        \mathbf{P^{(1)}} =
        \mathbf{G^{(0)}G^{(1)\dagger\diamond}}
        + \mathbf{G^{(1)}G^{(0)\dagger\diamond}}

    :param g0: zeroth order GHF coefficient matrix.
    :param g1: First order GHF coefficient matrix.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.
    :param nelec: Number of electrons in the molecule, determines the number
            of occupied orbitals.

    :returns: The first order density matrix.
    """

    if not complexsymmetric:
        p1 = (np.dot(g0[:,0:nelec], g1[:,0:nelec].T.conj())
              + np.dot(g1[:,0:nelec], g0[:,0:nelec].T.conj()))
    else:
        p1 = (np.dot(g0[:,0:nelec], g1[:,0:nelec].T)
              + np.dot(g1[:,0:nelec], g0[:,0:nelec].T))


    return p1


def get_hcore1(mol, atom, coord):

    r"""Calculates the first order core hamiltonian matrix.
    Each element is given by:

    .. math::

        \left(\mathbf{H_{core}^{(1)}}\right)_{\mu\nu}
        = \left(\frac{\partial\phi_{\mu}}{\partial a}\left|
        \mathbf{\hat{H}_{core}}\right|\phi_{\nu}\right)
        + \left(\phi_{\mu}\left|\frac{\partial\mathbf{\hat{H}_{core}}}
        {\partial a}\right|\phi_{\nu}\right)
        + \left(\phi_{\mu}\left|\mathbf{\hat{H}_{core}}\right|
        \frac{\partial\phi_{\nu}}{\partial a}\right)

    (Note that :math:'a' is a particular specified pertubation, e.g movement
    in the x direction of atom 1)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: First order core hamiltonian matrix.
    """

    mf = scf.RHF(mol)
    g = grad.rhf.Gradients(mf)

    hcore1 = g.hcore_generator(mol)(atom)[coord]

    omega = np.identity(2)
    hcore1 = np.kron(omega, hcore1)

    return hcore1


def get_pi1(mol, atom, coord):

    r"""Calculates the 4 dimensional first order pi tensor by digesting the
    of 2 electron integrals given by PySCF.
    Symmetry of the 2 electron integrals is manipulated to digest the PySCF
    tensor, in which the first MO of each 2 electron integral has been
    differentiated.
    Each element is given by:

    .. math::

       \mathbf{\Pi_{\delta'\mu',\epsilon'\nu',\delta\mu,\epsilon\nu}^{(1)}}
       = \mathbf{\Omega_{\delta'\delta}\Omega_{\epsilon'\epsilon}}
       \left(\mu'\mu|\nu'\nu\right)^{(1)}
       -\mathbf{\Omega_{\delta'\epsilon}\Omega_{\epsilon'\delta}}
       \left(\mu'\nu|\nu'\mu\right)^{(1)}

       \left(\mu'\mu|\nu'\nu\right)^{(1)}
       =\left(\frac{\partial\phi_{\mu'}}{\partial a}\phi_{\mu}|
       \phi_{\nu'}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\frac{\partial\phi_{\mu}}{\partial a}|
       \phi_{\nu'}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\phi_{\mu}|
       \frac{\partial\phi_{\nu'}}{\partial a}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\phi_{\mu}|
       \phi_{\nu'}\frac{\partial\phi_{\nu}}{\partial a}\right)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: First order 4 dimensional pi tensor.
    """

    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)

    pi_py = -mol.intor("int2e_ip1")[coord] #minus sign due to pyscf definition

    j1_spatial = np.zeros((pi_py.shape[0],pi_py.shape[0],pi_py.shape[0],
                           pi_py.shape[0]))

    for i in range(pi_py.shape[0]):

        lambda_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(pi_py.shape[0]):

            lambda_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            for k in range(pi_py.shape[0]):

                lambda_k = int(k in range(mol.aoslice_by_atom()[atom][2],
                                          mol.aoslice_by_atom()[atom][3]))

                for l in range(pi_py.shape[0]):

                    lambda_l = int(l in range(mol.aoslice_by_atom()[atom][2],
                                              mol.aoslice_by_atom()[atom][3]))

                    j1_spatial[i][j][k][l] += (pi_py[i][j][k][l] * lambda_i
                                               + pi_py[j][i][k][l] * lambda_j
                                               + pi_py[k][l][i][j] * lambda_k
                                               + pi_py[l][k][i][j] * lambda_l)

    j1_spatial = np.einsum("abcd->acbd", j1_spatial,
                           optimize='optimal') #convert to physicists
    j1 = np.kron(spin_j, j1_spatial)
    k1 = np.einsum("ijkl->ijlk", j1,
                   optimize='optimal') #physicists notation

    pi1 = j1 - k1

    return pi1


def get_j1(mol, atom, coord):

    r"""Calculates the 4 dimensional first order j tensor of coloumb integrals
    by digesting the of 2 electron integrals given by PySCF.
    Symmetry of the 2 electron integrals is manipulated to digest the PySCF
    tensor, in which the first MO of each 2 electron integral has been
    differentiated.
    Each element is given by:

    .. math::

       \mathbf{\Pi_{\delta'\mu',\epsilon'\nu',\delta\mu,\epsilon\nu}^{(1)}}
       = \mathbf{\Omega_{\delta'\delta}\Omega_{\epsilon'\epsilon}}
       \left(\mu'\mu|\nu'\nu\right)^{(1)}

       \left(\mu'\mu|\nu'\nu\right)^{(1)}
       =\left(\frac{\partial\phi_{\mu'}}{\partial a}\phi_{\mu}|
       \phi_{\nu'}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\frac{\partial\phi_{\mu}}{\partial a}|
       \phi_{\nu'}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\phi_{\mu}|
       \frac{\partial\phi_{\nu'}}{\partial a}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\phi_{\mu}|
       \phi_{\nu'}\frac{\partial\phi_{\nu}}{\partial a}\right)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: First order 4 dimensional j tensor.
    """

    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)

    twoe = -mol.intor("int2e_ip1")[coord] #minus sign due to pyscf definition

    j1_spatial = np.zeros((twoe.shape[0],twoe.shape[0],twoe.shape[0],
                           twoe.shape[0]))

    for i in range(twoe.shape[0]):

        lambda_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(twoe.shape[0]):

            lambda_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            for k in range(twoe.shape[0]):

                lambda_k = int(k in range(mol.aoslice_by_atom()[atom][2],
                                          mol.aoslice_by_atom()[atom][3]))

                for l in range(twoe.shape[0]):

                    lambda_l = int(l in range(mol.aoslice_by_atom()[atom][2],
                                              mol.aoslice_by_atom()[atom][3]))

                    j1_spatial[i][j][k][l] += (twoe[i][j][k][l] * lambda_i
                                               + twoe[j][i][k][l] * lambda_j
                                               + twoe[k][l][i][j] * lambda_k
                                               + twoe[l][k][i][j] * lambda_l)

    j1_spatial = np.einsum("abcd->acbd", j1_spatial,
                           optimize='optimal') #convert to physicists
    j1 = np.kron(spin_j, j1_spatial)

    return j1

def get_f1(pi0, p0, hcore1, pi1, p1):

    r"""Calculate the first order fock matrix, defined by

    .. math::

        \mathbf{F^{(1)}}=\mathbf{H_{core}^{(1)}}+\mathbf{\Pi^{(1)}}\cdot
        \mathbf{P^{(0)}}+\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(1)}}

    :param pi0: 4 dimensional zeroth order Pi tensor of 2 electron integrals
    :param p0: Zeroth order density matrix
    :param hcore1: First order core hamiltonian after particular pertubation
    :param pi1: 4 dimensional first order Pi tensor of differentiated 2
            electron integrals after particular pertubation
    :param p1: First order density matrix

    :returns: First order fock matrix.
    """

    f1_1 = hcore1
    f1_2 = np.einsum("ijkl,lj->ik", pi0, p1, optimize='optimal')
    f1_3 = np.einsum("ijkl,lj->ik", pi1, p0, optimize='optimal')

    f1 = f1_1 + f1_2 + f1_3

    return f1


def get_g1_x(f1_x, s1_x, eta0, nelec):

    r"""Calculates the transformed first order correction to the coefficient
    matrix G defined element wise by the following 2 equations:

    .. math:: \tilde{G}_{ij}^{(1)} = \frac{\tilde{F}_{ij}^{(1)}
              - \mathbf{\eta}_j^{(0)}\tilde{S}_{ij}^{(1)}}
              {\mathbf{\eta}_j^{(0)} - \mathbf{\eta}_i^{(0)}}

    .. math:: \tilde{G}_{jj}^{(1)} = -\frac{1}{2}\tilde{S}_{jj}^{(1)}

    :param f1_x: First order transformed fock matrix.
    :param s1_x: First order transformed overlap matrix.
    :param eta0: Vector of zeroth order energy eigenvalues.
    :param nelec: Number of electrons in the molecule, determines the number
            of occupied orbitals.

    :returns: transformed matrix G(1).
    """
    nbasis = f1_x.shape[1]
    nocc = nelec

    g1_x = np.zeros_like(f1_x)

    for j in range(nocc, nbasis):
        for i in range(nocc):

            delta_eta0 = eta0[j] - eta0[i]
            g1_x[i,j] = (f1_x[i,j] - eta0[j]*s1_x[i,j])/delta_eta0
            # print(i, j)
            # print("f1_x[i,j]:\n", f1_x[i,j])
            # print("occ orbital energy:", eta0[j])
            # print("vir orbital energy:", eta0[i])
            # print("delta_eta0:\n", delta_eta0)


    for j in range(nbasis):

        g1_x[j,j] = -0.5*s1_x[j,j]

    return g1_x


def g1_iteration(complexsymmetric: bool, mol, atom, coord, nelec,
                 g0_ghf = None):

    r"""Calculates the first order coefficient matrix self consistently given
    that :math:'\mathbf{G^{(1)}}' and :math:'\mathbf{F^{(1)}}' depend on one
    another.

    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'. Used here
            when transforming quantities using the X matrix.
    :param mol: Molecule class as defined by PySCF.
    :param g0: Matrix of zeroth order molecular orbital coefficients
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param g0_ghf: An optional argument for which the user can specify a g0
            zeroth order molecular coefficient matrix in RHF. By default this
            is set to None and g0 in RHF will be obtained from PySCF.

    :returns: The converged first order coefficient matrix.
    """

    if g0_ghf is None:
        m = scf.RHF(mol)
        m.verbose = 0
        m.kernel()
        g0_rhf = m.mo_coeff
        g0 = rhf_to_ghf(g0_rhf, nelec)
    else:
        g0 = g0_ghf

    x = g0
    s0 = mol.intor("int1e_ovlp")
    s0 = np.kron(np.identity(2), s0)
    s1 = get_s1(mol, atom, coord)
    p0 = get_p0(g0, complexsymmetric, nelec)
    hcore0 = get_hcore0(mol)
    pi0 = get_pi0(mol)
    f0 = get_f0(hcore0, pi0, p0)

    hcore1 = get_hcore1(mol, atom, coord)
    pi1 = get_pi1(mol, atom, coord)

    if not complexsymmetric:

        p0_x = np.linalg.multi_dot([x.T.conj(), p0, x])
        s1_x = np.linalg.multi_dot([x.T.conj(), s1, x])
        f0_x = np.linalg.multi_dot([x.T.conj(), f0, x])
        pi0_x = np.einsum("pi,ijkl,jr,kq,sl->prqs",
                          x.T.conj(), pi0, x, x, x.T.conj(),
                          optimize = 'optimal')

        hcore1_x = np.linalg.multi_dot([x.T.conj(), hcore1, x])
        pi1_x = np.einsum("pi,ijkl,jr,kq,sl->prqs",
                          x.T.conj(), pi1, x, x, x.T.conj(),
                          optimize = 'optimal')

    else:

        p0_x = np.linalg.multi_dot([x.T, p0, x])
        s1_x = np.linalg.multi_dot([x.T, s1, x])
        f0_x = np.linalg.multi_dot([x.T, f0, x])
        pi0_x = np.einsum("pi,ijkl,jr,kq,sl->prqs",
                          x.T, pi0, x, x, x.T,
                          optimize = 'optimal')

        hcore1_x = np.linalg.multi_dot([x.T, hcore1, x])
        pi1_x = np.einsum("pi,ijkl,jr,kq,sl->prqs",
                          x.T, pi1, x, x, x.T,
                          optimize = 'optimal')


    eta0, g0_x = np.linalg.eig(f0_x)
    index = np.argsort(eta0)
    eta0 = eta0[index]
    g0_x = g0_x[:, index] #Order g0 columns according to eigenvalues

    # eta0sci, _ = scipy.linalg.eig(f0, s0)
    # with np.printoptions(precision=3):
    #     print("Before transforming:\n")
    #     print("g0:\n", g0)
    #     print("eta0\n", eta0)
    # with np.printoptions(precision=3):
    #     print("Before ordering:\n")
    #     print("g0_x:\n", g0_x)
    #     print("eta0_x\n", eta0_x)
    # with np.printoptions(precision=3):
    #     print("After ordering:\n")
    #     print("g0_x:\n", g0_x)

    g1_x_guess = np.zeros_like(g0)
    g1_x = g1_x_guess
    iter_num = 0
    delta_g1_x = 1

    while delta_g1_x > 1e-14:

        iter_num += 1
        p1_x = get_p1(g0_x, g1_x, complexsymmetric, nelec)
        f1_x = get_f1(pi0_x, p0_x, hcore1_x, pi1_x, p1_x)
        # print("iteration number:", iter_num)
        # print("g1_x max:", np.max(g1_x))
        # print("f1_x max:", np.max(f1_x))
        # assert np.max(f1_x) < 1e10

        g1_x_last = g1_x
        g1_x = get_g1_x(f1_x, s1_x, eta0, nelec)
        delta_g1_x = np.max(np.abs(g1_x - g1_x_last))

    g1 = np.dot(x, g1_x)
    

    # print("f0:\n", f0)
    # print("f0_x:\n", f0_x)
    # print("p0:\n", p0)
    # print("p0_x:\n", p0_x)
    # print("eta0:\n", eta0)
    # print("eta0sci:", eta0sci)

    return g1


def get_e1_nuc(mol, atom, coord):

    r"""Calculates the first order nuclear repulsion energy.
    This is given by the following expresison, where X_A is a particular
    cartesian coordinate of atom A:

    .. math::

            $$E^{(1)}_{nuc} = \frac{\partial E^{(0)}_{nuc}}{\partial X_A}=
            \sum\limits_{B \neq A}^N
            \left(X_B-X_A\right)\frac{Z_AZ_B}{R^3_{AB}}$$

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: The first order nuclear repulsion energy.
    """

    e1_nuc = 0
    a = atom

    for b in range(len(mol.atom_charges())):

        if b == atom:
            continue

        r_ab2 = np.dot(mol.atom_coord(a) - mol.atom_coord(b),
                       mol.atom_coord(a) - mol.atom_coord(b))
        r_ab = np.sqrt(r_ab2)
        r_ab3 = r_ab ** 3

        x_ab = mol.atom_coord(b)[coord] - mol.atom_coord(a)[coord]

        e1_nuc += x_ab * (mol.atom_charge(a) * mol.atom_charge(b)) / r_ab3

    return e1_nuc


def get_e1_elec(mol, g1, atom, coord, complexsymmetric: bool, nelec,
                g0_ghf = None):

    r"""Calculates the first order electronic energy.
    Defined as follows:

    .. math::

        E^{(1)}_{elec} = Tr\left(\mathbf{F'}^{(0)}\mathbf{P}^{(1)}
        + \mathbf{F'}^{(1)}\mathbf{P}^{(0)}\right)

    where

    .. math::

        \mathbf{F'}^{(0)} = \mathbf{H_{core}^{(0)}}
        + \frac{1}{2}\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(0)}}

    and

    .. math::

        mathbf{F'}^{(1)} = \mathbf{H_{core}^{(1)}}
        + \frac{1}{2}\mathbf{\Pi^{(1)}}\cdot\mathbf{P^{(0)}}
        + \frac{1}{2}\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(1)}}

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param g0: The zeroth order coefficient matrix.
    :param g1: The first order coefficient matrix.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.

    :returns: The first order electronic energy
    """

    if g0_ghf is None:
        m = scf.RHF(mol)
        m.verbose = 0
        m.kernel()
        g0_rhf = m.mo_coeff
        g0 = rhf_to_ghf(g0_rhf, nelec)
    else:
        g0 = g0_ghf

    p0 = get_p0(g0, complexsymmetric, nelec)
    p1 = get_p1(g0, g1, complexsymmetric, nelec)

    hcore0 = get_hcore0(mol)
    pi0 = get_pi0(mol)
    hcore1 = get_hcore1(mol, atom, coord)
    pi1 = get_pi1(mol, atom, coord)

    f0_prime_1e = hcore0
    f1_prime_1e = hcore1

    f0_prime_2e = 0.5 * np.einsum("ijkl,jl->ik", pi0, p0)
    f1_prime_2e = (0.5 * np.einsum("ijkl,jl->ik", pi1, p0)
                   + 0.5 * np.einsum("ijkl,jl->ik", pi0, p1))

    e1_elec_1e = (np.einsum("ij,ji->",f0_prime_1e, p1)
                  + np.einsum("ij,ji->",f1_prime_1e, p0))
    e1_elec_2e = (np.einsum("ij,ji->",f0_prime_2e, p1)
                  + np.einsum("ij,ji->",f1_prime_2e, p0))

    e1_elec = e1_elec_1e + e1_elec_2e

    return e1_elec


def get_e1_scf(mol, atom, coord, nelec, complexsymmetric, g0_ghf=None):

    if g0_ghf is None:

        g1 = g1_iteration(complexsymmetric, mol, atom, coord, nelec)
        e1_elec = get_e1_elec(mol, g1, atom, coord, complexsymmetric, nelec)

    else:

        g1 = g1_iteration(complexsymmetric, mol, atom, coord, nelec, g0_ghf)
        e1_elec = get_e1_elec(mol, g1, atom, coord, complexsymmetric, nelec,
                              g0_ghf)

    e1_nuc = get_e1_nuc(mol, atom, coord)

    return e1_elec + e1_nuc


def write_e1_mat(mol, nelec, complexsymmetric, g0_ghf = None):

    r"""Writes matrix of ghf energy derivatives for each atom and coordinate.

    :param mol: PySCF molecule object.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.
    :param g0_ghf: An optional argument for which the user can specify a g0
            zeroth order molecular coefficient matrix in RHF. By default this
            is set to None and g0 in RHF will be obtained from PySCF.

    :returns: natom x 3 matrix of ghf energy derivatives.
    """

    e1_mat = np.zeros((mol.natm, 3))
    for i in range(mol.natm):
        for j in range(3):

            if g0_ghf is None:
                g1 = g1_iteration(complexsymmetric, mol, i, j, nelec)
                e1_elec = get_e1_elec(mol, g1, i, j, complexsymmetric, nelec)
            else:
                g1 = g1_iteration(complexsymmetric, mol, i, j, nelec,
                                  g0_ghf)
                e1_elec = get_e1_elec(mol, g1, i, j, complexsymmetric, nelec,
                                      g0_ghf)
            e1_nuc = get_e1_nuc(mol, i, j)
            e1 = e1_elec + e1_nuc

            e1_mat[i,j] += e1

    print("-------------- First order GHF energies --------------")
    print('Atom     x                y                z')
    for i, n in enumerate(range(mol.natm)):
        print('%d %s  %15.10f  %15.10f  %15.10f' %
              (n, mol.atom_symbol(n), e1_mat[i,0], e1_mat[i,1], e1_mat[i,2]))
    print("------------------------------------------------------")

    return e1_mat


def write_e1_single(mol, nelec, atom, coord, complexsymmetric, g0_ghf = None):

    r"""Gives energy derivative for a specific pertubation defined by an atom
    and coordinate.

    :param mol: PySCF molecule object.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.
    :param g0_ghf: An optional argument for which the user can specify a g0
            zeroth order molecular coefficient matrix in RHF. By default this
            is set to None and g0 in RHF will be obtained from PySCF.

    :returns: single scalar for the energy derivative of a specific
            pertubation.
    """

    if coord == 0:
        pert = "x"
    elif coord == 1:
        pert = "y"
    elif coord == 2:
        pert = "z"

    if g0_ghf is None:
        g1 = g1_iteration(complexsymmetric, mol, atom, coord, nelec)
        e1_elec = get_e1_elec(mol, g1, atom, coord, complexsymmetric, nelec)
    else:
        g1 = g1_iteration(complexsymmetric, mol, atom, coord, nelec, g0_ghf)
        e1_elec = get_e1_elec(mol, g1, atom, coord, complexsymmetric, nelec,
                              g0_ghf)

    e1_nuc = get_e1_nuc(mol, atom, coord)
    e1 = e1_elec + e1_nuc

    print("The molecule has atoms:")
    for i, n in enumerate(range(mol.natm)):
        print(n, mol.atom_pure_symbol(i), "at coordinates", mol.atom_coord(i))

    print("\nThe", mol.atom_pure_symbol(atom), "atom with index", atom,
          "at coordinates", mol.atom_coord(atom),
          "is perturbed in the positive", pert, "direction\n")

    print("########################")
    print("First order electronic energy:\n", e1_elec)
    print("First order nuclear repulsion energy:\n",
          get_e1_nuc(mol,atom,coord))
    print("Total first order energy:\n", get_e1_nuc(mol,atom,coord) + e1_elec)
    print("########################\n")

    return e1
