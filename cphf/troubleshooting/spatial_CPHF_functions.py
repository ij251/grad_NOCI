from pyscf import gto, scf, grad
import numpy as np

def make_ghf(g0_rhf, nelec):

    r"""Calculates the GHF MO coefficient matrix from the RHF, organised in
    blocks of occupied then virtual orbitals.

    :param g0_rhf: RHF MO coefficient matrix.
    :param nelec: Number of electrons in the molecule, determines the number
            of occupied orbitals.

    :returns: GHF MO coefficient matrix.
    """

    nbasis = g0_rhf.shape[1]

    if nelec % 2 == 0:
        nocc = int(nelec/2)
    else:
        nocc = int((nelec+1)/2)

    g0_ghf = np.block([[g0_rhf[:, 0:nocc],
                        np.zeros_like(g0_rhf[:, 0:nocc]),
                        g0_rhf[:, nocc:nbasis],
                        np.zeros_like(g0_rhf[:, nocc:nbasis])],
                       [np.zeros_like(g0_rhf[:, 0:nocc]),
                        g0_rhf[:, 0:nocc],
                        np.zeros_like(g0_rhf[:, nocc:nbasis]),
                        g0_rhf[:, nocc:nbasis]]])

    return g0_ghf

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

    s0 = mol.intor("int1e_ovlp")
    onee = -mol.intor("int1e_ipovlp") #minus from pyscf definition
    s1 = np.zeros_like(s0)

    for i in range(s0.shape[1]):

        lambda_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(s0.shape[1]):

            lambda_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            s1[i][j] += onee[coord][i][j]*lambda_i+onee[coord][j][i]*lambda_j

    # omega = np.identity(2)
    # s1 = np.kron(omega, s1)

    return s1


def get_x_lowdin(mol, thresh: float = 1e-14):

    r"""Calculates canonical basis orthogonalisation matrix x, defined by:

    .. math::

        \mathbf{X}=\mathbf{Us^{-\frac{1}{2}}}}

    where U is the matrix of eigenvectors of s_ao, and
    :math:'s^{-\frac{1}{2}}}' is the diagonal matrix of inverse square root
    eigenvalues of s_ao.

    :param s_ao: atomic orbital overlap matrix
    :param thresh: Threshold to consider an eigenvalue of the AO overlap
            as zero.

    :returns: the orthogonalisation matrix x
    """

    omega = np.identity(2)
    overlap_s = mol.intor('int1e_ovlp')
    # overlap_s = np.kron(omega, spatial_overlap_s)

    assert np.allclose(overlap_s, overlap_s.T.conj(), rtol=0, atol=thresh)
    s_eig, mat_u = np.linalg.eigh(overlap_s)
    overlap_indices = np.where(np.abs(s_eig) > thresh)[0]
    s_eig = s_eig[overlap_indices]
    mat_u = mat_u[:, overlap_indices]
    s_s = np.diag(1.0/s_eig)**0.5
    mat_x = np.dot(mat_u, s_s)

    return mat_x


def get_p0(g0, complexsymmetric: bool, nelec):

    r"""Calculates the zeroth order density matrix from the zeroth order
    coefficient matrix. It is defined by (only over occupied MOs):

    .. math::

        \mathbf{P^{(0)}}=\mathbf{G^{(0)}G^{(0)\dagger\diamond}}

    :param g0: zeroth order GHF coefficient matrix.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The zeroth order density matrix.
    """

    if not complexsymmetric:
        p0 = np.dot(g0[:,0:nelec], g0[:,0:nelec].T.conj())
    else:
        p0 = np.dot(g0[:,0:nelec], g0[:,0:nelec].T)

    return p0


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


def get_hcore0_spatial(mol):

    r"""Calculates The zeroth order core hamiltonian.
    Each element is given by:

    .. math::

        \left(\mathbf{H_{core}^{(0)}}\right)_{\mu\nu}
        =\left(\phi_{\mu}\left|\mathbf{\hat{H}_{core}}\right|\phi_{\nu}\right)

    :param mol: The class for a molecule as defined by PySCF.

    :returns: The zeroth order core hamiltonian matrix.
    """

    hcore0_spatial = (mol.intor('int1e_nuc')
                      + mol.intor('int1e_kin'))

    # omega = np.identity(2)
    # hcore0 =  np.kron(omega, hcore0)

    return hcore0_spatial


def get_pi0_spatial(mol):

    r"""Calculate the 4 dimensional zeroth order Pi tensor.
    Each element is given by:

    .. math::

        \mathbf{\Pi_{\delta'\mu',\epsilon'\nu',\delta\mu,\epsilon\nu}^{(0)}}
        = \mathbf{\Omega_{\delta'\delta}\Omega_{\epsilon'\epsilon}}
          \left(\mu'\mu|\nu'\nu\right)
        - \mathbf{\Omega_{\delta'\epsilon}\Omega_{\epsilon'\delta}}
          \left(\mu'\nu|\nu'\mu\right)

    :param mol: The class for a molecule as defined by PySCF.

    :returns: The zeroth order Pi tensor.
    """

    spatial_j = mol.intor('int2e')
    phys_spatial_j = np.einsum("abcd->acbd", spatial_j)
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    # j = np.kron(spin_j, phys_spatial_j)
    j = phys_spatial_j
    k = np.einsum("ijkl->ijlk", j)
    pi0_spatial = 2*j - k

    return pi0_spatial


def get_f0_spatial(hcore0, pi0, p0):

    r"""Calculates the zeroth order fock matrix, defined by:

    .. math::

        \mathbf{F^{(0)}}
        =\mathbf{H_{core}^{(0)}}
        +\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(0)}}

    :param hcore0: Zeroth order core hamiltonian matrix.
    :param pi0: Zeroth order 4 dimensional Pi tensor.
    :param p0: Zeroth order density matrix.

    :returns: The zeroth order fock matrix
    """

    f0_1e = hcore0
    f0_2e = np.einsum("ijkl,jl->ik", pi0, p0)
    f0_spatial = f0_1e + f0_2e

    return f0_spatial


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

    # omega = np.identity(2)
    # hcore1 = np.kron(omega, hcore1)

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

    twoe = -mol.intor("int2e_ip1")[coord] #minus sign from pyscf definition

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

    #j1_spatial = np.einsum("abcd->acbd", j1_spatial) #convert to physicists
    # j1 = np.kron(spin_j, j1_spatial)
    j1 = j1_spatial
    # k1 = np.einsum("ijkl->ijlk", j1) #physicists notation
    k1 = np.einsum("ijkl->ikjl", j1) #chemists notation

    pi1 = j1# - k1
    # print("twoe:\n", twoe)
    # print("pi1:\n", pi1)

    return pi1


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
    f1_2 = np.einsum("ijkl,lj->ik", pi0, p1)
    f1_3 = np.einsum("ijkl,lj->ik", pi1, p0)

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

    for i in range(nocc, nbasis):
        for j in range(nocc):
            delta_eta0 = eta0[j] - eta0[i]

            g1_x[i,j] = (f1_x[i,j] - eta0[j]*s1_x[i,j])/delta_eta0

            "printing"
            # print(i,j)
            # print(delta_eta0)
            # print(g1_x[i,j])

    for j in range(nbasis):

        g1_x[j,j] = -0.5*s1_x[j,j]

    # print(g1_x)
    return g1_x


def g1_iteration(complexsymmetric: bool, mol, g0, x, atom, coord, nelec,
                 g1_x_guess):

    r"""Calculates the first order coefficient matrix self consistently given
    that :math:'\mathbf{G^{(1)}}' and :math:'\mathbf{F^{(1)}}' depend on one
    another.

    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.
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
    :param p1_guess: An initial guess for the first order coefficient matrix,
            a matrix of zeros seems to work well for now, perhaps due to the
            pertubation being necessarily small, and other guesses converge to
            the same matrix.

    :returns: The converged first order coefficient matrix.
    """

    s1 = get_s1(mol, atom, coord)
    s1_x = np.linalg.multi_dot([x.T.conj(), s1, x])

    g0_x = np.dot(np.linalg.inv(x), g0)
    p0 = get_p0(g0, complexsymmetric, nelec)
    p0_x = np.linalg.multi_dot([x.T.conj(), p0, x])
    hcore0 = get_hcore0(mol)
    pi0 = get_pi0(mol)
    pi0_x = np.einsum("pi,ijkl,jr,kq,sl->prqs",
                      x.T.conj(), pi0, x, x, x.T.conj())
    f0 = get_f0(hcore0, pi0, p0)
    f0_x = np.linalg.multi_dot([x.T.conj(), f0, x])
    eta0 = np.linalg.eig(f0_x)[0]
    index = np.argsort(eta0)
    eta0 = eta0[index]

    hcore1 = get_hcore1(mol, atom, coord)
    hcore1_x = np.linalg.multi_dot([x.T.conj(), hcore1, x])
    pi1 = get_pi1(mol, atom, coord)
    pi1_x = np.einsum("pi,ijkl,jr,kq,sl->prqs",
                      x.T.conj(), pi1, x, x, x.T.conj())

    g1_x = g1_x_guess
    iter_num = 0
    delta_g1_x = 1

    while delta_g1_x > 1e-10:

        iter_num += 1
        p1_x = get_p1(g0_x, g1_x, complexsymmetric, nelec)
        f1_x = get_f1(pi0_x, p0_x, hcore1_x, pi1_x, p1_x)

        g1_x_last = g1_x
        g1_x = get_g1_x(f1_x, s1_x, eta0, nelec)
        delta_g1_x = np.max(np.abs(g1_x - g1_x_last))

    # print("f1_x:\n", f1_x)
    # print("s1_x:\n", s1_x)
    # print("g1_x:\n", g1_x)
    # print("Number of iterations:\n", iter_num)
    g1 = np.dot(x, g1_x)

    return g1


def get_e0_nuc(mol):

    r"""Calculates the zeroth order nuclear repulsion energy.
    This is given by the following expression, where N is the total number of
    nuclei in the system, A and B are nuclear indices, Z is the atomic number,
    and R_{AB} is the distance between nucei A and B:

    .. math::

        E^{(0)}_{nuc} = \sum\limits_{A>B}^N\frac{Z_AZ_B}{R_{AB}}

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.

    :returns: The zeroth order nuclear repulsion energy.
    """

    e0_nuc = 0

    for a in range(len(mol.atom_charges())):
        for b in range(a+1, len(mol.atom_charges())):

            r_ab2 = np.dot(mol.atom_coord(a) - mol.atom_coord(b),
                           mol.atom_coord(a) - mol.atom_coord(b))
            r_ab = np.sqrt(r_ab2)

            e0_nuc += (mol.atom_charge(a) * mol.atom_charge(b)) / r_ab

    return e0_nuc


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


def get_e0_elec(mol, g0):

    r"""Calculates the zeroth order electronic energy.
    This is a contraction of the zeroth order density matrix with zeroth order
    core hamiltonian and pi tensors:

    .. math::

        E^{(0)}_{elec} = Tr\left(\mathbf{F'}^{(0)}\mathbf{P}^{(0)}\right)

    where

    .. math::

        \mathbf{F'}^{(0)} = \mathbf{H_{core}^{(0)}}
        + \frac{1}{2}\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(0)}}

    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param g0: The zeroth order coefficient matrix.

    :returns: The zeroth order Hartree Fock electronic energy
    """

    p0 = get_p0(g0, complexsymmetric, nelec)
    hcore0 = get_hcore0(mol)
    pi0 = get_pi0(mol)

    f0_prime_1e = hcore0
    f0_prime_2e = 0.5 * np.einsum("ijkl,jl->ik", pi0, p0)

    e0_elec_1e = np.trace(np.dot(f0_prime_1e, p0))
    e0_elec_2e = np.trace(np.dot(f0_prime_2e, p0))

    e0_elec = e0_elec_1e + e0_elec_2e

    return e0_elec


def get_e1_elec(mol, g0, g1, atom, coord, complexsymmetric: bool, nelec):

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

    e1_elec_1e = (np.einsum("ij,ij->",f0_prime_1e, p1)
                  + np.einsum("ij,ij->",f1_prime_1e, p0))
    e1_elec_2e = (np.einsum("ij,ij->",f0_prime_2e, p1)
                  + np.einsum("ij,ij->",f1_prime_2e, p0))

    # print(np.einsum("ij,ij->",f0_prime_2e, p1))
    # print(np.einsum("ij,ij->",f1_prime_2e, p0))
    # assert False
    e1_elec = e1_elec_1e + e1_elec_2e

    print("########################")
    print("1 electron e1_elec:\n", e1_elec_1e)
    print("2 electron e1_elec:\n", e1_elec_2e)
    print("total e1_elec:\n", e1_elec)
    print("Nuclear repulsion e1_nuc:\n", get_e1_nuc(mol, atom, coord))
    print("########################")

    return e1_elec
