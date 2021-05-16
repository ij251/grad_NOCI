import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import g1_iteration, write_e1_single, get_e1_elec, get_e1_nuc, get_e1_scf
from cphf.zeroth_order_ghf import uhf_to_ghf
from overlap_derivative import get_s1mat, get_g1_list
from hamiltonian_derivative import get_h1mat
from energy_derivative import get_e1_noci


def get_analytic_derivatives_h2(g_num, states_num, atoms, nelec, nalpha, nbeta,
                             atom, coord, complexsymmetric):


    for gi in range(g_num):

        mol = gto.Mole()
        mol.basis = 'sto-3g'
        mol.unit = 'bohr'
        mol.atom = []
        for atom in range(atoms):

            pos_vec = np.loadtxt(f"coords.g{gi}.atom{atom}.txt")

            mol.atom.extend([['H',(pos_vec[0], pos_vec[1], pos_vec[2])]])

        mol.build()
        g0_list = []

        for si in range(states_num):

            g0_s_a = np.loadtxt(f"mo.coeffs.g{gi}.s{si}.a.txt")
            g0_s_b = np.loadtxt(f"mo.coeffs.g{gi}.s{si}.b.txt")
            g0_s = uhf_to_ghf(g0_s_a, g0_s_b, nalpha, nbeta)
            g0_list.append(g0_s)

        a = np.loadtxt(f"noci.coeffs.g{gi}.txt")

        e1_noci = get_e1_noci(a, mol, atom, coord, g0_list, nelec,
                              complexsymmetric)

        bond_length = mol.atom[1][1][2] - mol.atom[0][1][2]

        with open("analytic_derivatives.txt", "a") as f:
            f.write(f"{bond_length}     {e1_noci}\n")


def get_numerical_derivatives_h2(g_num, states_num):

    for gi in range(30,g_num-1):

        bond_length = (np.loadtxt(f"coords.g{gi}.atom1.txt")[2]
                       - np.loadtxt(f"coords.g{gi}.atom0.txt")[2])

        delta_bl = (np.loadtxt(f"coords.g{gi+1}.atom1.txt")[2]
                    - np.loadtxt(f"coords.g{gi}.atom1.txt")[2])

        delta_e = (np.loadtxt(f"noci.energy.g{gi+1}.txt")
                   - np.loadtxt(f"noci.energy.g{gi}.txt"))

        num_deriv = delta_e/delta_bl

        with open("numerical_derivatives.txt", "a") as f:
            f.write(f"{bond_length}     {num_deriv}\n")


def get_scf_derivatives_h2(g_num):


    for gi in range(g_num):

        mol = gto.Mole()
        mol.basis = 'sto-3g'
        mol.unit = 'bohr'
        mol.atom = []
        for atom in range(2):

            pos_vec = np.loadtxt(f"coords.g{gi}.atom{atom}.txt")

            mol.atom.extend([['H',(pos_vec[0], pos_vec[1], pos_vec[2])]])

        mol.build()

        g0_s_a = np.loadtxt(f"mo.coeffs.g{gi}.s{0}.a.txt")
        g0_s_b = np.loadtxt(f"mo.coeffs.g{gi}.s{0}.b.txt")
        g0 = uhf_to_ghf(g0_s_a, g0_s_b, 1, 1)

        e1_scf = get_e1_scf(mol, 1, 2, 2, False, g0)

        bond_length = mol.atom[1][1][2] - mol.atom[0][1][2]

        with open("scf_derivatives.txt", "a") as f:
            f.write(f"{bond_length}     {e1_scf}\n")


def get_noci_energy_h2(g_num):

    for gi in range(0,g_num-1):

        bond_length = (np.loadtxt(f"coords.g{gi}.atom1.txt")[2]
                       - np.loadtxt(f"coords.g{gi}.atom0.txt")[2])


        e = np.loadtxt(f"noci.energy.g{gi+1}.txt")


        with open("noci_energy.txt", "a") as f:
            f.write(f"{bond_length}     {e}\n")
# get_analytic_derivatives_h2(91, 2, 2, 2, 1, 1, 1, 2, False)
# get_numerical_derivatives_h2(91, 2)
# get_scf_derivatives_h2(91)
get_noci_energy_h2(91)
