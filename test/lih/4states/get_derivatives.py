import numpy as np
from pyscf import gto, scf, grad
from cphf.first_order_ghf import g1_iteration, write_e1_single
from cphf.zeroth_order_ghf import uhf_to_ghf
from overlap_derivative import get_s1mat, get_g1_list
from hamiltonian_derivative import get_h1mat
from energy_derivative import get_e1_noci


def get_analytic_derivatives_lih(g_num, states_num, complexsymmetric):


    for gi in range(g_num):

        print(gi)
        mol = gto.Mole()
        mol.basis = 'sto-3g'
        mol.unit = 'bohr'
        mol.atom = []

        pos_vec0 = np.loadtxt(f"coords.g{gi}.atom{0}.txt")
        pos_vec1 = np.loadtxt(f"coords.g{gi}.atom{1}.txt")

        mol.atom.extend([['Li',(pos_vec0[0], pos_vec0[1], pos_vec0[2])]])
        mol.atom.extend([['H',(pos_vec1[0], pos_vec1[1], pos_vec1[2])]])

        mol.build()
        g0_list = []

        for si in range(states_num):

            g0_s_a = np.loadtxt(f"mo.coeffs.g{gi}.s{si}.a.txt")
            g0_s_b = np.loadtxt(f"mo.coeffs.g{gi}.s{si}.b.txt")
            g0_s = uhf_to_ghf(g0_s_a, g0_s_b, 2, 2)
            g0_list.append(g0_s)

        a = np.loadtxt(f"noci.coeffs.g{gi}.txt")

        e1_noci = get_e1_noci(a, mol, 1, 2, g0_list, 4,
                              complexsymmetric)

        bond_length = mol.atom[1][1][2] - mol.atom[0][1][2]

        with open("analytic_derivatives.txt", "a") as f:
            f.write(f"{bond_length}     {e1_noci}\n")


def get_numerical_derivatives_lih(g_num):

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


# get_analytic_derivatives_lih(91, 4, False)
get_numerical_derivatives_lih(91)
