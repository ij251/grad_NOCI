import numpy as np
from pyscf import gto, scf, grad
from rhf_CPHFfunctions import get_p0, get_p1, get_hcore0, get_pi0,\
get_f0, get_s1, get_hcore1, get_pi1, get_f1, get_g1_x, g1_iteration,\
get_e0_elec, get_e0_nuc, get_e1_elec, get_e1_nuc, make_ghf

h2_mol_a = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)

h2_mol_b = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1.05;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)

h2_nelec = 2
h2_atom = 1
h2_coord = 2
m = scf.RHF(h2_mol_a)
print(h2_mol_a.ao_labels())

h3_mol_a = gto.M(
        atom = (
            f"H 0 0.3745046 -1.9337695;"
            f"H 0 -0.7492090 0;"
            f"H 0 0.3745046 1.9337695;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 1,
        spin = 0)

h3_mol_b = gto.M(
        atom = (
            f"H 0 0.3745046 -1.9337695;"
            f"H 0 -0.6992090 0;"
            f"H 0 0.3745046 1.9337695;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 1,
        spin = 0)
h3_nelec = 2
h3_atom = 1
h3_coord = 1


h2o_mol_a = gto.M(
        atom = (
            f"O 0 0.1088584 0;"
            f"H 0 -0.8636449 1.2990232;"
            f"H 0 -0.8636449 -1.2990232;"
        ),
        basis = '6-31g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)

h2o_mol_b = gto.M(
        atom = (
            f"O 0 0.1588584 0;"
            f"H 0 -0.8636449 1.2990232;"
            f"H 0 -0.8636449 -1.2990232;"
        ),
        basis = '6-31g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)
h2o_nelec = 10
h2o_atom = 0
h2o_coord = 1

def compare_pyscf(mol_a, mol_b, nelec, atom, coord):

    print("\n###############################################################")
    print("Geometry a")
    print("###############################################################\n")

    m_a = scf.RHF(mol_a)
    s0_a = mol_a.intor("int1e_ovlp")
    hcore0_a = (mol_a.intor('int1e_nuc') + mol_a.intor('int1e_kin'))
    spatial_j_a = mol_a.intor('int2e')
    print("s0_a:\n", s0_a)
    print("hcore0_a:\n", hcore0_a)
    # print("spatial_j_a:\n", spatial_j_a)
    print("e0_a:")
    m_a.kernel()

    print("\n###############################################################")
    print("Geometry b")
    print("###############################################################\n")

    m_b = scf.RHF(mol_b)
    s0_b = mol_b.intor("int1e_ovlp")
    hcore0_b = (mol_b.intor('int1e_nuc') + mol_b.intor('int1e_kin'))
    spatial_j_b = mol_b.intor('int2e')
    print("s0_b:\n", s0_b)
    print("hcore0_b:\n", hcore0_b)
    # print("spatial_j_b:\n", spatial_j_b)
    print("e0_b:")
    m_b.kernel()


    print("\n###############################################################")
    print("PySCF differences b - a")
    print("###############################################################\n")

    print("s0 difference:\n", s0_b - s0_a)
    print("hcore0 difference:\n", hcore0_b - hcore0_a)
    print("spatial_j difference:\n", spatial_j_b - spatial_j_a)

    print("\n###############################################################")
    print("My quantities * pertubation distance")
    print("###############################################################\n")

    hcore1 = get_hcore1(mol_a, atom, coord)
    j1_spatial = get_pi1(mol_a, atom, coord)
    s1 = get_s1(mol_a, atom, coord)

    print("s1:\n", s1*0.05)
    print("hcore1:\n", hcore1*0.05)
    print("j1_spatial:\n", j1_spatial*0.05)

    print("\n###############################################################")
    print("Difference between my first order objects and pyscf (expect zero)")
    print("###############################################################\n")

    delta_s1 = abs(s0_b - s0_a) - abs(s1*0.05)
    delta_hcore1 = abs(hcore0_b - hcore0_a) - abs(hcore1*0.05)
    delta_j1_spatial = abs(spatial_j_b - spatial_j_a) - abs(j1_spatial*0.05)
    print("s1 difference:\n", delta_s1)
    print("hcore1 difference:\n", delta_hcore1)
    print("spatial_j1 difference:\n", delta_j1_spatial)
    if np.allclose(delta_s1, np.zeros_like(delta_s1),
                   rtol=0, atol=1e-3):
        print("Change in overlap matches to first order")
    if np.allclose(delta_hcore1, np.zeros_like(delta_hcore1),
                   rtol=0, atol=1e-2):
        print("Change in hcore matches to first order")
    if np.allclose(delta_j1_spatial, np.zeros_like(delta_j1_spatial),
                   rtol=0, atol=1e-3):
        print("Change in j1_spatial matches to first order")


def compare_pyscf_energy(mol_a, mol_b):

    print("Total energy difference:\n", scf.hf.energy_tot(mol_b)
                                        - scf.hf.energy_tot(mol_a))

compare_pyscf(h2_mol_a, h2_mol_b, h2_nelec, h2_atom, h2_coord)
# compare_pyscf(h3_mol_a, h3_mol_b, h3_nelec, h3_atom, h3_coord)
# compare_pyscf(h2o_mol_a, h2o_mol_b, h2o_nelec, h2o_atom, h2o_coord)
