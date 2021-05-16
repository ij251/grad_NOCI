import numpy as np
from pyscf import gto, scf, grad
from cphf.first_order_ghf import get_e1_elec, g1_iteration
from cphf.zeroth_order_ghf import rhf_to_ghf, uhf_to_ghf
from non_ortho import lowdin_pairing
from energy_derivative import get_e1_noci
from overlap_derivative import get_s0mat, get_s1mat, get_g1_list
from hamiltonian_derivative import get_h1mat

mol = gto.M(
        atom = (
            f"Li 0 0 0.5;"
            f"H 0 0 0;"
        ),
        basis = '6-31g*',
        unit = 'Bohr',
        charge = 0,
        spin = 0)

nelec = 4
atom = 1
coord = 2

m = scf.UHF(mol)
m.kernel()
g0 = m.mo_coeff
for i in range(g0.shape[1]):
    print(f"g0 alpha row {i+1}:\n", g0[0][i,:])
for i in range(g0.shape[1]):
    print(f"g0 beta row {i+1}:\n", g0[1][i,:])
    
print("energy:\n", m.mo_energy)

cartlables= mol.cart_labels()
purelables= mol.pure_lables()
print("cart labels:\n", cartlables)
print("pure labels:\n", purelables)

