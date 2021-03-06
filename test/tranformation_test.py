import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *
from energy_derivative import *



h3 = h3_states()
mol = h3.mol
nelec = h3.nelec
atom = h3.atom
coord = h3.coord
g00 = h3.g00
g10 = h3.g10
g20 = h3.g20
g30 = h3.g30
a = h3.a
g0_list = [g00, g10, g20, g30]

''' First index is the state number, second is the order'''

lambda0_01, g00_t, g10_t = lowdin_pairing0(g00, g10, mol, nelec, False)

print("Non transformed:\n")
print("non transformed MO matrix:\n", g00)
g01 = g1_iteration(False, mol, atom, coord, nelec, g00)
g11 = g1_iteration(False, mol, atom, coord, nelec, g10)
print("transformed:\n")
print("transformed MO matrix:\n", g00_t)
g01_t = g1_iteration(False, mol, atom, coord, nelec, g00_t)
g11_t = g1_iteration(False, mol, atom, coord, nelec, g10_t)
# print("non transformed derivative matrix:\n", g01)
# print("transformed derivative matrix:\n", g01_t)

e0 = get_e0_elec(mol, g00, False, nelec)
e0_t = get_e0_elec(mol, g00_t, False, nelec)

e01 = get_e1_elec(mol, g00, g01, atom, coord, False, nelec)
e01_t = get_e1_elec(mol, g00_t, g01_t, atom, coord, False, nelec)
print("SCF e1 using non transformed g1 zeroth state:\n", e01)
print("SCF e1 using transformed g1 zeroth state:\n", e01_t)
e11 = get_e1_elec(mol, g10, g11, atom, coord, False, nelec)
e11_t = get_e1_elec(mol, g10_t, g11_t, atom, coord, False, nelec)
print("SCF e1 using non transformed g1 first state:\n", e11)
print("SCF e1 using transformed g1 first state:\n", e11_t)

sao0 = mol.intor("int1e_ovlp")
omega = np.identity(2)
sao0 = np.kron(omega, sao0)
