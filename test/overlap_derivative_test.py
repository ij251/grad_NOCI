import numpy as np
from pyscf import gto, scf, grad
from cphf.first_order_ghf import g1_iteration
from overlap_derivative import get_g1_list, get_s1mat, get_s0mat
from hamiltonian_derivative import get_swx1_bang
from h3states.h3_states import h3_states2, h3_states2_g1



''' First index is the state number, second is the order'''

h3 = h3_states2()
mol = h3.mol
nelec = h3.nelec
atom = h3.atom
coord = h3.coord
g00 = h3.g00
g10 = h3.g10
# g20 = h3.g20
# g30 = h3.g30
a = h3.a
g0_list = [g00, g10]
g1_list = get_g1_list(mol, atom, coord, g0_list, nelec, False)

s0matg0 = get_s0mat(mol, g0_list, nelec, False)
np.set_printoptions(precision=16)
print("s0mat at unperturbed geometry:\n", s0matg0)
# s1mat = get_s1mat(mol, atom, coord, g0_list, g1_list, nelec, False)
# print("s1mat:\n", s1mat)

h3 = h3_states2_g1()
mol = h3.mol
nelec = h3.nelec
atom = h3.atom
coord = h3.coord
g00 = h3.g00
g10 = h3.g10
# g20 = h3.g20
# g30 = h3.g30
a = h3.a
g0_list = [g00, g10]
g1_list = get_g1_list(mol, atom, coord, g0_list, nelec, False)

s0matg1 = get_s0mat(mol, g0_list, nelec, False)
np.set_printoptions(precision=16)
print("s0mat at perturbed geometry:\n", s0matg1)
# s1mat = get_s1mat(mol, atom, coord, g0_list, g1_list, nelec, False)
# print("s1mat:\n", s1mat)
print("numerical derivative:\n", (s0matg1 - s0matg0)*(1/0.0000001))
