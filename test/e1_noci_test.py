import numpy as np
from pyscf import gto, scf, grad
from cphf.first_order_ghf import g1_iteration
from overlap_derivative import get_s1mat, get_g1_list
from hamiltonian_derivative import get_h1mat
from energy_derivative import get_e1_noci
from h3states.h3_states import h3_states2, h3_states4


h3 = h3_states2()
# h3 = h3_states4()
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
# g0_list = [g00, g10, g20, g30]

g1_list = get_g1_list(mol, atom, coord, g0_list, nelec, False)

print(g10)

s1mat = get_s1mat(mol, atom, coord, g0_list, g1_list, nelec, False)
h1mat = get_h1mat(mol, atom, coord, g0_list, g1_list, nelec, False)
e1_noci = get_e1_noci(a, mol, atom, coord, g0_list, nelec, False)

print("s1mat:\n", s1mat)
print("h1mat:\n", h1mat)
print("e1_noci:\n", e1_noci)
