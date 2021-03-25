import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *
from energy_derivative import *
from h3states.h3_states import *


''' First index is the state number, second is the order'''
print("First index is the state number, second is the order")

h3 = h3_states4()
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
print("g00:\n", g00)
print("g10:\n", g10)
print("g20:\n", g20)
print("g30:\n", g30)

# g01 = g1_iteration(False, mol, atom, coord, nelec, g00)
# g11 = g1_iteration(False, mol, atom, coord, nelec, g10)
# g21 = g1_iteration(False, mol, atom, coord, nelec, g20)
g31 = g1_iteration(False, mol, atom, coord, nelec, g30)

# print("g01:\n", g01)
# print("g11:\n", g11)
# print("g21:\n", g21)
print("g31:\n", g31)
# write_e1_mat(mol, nelec, False, g10)

