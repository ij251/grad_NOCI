import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *
from energy_derivative import *
from h3states.h3_states import *



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

s1mat = get_s1mat(mol, atom, coord, g0_list, g1_list, nelec, False)
print("s1mat:\n", s1mat)

# lambda0_01,_,_ = lowdin_pairing0(g00, g10, mol, nelec, False)
# print(lambda0_01)

