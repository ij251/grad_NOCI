import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *
from energy_derivative import *
from h3states.h3_states import *


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

for i in range(len(g0_list)):
    print("g"+ str(i) + "0:\n", g0_list[i])


# h0mat = get_h0mat(mol, g0_list, nelec, False)
e0 = get_e0_noci(a, mol, g0_list, nelec, False)
s0mat = get_s0mat(mol, g0_list, nelec, False)
print("\noverlap matrix:\n", s0mat)
# print("hamiltonian matrix:\n", h0mat)
print("\nZeroth order NOCI energy:", e0)

