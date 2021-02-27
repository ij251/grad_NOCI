import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *
from energy_derivative import *


mol = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)
nelec = 2
atom = 1
coord = 2

'h2 noci states'

g0rhf0 = np.array([[0.52754646684681505420, -1.56782302496071901388],
                   [0.52754646684681505420, 1.56782302496071901388]])

g0ghf0 = rhf_to_ghf(g0rhf0, nelec) #energy = -1.0659994622
print("g0 for determinant 0:\n", g0ghf0)

g0rhf1 = np.array([[-1.56782302496071901388, 0.52754646684681505420],
                   [1.56782302496071901388, 0.52754646684681505420]])

g0ghf1 = rhf_to_ghf(g0rhf1, nelec) #energy = 1.1555300839
print("g0 for determinant 1:\n", g0ghf1)

a = [0.9971103, -0.0759676]

g0_list = [g0ghf0, g0ghf1]

# lambda0_00 = get_wxlambda0(g0_list[0], g0_list[0], mol, False)
lambda0_01 = get_wxlambda0(g0_list[0], g0_list[1], mol, nelec, False)
# lambda0_10 = get_wxlambda0(g0_list[1], g0_list[0], mol, False)
# lambda0_11 = get_wxlambda0(g0_list[1], g0_list[1], mol, False)

# print("lowdin overlaps between 0 and 0:\n", lambda0_00)
print("lowdin overlaps between 0 and 1:\n", lambda0_01)
# print("lowdin overlaps between 1 and 0:\n", lambda0_10)
# print("lowdin overlaps between 1 and 1:\n", lambda0_11)

s0mat = get_s0mat(mol, g0_list, nelec, False)
print("matrix of overlaps between determinants:\n", s0mat)

h0mat = get_h0mat(mol, g0_list, nelec, False)
print("matrix of hamiltonaian elements between determinants:\n", h0mat)
