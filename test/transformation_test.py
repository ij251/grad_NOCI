import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *


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

g00 = g0_list[0]
g10 = g0_list[1]
_, g00_t, g10_t = lowdin_pairing0(g00, g10, mol, nelec, False)

g01_t = g1_iteration(False, mol, atom, coord, nelec, g00_t)
g11_t = g1_iteration(False, mol, atom, coord, nelec, g10_t)

g01 = g1_iteration(False, mol, atom, coord, nelec, g00)
g11 = g1_iteration(False, mol, atom, coord, nelec, g10)

e1_t = get_e1_elec(mol, g01_t, atom, coord, False, nelec)
e1 = get_e1_elec(mol, g01, atom, coord, False, nelec)
print("g0:\n", g00)
print("g0_t:\n", g00_t)

print("e1 from non transformed g0:\n", e1)
print("e1 from transformed g0:\n", e1_t)
