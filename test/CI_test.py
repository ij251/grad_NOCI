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

g0rhf1 = np.array([[0.52754646684681505420, -1.56782302496071901388],
                   [0.52754646684681505420, 1.56782302496071901388]])

g0ghf1 = rhf_to_ghf(g0rhf1, nelec) #energy = -1.0659994622

g0rhf2 = np.array([[-1.56782302496071901388, 0.52754646684681505420],
                   [1.56782302496071901388, 0.52754646684681505420]])

g0ghf2 = rhf_to_ghf(g0rhf2, nelec) #energy = 1.1555300839

a = [0.9971103, -0.0759676]

g0_list = [g0ghf1, g0ghf2]

'''test energy'''
e0 = get_e0_noci(a, mol, g0_list, nelec, False)
print("e0:\n", e0)
e1 = get_e1_noci(a, mol, atom, coord, g0_list, nelec, False)
print("e1:\n", e1)

'''test zeroth order overlap/hamiltonian'''
lambda0_00,_,_ = lowdin_pairing0(g0_list[0], g0_list[0], mol, nelec, False)
# lambda0_01,_,_ = lowdin_pairing0(g0_list[0], g0_list[1], mol, nelec, False)
# lambda0_10,_,_ = lowdin_pairing0(g0_list[1], g0_list[0], mol, nelec, False)
# lambda0_11,_,_ = lowdin_pairing0(g0_list[1], g0_list[1], mol, nelec, False)

# print("lowdin overlaps between 0 and 0:\n", lambda0_00)
# print("lowdin overlaps between 0 and 1:\n", lambda0_01)
# print("lowdin overlaps between 1 and 0:\n", lambda0_10)
# print("lowdin overlaps between 1 and 1:\n", lambda0_11)

# s0mat = get_s0mat(mol, g0_list, nelec, False)
# print("matrix of overlaps between determinants:\n", s0mat)

# h0mat = get_h0mat(mol, g0_list, nelec, False)
# print("matrix of hamiltonaian elements between determinants:\n", h0mat)

'''test derivative of transformed MO matrices'''
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
