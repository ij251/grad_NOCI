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

e0 = get_e0_noci(a, mol, g0_list, nelec, False)
print("e0:\n", e0)
e1 = get_e1_noci(a, mol, atom, coord, g0_list, nelec, False)
print("e1:\n", e1)
