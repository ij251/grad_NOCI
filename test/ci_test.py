import numpy as np
from pyscf import gto, scf, grad
from cphf.first_order_ghf import get_e1_elec, g1_iteration
from cphf.zeroth_order_ghf import rhf_to_ghf, uhf_to_ghf
from non_ortho import lowdin_pairing
from energy_derivative import get_e1_noci
from overlap_derivative import get_s0mat, get_s1mat, get_g1_list
from hamiltonian_derivative import get_h1mat


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

# sigma g^2
g0rhf1 = np.array([[0.52754646684681505420, -1.56782302496071901388],
                   [0.52754646684681505420, 1.56782302496071901388]])

g0ghf1 = rhf_to_ghf(g0rhf1, nelec)

# sigma g sigma u
g0uhfalpha0 = np.array([[0.52754646684681505420, -1.56782302496071901388],
                        [0.52754646684681505420, 1.56782302496071901388]])

g0uhfbeta0 = np.array([[-1.56782302496071901388, 0.52754646684681505420],
                        [1.56782302496071901388, 0.52754646684681505420]])
g0ghf2 = uhf_to_ghf(g0uhfalpha0, g0uhfbeta0, 1, 1)

# sigma u^2
# g0rhf2 = np.array([[-1.56782302496071901388, 0.52754646684681505420],
#                    [1.56782302496071901388, 0.52754646684681505420]])

# g0ghf2 = rhf_to_ghf(g0rhf2, nelec) #energy = 1.1555300839

a_g2u2 = [0.9971103, -0.0759676]
a_g2ug = [1, 0] #???

g0_list = [g0ghf1, g0ghf2]
# g0_list = [g0ghf2]
g1_list = get_g1_list(mol, atom, coord, g0_list, nelec, False)

e1 = get_e1_noci(a_g2ug, mol, atom, coord, g0_list, nelec, False)
print("e1_noci:\n", e1)




"Slow troubleshooting"

print("0_g0 sigma g 2:\n", g0ghf1)
print("0_g1 sigma g 2 (differentiated):\n", g1_list[0])
# print("1_g0 sigma g sigma u:\n", g0ghf2)
s1mat = get_s1mat(mol, atom, coord, g0_list, g1_list, nelec, False)
print("s1mat:\n", s1mat)
h1mat = get_h1mat(mol, atom, coord, g0_list, g1_list, nelec, False)
print("h1mat:\n", h1mat)
