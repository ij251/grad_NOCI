import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import g1_iteration, write_e1_mat, write_e1_single
from zeroth_order_ghf import uhf_to_ghf

"H3 UHF"
mol = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 2;"
            f"H 0 1 0;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 1)
nelec = 3
nalpha = 2
nbeta = 1
atom = 1
coord = 2

'h3 uhf noci states'

g0alpha0 = np.array([[0.431728, -0.325562, 1.628633],
                     [0.315151, 1.068455, -0.186595],
                     [0.430280, -0.481068, -1.525352]])

g0beta0 = np.array([[0.529658, -0.149873, 1.625370],
                    [0.099575, 1.097291, -0.248539],
                    [0.473328, -0.471677, -1.515503]])

g0ghf0 = uhf_to_ghf(g0alpha0, g0beta0, nalpha, nbeta) #energy = -1.4222053120

m = scf.UHF(mol)
m.kernel()
print("\nPySCF output of first order energies:")
g = grad.uhf.Gradients(m)
g.kernel()

write_e1_single(mol, nelec, atom, coord, False, g0ghf0)
write_e1_mat(mol, nelec, False, g0ghf0)

