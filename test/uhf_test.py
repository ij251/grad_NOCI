import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *
from energy_derivative import *


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

g0alpha1 = np.array([[0.512116, 1.626205, -0.194988],
                     [0.172477, -0.240733, 1.089970],
                     [0.477400, -1.516960, -0.491867]])

g0beta1 = np.array([[0.311741, -0.438851, 1.629434],
                    [0.606218, 0.935777, -0.180431],
                    [0.285625, -0.579312, -1.525150]])

g0ghf1 = uhf_to_ghf(g0alpha1, g0beta1, nalpha, nbeta) #energy = -0.6995970325

g10 = g1_iteration(False, mol, atom, coord, nelec, g0ghf0)

print(g10)
