import numpy as np
from pyscf import gto, scf, grad
from zeroth_order_ghf import *


'h3 uhf noci states'

g0alpha0 = np.loadtxt("h3states/state0_a")
g0beta0 = np.loadtxt("h3states/state0_b")
# g0alpha0 = np.array([[0.431728, -0.325562, 1.628633],
#                      [0.315151, 1.068455, -0.186595],
#                      [0.430280, -0.481068, -1.525352]])

# g0beta0 = np.array([[0.529658, -0.149873, 1.625370],
#                     [0.099575, 1.097291, -0.248539],
#                     [0.473328, -0.471677, -1.515503]])


g0alpha1 = np.loadtxt("h3states/state1_a")
g0beta1 = np.loadtxt("h3states/state1_b")
# g0alpha1 = np.array([[-0.413134, 1.627175, 0.355595],
#                      [0.993907, -0.175383, 0.507073],
#                      [-0.538985, -1.528109, 0.342997]])

# g0beta1 = np.array([[1.624680, 0.431407, -0.345149],
#                     [-0.205444, 0.405565, 1.033946],
#                     [-1.526603, 0.357070, -0.534090]])


g0alpha2 = np.loadtxt("h3states/state2_a")
g0beta2 = np.loadtxt("h3states/state2_b")

# g0alpha2 = np.array([[0.512116, 1.626205, -0.194988],
#                      [0.172477, -0.240733, 1.089970],
#                      [0.477400, -1.516960, -0.491867]])

# g0beta2 = np.array([[0.311741, -0.438851, 1.629434],
#                     [0.606218, 0.935777, -0.180431],
#                     [0.285625, -0.579312, -1.525150]])

g0alpha3 = np.loadtxt("h3states/state3_a")
g0beta3 = np.loadtxt("h3states/state3_b")

# g0alpha3 = np.array([[0.525621, 0.022494, 1.633415],
#                      [0.019427, 1.077924, -0.366805],
#                      [0.520028, -0.555978, -1.470961]])

# g0beta3 = np.array([[-0.133624, 0.493226, 1.638200],
#                     [1.092289, 0.119214, -0.261584],
#                     [-0.499144, 0.498961, -1.498486]])


''' First index is the state number, second is the order'''

class h3_states4:
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
    a = [0.9989168, 0.0545437, 0.0109276, 0.0025334]
    g00 = uhf_to_ghf(g0alpha0, g0beta0, nalpha, nbeta) #energy = -1.4222053120
    g10 = uhf_to_ghf(g0alpha1, g0beta1, nalpha, nbeta) #energy = 0.9899506826
    g20 = uhf_to_ghf(g0alpha2, g0beta2, nalpha, nbeta) #energy = -0.6995970325
    g30 = uhf_to_ghf(g0alpha3, g0beta3, nalpha, nbeta) #energy = -0.6985647926

class h3_states2:
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
    a = [0.9985121, 0.0545969]
    g00 = uhf_to_ghf(g0alpha0, g0beta0, nalpha, nbeta) #energy = -1.4222053120
    g10 = uhf_to_ghf(g0alpha1, g0beta1, nalpha, nbeta) #energy = 0.9899506826
    # g00[:,[2,1]]=g00[:,[1,2]]
    # g10[:,[2,1]]=g10[:,[1,2]]
