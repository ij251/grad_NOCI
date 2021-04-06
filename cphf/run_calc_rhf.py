import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import g1_iteration, write_e1_mat, write_e1_single
from zeroth_order_ghf import rhf_to_ghf

"H2"
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


"OH-"
# mol = gto.M(
#         atom = (
#             f"O 0 0 0;"
#             f"H 0 0 1;"
#         ),
#         basis = 'sto-3g',
#         unit = 'Bohr',
#         charge = -1,
#         )
# nelec = 10
# atom = 1
# coord = 2


"H3+ C2v"
# mol = gto.M(
#         atom = (
#             f"H 0 0.3745046 -1.9337695;"
#             f"H 0 -0.7492090 0;"
#             f"H 0 0.3745046 1.9337695;"
#         ),
#         basis = 'sto-3g',
#         unit = 'Bohr',
#         charge = 1,
#         spin = 0)
# nelec = 2
# atom = 1
# coord = 1


"H2O"
# mol = gto.M(
#         atom = (
#             f"O 0 0.1088584 0;"
#             f"H 0 -0.8636449 1.2990232;"
#             f"H 0 -0.8636449 -1.2990232;"
#         ),
#         basis = 'sto-3g',
#         unit = 'Bohr',
#         charge = 0,
#         spin = 0)
# nelec = 10
# atom = 0
# coord = 1


m = scf.RHF(mol)
m.kernel()
print("\nPySCF output of first order energies:")
g = grad.rhf.Gradients(m)
g.kernel()

g1 = g1_iteration(False, mol, atom, coord, nelec)
# print("g1:\n", g1)
write_e1_single(mol, nelec, atom, coord, False)
write_e1_mat(mol, nelec, False)
