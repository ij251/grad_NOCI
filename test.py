import sys
from pyscf import gto, scf, grad
from first_order_ghf import get_s1

print(sys.path)

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


print(get_s1(mol, atom, coord))






