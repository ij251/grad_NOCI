from pyscf import mol, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from overlap_derivative import *

def get_nucwx1(mol, atom, coord, gw0, gx0, gw1, gx1, nelec,
               complexsymmetric: bool):

    wxlambda0 = get_wxlambda0(gw0, gx0, mol, complexsymmetric)
    wxlambda1 = get_wxlambda1(gw0, gw1, gx0, gx1, mol, atom, coord,
                              complexsymmetric)
    swx0 = lowdin_prod(wxlambda0, [])
    swx1 = get_swx1(wxlambda0, wxlambda1, nelec)

    e0_nuc = get_e0_nuc(mol)
    e1_nuc = get_e1_nuc(mol, atom, coord)

    nucwx1 = swx1 * e0_nuc + swx0 * e1_nuc

    return nucwx1

