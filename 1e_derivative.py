from pyscf import mol, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from overlap_derivative import *

def get_xwp0(gw0_t, gx0_t, orbital_num, complexsymmetric):

    if not complexsymmetric:
        xwp0 = np.dot(gx0_t[:, orbital_num], gw0_t[:, orbital_num].T.conj())
    else:
        xwp0 = np.dot(gx0_t[:, orbital_num], gw0_t[:, orbital_num].T)

return xwp0


def get_xwp1(gw0_t, gx0_t, gw1_t, gx1_t, orbital_num, complexsymmetric):

    if not complexsymmetric:
        xwp1 = (np.dot(gx0_t[:, orbital_num],gw1_t[:, orbital_num].T.conj())
                +np.dot(gx1_t[:, orbital_num],gw0_t[:, orbital_num].T.conj()))
    else:
        xwp1 = (np.dot(gx0_t[:, orbital_num],gw1_t[:, orbital_num].T)
                +np.dot(gx1_t[:, orbital_num],gw0_t[:, orbital_num].T))

return xwp1


def get_onewx1(mol, atom, coord, gw0, gx0, gw1, gx1, nelec, complexsymmetric):

    hcore0 = get_hcore0(mol)
    hcore1 = get_hcore1(mol, atom, coord)
    wxlambda0 = get_wxlambda0(gw0, gx0, mol, complexsymmetric)
    wxlambda1 = get_wxlambda1(gw0, gw1, gx0, gx1, mol, atom, coord,
                              complexsymmetric)
    gw0_t, gx0_t = transform_g(gw0, gx0, mol, complexsymmetric)


    for m in range(nelec):

        lowdin_prod = lowdin_prod(wxlambda0, m)
        xwp0 = get_xwp0(gw0, 






