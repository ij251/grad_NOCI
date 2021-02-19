import numpy as np
from pyscf import mol, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *


def get_swx1(wxlambda0, wxlambda1, nelec):

    wxlambda1_diag = np.diag(wxlambda1)
    swx1 = np.sum(wxlambda1_diag[j,j]*lowdin_prod(wxlambda0, [j])
                  for j in range(nelec))

    return get_swx1


def get_s1mat(mol, atom, coord, g0_list, nelec, complexsymmetric: bool):

    g1_list = get_g1_list(mol, atom, coord, g0_list, nelec, complexsymmetric)
    nnoci = g0_list.shape[0]
    s1mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            wxlambda0 = get_wxlambda0(g0_list[w], g0_list[x], mol,
                                      complexsymmetric)
            wxlambda1 = get_wxlambda1(g0_list[w], g1_list[w], g0_list[x],
                                      g1_list[x], mol, atom, coord,
                                      complexsymmetric)

            s1mat[w,x] = get_swx1(wxlambda0, wxlambda1, nelec)

    return s1mat
