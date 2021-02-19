import numpy as np
from pyscf import mol, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from overlap_derivative import *
from hamiltonian_derivative import *
from non_ortho import *


def get_e0_noci(a, mol, g0_list, nelec, complexsymmetric):

    h0mat = get_h0mat(mol, g0_list, nelec, complexsymmetric)
    nnoci = g0_list.shape[0]

    if not complexsymmetric:
        e0_noci = np.sum(a[w].conj()*a[x]*h0mat[w,x]
                         for w in range(nnoci) for x in range(nnoci))
    else:
        e0_noci = np.sum(a[w]*a[x]*h0mat[w,x]
                         for w in range(nnoci) for x in range(nnoci))

    return e0_noci


def get_e1_noci(a, mol, atom, coord, g0_list, nelec, complexsymmetric):

    e0_noci = get_e0_noci(a, mol, g0_list, nelec, complexsymmetric)
    s1mat = get_s1mat(mol, atom, coord, g0_list, nelec, complexsymmetric)
    h1mat = get_h1mat(mol, atom, coord, g0_list, nelec, complexsymmetric)
    nnoci = g0_list.shape[0]

    if not complexsymmetric:
        e1_noci = np.sum(a[w].conj()*a[x]*(h1mat[w,x] - e0_noci*s1mat[w,x])
                         for w in range(nnoci) for x in range(nnoci))
    else:
        e1_noci = np.sum(a[w]*a[x]*(h1mat[w,x] - e0_noci*s1mat[w,x])
                         for w in range(nnoci) for x in range(nnoci))

    return e1
