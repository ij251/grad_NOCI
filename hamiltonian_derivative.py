import numpy as np
from pyscf import mol, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from non_ortho import *


def get_onewx0(mol, gw0_t, gx0_t, wxlambda0, nelec, complexsymmetric: bool):

    hcore0 = get_hcore0(mol)

    onewx0 = 0
    # onewx0 = np.sum(lowdin_prod(wxlambda0, [m])
    #                 * np.einsum("ij,ji->",
    #                             hcore0,
    #                             get_xwp0(gw0_t, gx0_t, m, complexsymmetric))
    #                 for m in range(nelec))

    for m in range(nelec):

        lowdin_prod0 = lowdin_prod(wxlambda0, [m])
        xwp0 = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)

        onewx0 += lowdin_prod0 * np.einsum("ij,ji->", hcore0, xwp0)

    return onewx0


def get_onewx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t, wxlambda0,
               wxlambda1, nelec, complexsymmetric: bool):

    hcore0 = get_hcore0(mol)
    hcore1 = get_hcore1(mol, atom, coord)
    wxlambda1_diag = np.diag(wxlambda1)

    onewx1 = 0

    for m in range(nelec):

        lowdin_prod0 = lowdin_prod(wxlambda0, [m])
        lowdin_prod1 = (np.sum(wxlambda1_diag[j,j]*lowdin_prod(wxlambda0,
                                                               [j,m])
                              for j in range(nelec))
                        - wxlambda1_diag[m,m]*lowdin_prod(wxlambda0,[m]))
        xwp0 = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)
        xwp1 = get_xwp1(gw0_t, gx0_t, gw0_t, gx1_t, m, complexsymmetric)

        a = lowdin_prod0 * np.einsum("ij,ji->", hcore1, xwp0)
        b = lowdin_prod1 * np.einsum("ij,ji->", hcore0, xwp0)
        c = lowdin_prod0 * np.einsum("ij,ji->", hcore0, xwp1)

        onewx1 += a + b + c

    return onexw1


def get_twowx0(mol, gw0_t, gx0_t, wxlambda0, nelec, complexsymmetric: bool):

    j0 = get_j0(mol)

    twowx0 = 0
    for m in range(nelec):
        for n in range(nelec):

            lowdin_prod0 = lowdin_prod(wxlambda0, [m,n])
            xwp0_m = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)
            xwp0_n = get_xwp0(gw0_t, gx0_t, n, complexsymmetric)

            twowx0 += lowdin_prod0 * (np.einsum("ijkl,ji,lk->",
                                                j0,xwp0_m,xwp0_n)
                                      - np.einsum("ijkl,li,jk->",
                                                  j0,xwp0_m,xwp0_n)) * 0.5

    return twowx0


def get_twowx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t, wxlambda0,
               wxlambda1, nelec, complexsymmetric: bool):

    j0 = get_j0(mol)
    j1 = get_j1(mol, atom, coord)
    wxlambda1_diag = np.diag(wxlambda1)

    twowx1 = 0
    for m in range(nelec):
        for n in range(nelec):


            lowdin_prod0 = lowdin_prod(wxlambda0, [m,n])
            lowdin_prod1 = (np.sum(wxlambda1_diag[j,j]*lowdin_prod(wxlambda0,
                                                                   [j,m,n])
                                  for j in range(nelec))
                            - wxlambda1_diag[m,m]*lowdin_prod(wxlambda0,[m])
                            - wxlambda1_diag[n,n]*lowdin_prod(wxlambda0,[n]))
            xwp0_m = get_xwp0(gw0_t, gx0_t, m, complexsymmetric)
            xwp1_m = get_xwp1(gw0_t, gx0_t, gw0_t, gx1_t, m, complexsymmetric)
            xwp0_n = get_xwp0(gw0_t, gx0_t, n, complexsymmetric)
            xwp1_n = get_xwp1(gw0_t, gx0_t, gw0_t, gx1_t, n, complexsymmetric)

            a = lowdin_prod0 * (np.einsum("ijkl,ji,lk->",j1,xwp0_m,xwp0_n)
                                - np.einsum("ijkl,li,jk->",j1,xwp0_m,xwp0_n))
            b = lowdin_prod1 * (np.einsum("ijkl,ji,lk->",j0,xwp0_m,xwp0_n)
                                - np.einsum("ijkl,li,jk->",j0,xwp0_m,xwp0_n))
            c = lowdin_prod0 * (np.einsum("ijkl,ji,lk->",j0,xwp1_m,xwp0_n)
                                + np.einsum("ijkl,ji,lk->",j0,xwp0_m,xwp1_n)
                                - np.einsum("ijkl,li,jk->",j0,xwp1_m,xwp0_n)
                                - np.einsum("ijkl,li,jk->",j0,xwp0_m,xwp1_n))

            twowx1 += 0.5 * (a + b + c)

    return twoxw1


def get_nucwx0(mol, wxlambda0, complexsymmetric: bool):

    swx0 = lowdin_prod(wxlambda0, [])
    e0_nuc = get_e0_nuc(mol)

    nucwx0 = swx0 * e0_nuc

    return nucwx0


def get_nucwx1(mol, atom, coord, wxlambda0, wxlambda1, nelec,
               complexsymmetric: bool):


    swx0 = lowdin_prod(wxlambda0, [])
    swx1 = get_swx1(wxlambda0, wxlambda1, nelec)

    e0_nuc = get_e0_nuc(mol)
    e1_nuc = get_e1_nuc(mol, atom, coord)

    nucwx1 = swx1 * e0_nuc + swx0 * e1_nuc

    return nucwx1


def get_h0mat(mol, g0_list, nelec, complexsymmetric: bool):

    nnoci = g0_list.shape[0]
    h0mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            gw0 = g0_list[w]
            gx0 = g0_list[x]
            gw0_t, gx0_t = transform_g(gw0, gx0, mol, complexsymmetric)

            wxlambda0 = get_wxlambda0(gw0, gx0, mol, complexsymmetric)

            onewx0 = get_onewx0(mol, gw0_t, gx0_t, wxlambda0, nelec,
                                complexsymmetric)
            twowx0 = get_twowx0(mol, gw0_t, gx0_t, wxlambda0, nelec,
                                complexsymmetric)
            nucwx0 = get_nucwx0(mol, wxlambda0, complexsymmetric)

            h0mat[w,x] = onewx0 + twowx0 + nucwx0

    return h0mat


def get_h1mat(mol, atom, coord, g0_list, nelec, complexsymmetric: bool):

    nnoci = g0_list.shape[0]
    h1mat = np.zeros((nnoci,nnoci))

    for w in range(nnoci):
        for x in range(nnoci):

            gw0 = g0_list[w]
            gx0 = g0_list[x]
            gw0_t, gx0_t = transform_g(gw0, gx0, mol, complexsymmetric)
            gw1_t = g1_iteration(complexsymmetric, mol, atom, coord, nelec,
                                 gw0_t)
            gx1_t = g1_iteration(complexsymmetric, mol, atom, coord, nelec,
                                 gx0_t)


            wxlambda0 = get_wxlambda0(gw0, gx0, mol, complexsymmetric)
            wxlambda1 = get_wxlambda1(gw0, gw1, gx0, gx1, mol, atom, coord,
                                      complexsymmetric)

            onewx1 = get_onewx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t,
                                wxlambda0, wxlambda1, nelec, complexsymmetric)
            twowx1 = get_twowx1(mol, atom, coord, gw0_t, gx0_t, gw1_t, gx1_t,
                                wxlambda0, wxlambda1, nelec, complexsymmetric)
            nucwx1 = get_nucwx1(mol, atom, coord, wxlambda0, wxlambda1, nelec,
                                complexsymmetric)

            h1mat[w,x] = onewx1 + twowx1 + nucwx1

    return h1mat
