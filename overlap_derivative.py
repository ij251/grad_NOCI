from pyscf import mol, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *


def get_wxlambda0(gw0, gx0, mol, complexsymmetric: bool):

    sao0 = mol.intor("int1e_ovlp")
    if not complexsymmetric:
        wxs0 = np.linalg.multi_dot([gw0.T.conj(), sao0, gx0])
    else:
        wxs0 = np.linalg.multi_dot([gw0.T, sao0, gx0])

    wxu,_,wxvh = np.linalg.svd(wxs0)
    wxv = wxvh.T.conj()

    wxlambda0 = np.linalg.multi_dot([wxu.T.conj(), wxs0, wxv])

    return wxlambda0


def get_wxlambda1(gw0, gw1, gx0, gx1, mol, atom, coord,
                  complexsymmetric: bool):

    sao0 = mol.intor("int1e_ovlp")
    sao1 = get_s1(mol, atom, coord)

    if not complexsymmetric:
        wxs0 = np.linalg.multi_dot([gw0.T.conj(), sao0, gx0])
        wxs1 = (np.linalg.multi_dot([gw1.T.conj(), sao0, gx0])
                + np.linalg.multi_dot([gw0.T.conj(), sao1, gx0])
                + np.linalg.multi_dot([gw0.T.conj(), sao0, gx1]))
    else:
        wxs0 = np.linalg.multi_dot([gw0.T, sao0, gx0])
        wxs1 = (np.linalg.multi_dot([gw1.T, sao0, gx0])
                + np.linalg.multi_dot([gw0.T, sao1, gx0])
                + np.linalg.multi_dot([gw0.T, sao0, gx1]))

    wxu,_,wxvh = np.linalg.svd(wxs0)
    wxv = wxvh.T.conj()

    wxlambda1 = np.diag(np.diag(np.linalg.multi_dot([wxu.T.conj(),wxs1,wxv])))

    return wxlambda1


def lowdin_prod(wxlambda0, rmind):

    wxlambda0_diag = np.diag(wxlambda0)
    ind = list(set(range(len(l)))-set(rmind))
    lowdin_prod = np.prod(wxlambda0_diag[ind])

    return lowdin_prod


def get_swx1(wxlambda0, wxlambda1, nelec):

    swx1 = np.sum(wxlambda1_diag[j,j]*lowdin_prod(wxlambda0, [j])
                  for j in range(nelec))

    return get_swx1


def get_g1_list(g0_list, complexsymmetric: bool, mol, atom, coord, nelec):

    g1_list = [g1_iteration(complexsymmetric, mol, atom, coord, nelec,
                            g0_list[i]) for i in range(g0_list.shape[0])]

    return g1_list


def get_s1mat(nnoci, g0_list, g1_list, mol, atom, coord, nelec,
              complexsymmetric: bool):

    s1mat = np.zeros(nnoci)

    for w in range(nnoci):
        for x in range(nnoci):

            wxlambda0 = get_wxlambda0(g0_list[w], g0_list[x], mol,
                                      complexsymmetric)
            wxlambda1 = get_wxlambda1(g0_list[w], g1_list[w], g0_list[x],
                                      g1_list[x], mol, atom, coord,
                                      complexsymmetric)

            s1mat[w,x] = get_swx1(wxlambda0, wxlambda1, nelec)

    return s1mat


def transform_g(gw0, gx0, mol, complexsymmetric: bool):

    sao0 = mol.intor("int1e_ovlp")
    if not complexsymmetric:
        wxs0 = np.linalg.multi_dot([gw0.T.conj(), sao0, gx0])
    else:
        wxs0 = np.linalg.multi_dot([gw0.T, sao0, gx0])

    wxu,_,wxvh = np.linalg.svd(wxs0)
    wxv = wxvh.T.conj()
    det_wxu = np.linalg.det(wxu)
    det_wxv = np.linalg.det(wxv)

    if not complexsymmetric:
        gw0_t = np.dot(gw0, wxu)
    else:
        gw0_t = np.dot(gw0, wxu.conj())

    gx0_t = np.dot(gx0, wxv)

    gw0_t[:,0] *= 1/det_wxu
    gx0_t[:,0] *= 1/det_wxv

    return gw0_t, gx0_t
