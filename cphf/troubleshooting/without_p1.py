import numpy as np
from pyscf import gto, scf, grad
from rhf_CPHFfunctions import get_p0, get_p1, get_hcore0, get_pi0,\
get_f0, get_s1, get_hcore1, get_pi1, get_f1, get_g1_x, g1_iteration,\
get_e0_elec, get_e0_nuc, get_e1_elec, get_e1_nuc, make_ghf, get_x_lowdin

mol = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = '6-31g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)
nelec = 2
atom = 1
coord = 2

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

m = scf.RHF(mol)
m.kernel()
g0 = m.mo_coeff
# g0 = make_ghf(m.mo_coeff, nelec)

def get_pe0(g0, nelec, mol):

    """Calculates energy weighted density matrix"""

    x = get_x_lowdin(mol)
    g0_x = np.dot(np.linalg.inv(x), g0)
    p0 = get_p0(g0, False, int(nelec/2))
    hcore0 = get_hcore0(mol)
    pi0 = get_pi0(mol)
    f0 = get_f0(hcore0, pi0, p0)
    f0_x = np.linalg.multi_dot([x.T.conj(), f0, x])
    eta0 = np.linalg.eig(f0_x)[0]
    print(eta0)
    index = np.argsort(eta0)
    eta0 = eta0[index]

    pe0 = np.zeros_like(g0)

    for k in range(g0.shape[1]):
        for l in range(g0.shape[1]):

            for i in range(int(nelec/2)):

                pe0[k,l] += g0[k,i]*g0[l,i]*eta0[i]

    # print("p0:\n", p0)
    # print("f0:\n", f0)
    # print("eta0:\n", eta0)
    # print("pe0:\n", pe0)
    return pe0


def get_e1_without_p1(g0, mol, atom, coord, nelec):

    """Get e1 using the Henry Schafer formula"""

    hcore1 = get_hcore1(mol, atom, coord)
    pi1 = get_pi1(mol, atom, coord)
    p0 = get_p0(g0, False, int(nelec/2))
    s1 = get_s1(mol, atom, coord)
    pe0 = get_pe0(g0, nelec, mol)

    e1 = 0
    blah = 0
    bleh = 0
    bluh = 0

    for p in range(g0.shape[1]):
        for q in range(g0.shape[1]):

            e1 += 2*(p0[p,q]*1*hcore1[p,q])
            e1 -= 2*(pe0[p,q]*s1[p,q])
            blah += 2*(p0[p,q]*1*hcore1[p,q])
            bleh += 2*(pe0[p,q]*1*s1[p,q])
            # print("One electron:\n",
                  # 2*(p0[p,q]*1*hcore1[p,q] - pe0[p,q]*s1[p,q]))

            for r in range(g0.shape[1]):
                for s in range(g0.shape[1]):

                    e1 += (2*p0[p,q]*p0[r,s] - p0[p,r]*p0[q,s])*pi1[p,q,r,s]
                    bluh += (2*p0[p,q]*p0[r,s] - p0[p,r]*p0[q,s])*pi1[p,q,r,s]
                    # print("two electron:\n",
                          # (2*p0[p,q]*p0[r,s] - p0[p,r]*p0[q,s])*pi1[p,q,r,s])
    print(blah)
    print(bluh)
    print(bleh)
    print(e1)

    return e1

e1 = get_e1_without_p1(g0, mol, atom, coord, nelec)
# print("e1:\n", e1)
# print("e1*0.05:\n", e1*0.05)
# print("e1 pyscf:\n", -1.08215657128714 +1.06599946214331)

