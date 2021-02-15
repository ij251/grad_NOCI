from pyscf import mol, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *

def get_e1(a, h1, s1):

    nnoci = a.shape[0]
    e1 = 0
    for w in range(nnoci):
    for x in range(nnoci):

        e1 += a[w].conj()*a[x]*(h1[w,x]-e0*s1[w,x])

    return e1


