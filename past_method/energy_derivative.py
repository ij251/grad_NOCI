import numpy as np
from pyscf import gto, scf, grad
from first_order_ghf import *
from zeroth_order_ghf import *
from overlap_derivative import *
from hamiltonian_derivative import *
from non_ortho import *


def get_e0_noci(a, mol, g0_list, nelec, complexsymmetric):

    r"""Calculates the zeroth order noci energy.

    .. math::

            E_{\mathrm{NOCI}}&=\langle\Phi_{\mathrm{NOCI}}|\hat{\mathscr{H}}|
            \Phi_{\mathrm{NOCI}}\rangle
            = \sum \limits_{wx}^{\mathrm{NOCI}}A_w^{*\diamond}A_xH_{wx}

    :param a: python list of noci expansion coefficients.
    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param g0_list: Python list of molecular orbital coefficient matrices.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The noci energy.
    """

    h0mat = get_h0mat(mol, g0_list, nelec, complexsymmetric)
    nnoci = len(g0_list)

    if not complexsymmetric:
        e0_noci = np.sum(np.conj(a[w])*a[x]*h0mat[w,x]
                         for w in range(nnoci) for x in range(nnoci))
    else:
        e0_noci = np.sum(a[w]*a[x]*h0mat[w,x]
                         for w in range(nnoci) for x in range(nnoci))

    return e0_noci


def get_e1_noci(a, mol, atom, coord, g0_list, nelec, complexsymmetric):

    r"""Calculates the noci energy derivative.

    .. math::

            \frac{\partial E_{\mathrm{NOCI}}}{\partial X_A}
            = \sum \limits_{wx} A^{*\diamond}_wA_x \left[
            \frac{\partial H_{wx}}{\partial X_A} - E_{\mathrm{NOCI}}
            \frac{\partial S_{wx}}{\partial X_A}  \right]

    :param a: python list of noci expansion coefficients.
    :param mol: The pyscf molecule class, from which the nuclear coordinates
            and atomic numbers are taken.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param g0_list: Python list of molecular orbital coefficient matrices.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The noci energy derivative.
    """

    e0_noci = get_e0_noci(a, mol, g0_list, nelec, complexsymmetric)
    s1mat = get_s1mat(mol, atom, coord, g0_list, nelec, complexsymmetric)
    h1mat = get_h1mat(mol, atom, coord, g0_list, nelec, complexsymmetric)
    nnoci = len(g0_list)

    if not complexsymmetric:
        e1_noci = np.sum(np.conj(a[w])*a[x]*(h1mat[w,x] - e0_noci*s1mat[w,x])
                         for w in range(nnoci) for x in range(nnoci))
    else:
        e1_noci = np.sum(a[w]*a[x]*(h1mat[w,x] - e0_noci*s1mat[w,x])
                         for w in range(nnoci) for x in range(nnoci))

    return e1_noci
