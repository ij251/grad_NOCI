
import numpy as np
import matplotlib.pyplot as plt

def plot_derivatives():

    noci = np.loadtxt("noci.xc0.noci.dat")
    scf1 = np.loadtxt("scf1.xc0.energy.dat")
    scf2 = np.loadtxt("scf2.xc0.energy.dat")


    x0 = noci[:,0]
    x1 = scf1[:,0]
    x2 = scf2[:,0]

    e0 = noci[:,1]
    e1 = scf1[:,1]
    e2 = scf2[:,1]

    plt.plot(x0, e0, label='NOCI energy', c='m',
             lw='0.5',)
    plt.plot(x1, e1, label='$\sigma_g^2$ energy', c='c',
             lw='0.5',)
    plt.plot(x2, e2, label='$\sigma^2_u$ energy', c='r',
             lw='0.5',)
    # plt.plot(x_numerical, e1_numerical, label='Numerical NOCI Derivative', c='b',
    #          lw='0', marker = 'x', markevery = (0.0,0.05))

    plt.xlabel("Bond length / Bohr")
    plt.xlim([0,5])
    plt.ylabel("Energy / Hartrees")
    plt.title("H$_2$ STO-3G $\sigma_g^2$ and $\sigma_u^2$ NOCI and SCF solution energies")
    plt.legend(loc="upper right", fontsize='x-large')

    plt.savefig("h2_sto3g_g2u2_energy_plot")
    plt.show()



plot_derivatives()
