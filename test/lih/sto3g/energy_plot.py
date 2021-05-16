
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
    plt.plot(x1, e1, label='SCF solution 1', c='c',
             lw='0', marker ='x', markevery = (0.0,0.05))
    plt.plot(x2, e2, label='SCF solution 2', c='r',
             lw='0.5',)
    # plt.plot(x_numerical, e1_numerical, label='Numerical NOCI Derivative', c='b',
    #          lw='0', marker = 'x', markevery = (0.0,0.05))

    plt.xlabel("Bond length / Bohr")
    plt.xlim([0,5])
    plt.ylabel("Energy / Hartrees")
    plt.title("LiH STO-3G NOCI and SCF solution energies")
    plt.legend(loc="upper right", fontsize='x-large')

    plt.savefig("lih_sto3g_2state_energy_plot")
    plt.show()



plot_derivatives()
