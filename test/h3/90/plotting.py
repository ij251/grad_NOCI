import numpy as np
import matplotlib.pyplot as plt

def plot_derivatives():

    analytic = np.loadtxt("analytic_derivatives.txt")
    numerical = np.loadtxt("numerical_derivatives.txt")


    x_analytic = analytic[:,0]
    x_numerical = numerical[:,0]

    e1_analytic = analytic[:,1]
    e1_numerical = numerical[:,1]

    plt.plot(x_analytic, e1_analytic, label='Analytic NOCI Derivative', c='r',
             lw='0.5',)
    plt.plot(x_numerical, e1_numerical, label='Numerical NOCI Derivative', c='b',
             lw='0', marker = 'x', markevery = (0.0,0.05))

    plt.xlabel("Bond length of one H-H bond / Bohr")
    plt.xlim([0,5])
    plt.ylabel("Energy derivative / Hartrees Bohr$^{-1}$")
    plt.title("H$_3$ STO-3G 90 Degree Bond Angle NOCI derivative")
    plt.legend(loc="lower right", fontsize='x-large')
    plt.axhline(y=0, lw='0.5', c='k')

    plt.savefig("h3_90_plot")
    plt.show()



plot_derivatives()
