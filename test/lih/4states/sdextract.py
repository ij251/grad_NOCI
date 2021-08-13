import pickle
import numpy as np

def get_mo_coefficients(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    states = SD.StateList

    for gi in range(len(states)):
        for si in states[gi].keys():

            np.savetxt("mo.coeffs.g{}.s{}.a.txt".format(gi, si),
                       states[gi][si][0].Coefficients[0])
            np.savetxt("mo.coeffs.g{}.s{}.b.txt".format(gi, si),
                       states[gi][si][0].Coefficients[1])


def get_noci_coefficients(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    states = SD.NOCIList

    for gi in range(len(states)):

            np.savetxt("noci.coeffs.g{}.txt".format(gi),
                       states[gi][0].Eigenvectors[:,0])


def get_noci_energies(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    states = SD.NOCIList

    for gi in range(len(states)):

            with open("noci.energy.g{}.txt".format(gi), "w+") as g:
                g.write(str(states[gi][0].Eigenvalues[0]))


def get_coords(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    geom = SD.GeomList

    for gi in range(len(geom)):
        for atom in range(len(geom[0])):

            np.savetxt("coords.g{}.atom{}.txt".format(gi, atom),
                       geom[gi].AllAtoms()[atom].AbsPosition())



get_noci_coefficients("noci.sd")
get_mo_coefficients("noci.sd")
# get_noci_energies("noci.sd")
get_coords("noci.sd")
