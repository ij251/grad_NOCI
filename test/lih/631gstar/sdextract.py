import pickle
import numpy as np

def get_mo_coefficients(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    states = SD.StateList

    for gi in range((len(states)-1)/5):
        for si in states[gi].keys():

            np.savetxt("mo.coeffs.g{}.s{}.a.txt".format(gi*5, si),
                       states[gi*5][si][0].Coefficients[0])
            np.savetxt("mo.coeffs.g{}.s{}.b.txt".format(gi*5, si),
                       states[gi*5][si][0].Coefficients[1])


def get_noci_coefficients(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    states = SD.NOCIList

    for gi in range((len(states)-1)/5):

            np.savetxt("noci.coeffs.g{}.txt".format(gi*5),
                       states[gi*5][0].Eigenvectors[:,0])


def get_noci_energies(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    states = SD.NOCIList

    for gi in range((len(states)-1)/5):

            with open("noci.energy.g{}.txt".format(gi*5), "w+") as g:
                g.write(str(states[gi*5][0].Eigenvalues[0]))


def get_coords(sd_file):

    with open(sd_file) as f:
        D = pickle.load(f) #unpickle SD file for dictionary D

    SD = D['SysData']
    geom = SD.GeomList

    for gi in range((len(geom)-1)/5):
        for atom in range(2):

            np.savetxt("coords.g{}.atom{}.txt".format(gi*5, atom),
                       geom[gi*5].AllAtoms()[atom].AbsPosition())



get_noci_coefficients("noci.sd")
get_mo_coefficients("noci.sd")
get_noci_energies("noci.sd")
get_coords("noci.sd")
