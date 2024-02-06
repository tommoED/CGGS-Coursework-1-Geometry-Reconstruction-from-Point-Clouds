import os
import numpy as np


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces

def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=[], useOffPoints=True,
                        sparsify=False, l=-1):

    ##RBF computation is done in this function
    w=[]  #RBF weights
    RBFCentres=[] #RBF centres
    a=[] #polynomial coefficients (for Section 2)
    return w, RBFCentres, a


def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):
    values = np.zeros(xyz.shape[0])

    ###Complate RBF evaluation here
    return values

