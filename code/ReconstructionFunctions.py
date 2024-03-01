import os
import numpy as np
from scipy.spatial import distance
import scipy.linalg


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces

def polyharmonic(r, l = 3):
    return np.power(r, l)
    
def Wendland(r, beta=0.1):
    return (1/12) * np.power(np.maximum(1 - beta * r, 0), 3) * (1 - 3 * beta * r)

def biharmonic(r):
    return r


def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=[], useOffPoints=True,
                        sparsify=False, l=-1):
    SPACE_DIM = 3
        

    ## calculate RBF centres

    RBFCentres = np.concatenate ((
        inputPoints,
        inputPoints + epsilon * inputNormals,
        inputPoints - epsilon * inputNormals
    ))


    ## calculate RBF matrix A

    RBFMatrix = np.zeros((RBFCentres.shape[0], RBFCentres.shape[0]))
    
    pairwise_distances = distance.cdist(RBFCentres, RBFCentres, 'euclidean')


    RBFMatrix = RBFFunction(pairwise_distances)

    ## apply RBF Function to RBFMatrix
    ##  Solve Aw = b
            
    target_vals = np.repeat([0, epsilon, -epsilon], inputPoints.shape[0])

    # LU_factorization = scipy.linalg.lu_factor(RBFMatrix)

    # weights = scipy.linalg.lu_solve(LU_factorization, target_vals)

    
    ## calculate polynomial coefficients
    
    ## indices (integers < l)
    indices = np.arange(l+1)
    print (indices)
    ## produce all combinations of indices where the sum of the indices is less than or equal to l
    indices_combinations = np.array(np.meshgrid(*[indices]*SPACE_DIM)).T.reshape(-1, SPACE_DIM)

    indices_combinations = indices_combinations[indices_combinations.sum(axis=1) <= l]


    Q = np.zeros((RBFCentres.shape[0], indices_combinations.shape[0]))
    
    # Q[:, 0] = 1

    # for i, combination in enumerate(indices_combinations):
    #     Q[:, i] = np.prod(np.power(RBFCentres, combination), axis=1)

    RBFCentres_bc = RBFCentres[:, np.newaxis, :]
    indices_combinations_bc = indices_combinations[np.newaxis, :, :]


    Q = np.prod(
        np.power(RBFCentres_bc, indices_combinations_bc)
    , axis=2)

    assert RBFMatrix.shape[0] == Q.shape[0], "Q: {} RBFMatrix: {}".format(Q.shape, RBFMatrix.shape)

    # new system of equations:
        
    # | A   Q | w = b
    # | Q^T 0 | a = 0
    
    n_poly_terms = Q.shape[1]
    zeroes = np.zeros((n_poly_terms, n_poly_terms))

    M = np.block([
        [RBFMatrix,          Q],
        [Q.T,           zeroes]
    ])

    target_vals = np.concatenate((
        target_vals,
        np.zeros(Q.shape[1])
    ) , axis=0)

    LU_factorization = scipy.linalg.lu_factor(M)

    Solution = scipy.linalg.lu_solve(LU_factorization, target_vals)

    weights = Solution[:RBFCentres.shape[0]]
    a =       Solution[RBFCentres.shape[0]:]
    

    return weights, RBFCentres, a


def evaluate_RBF(xyz, centres, RBFFunction, weights, l=-1, a=[]):
    """
    Evaluate the RBF function at the given points.
    """
    

    distances = distance.cdist(xyz, centres, 'euclidean')

    values = np.dot(RBFFunction(distances), weights)

    if RBFFunction is polyharmonic:
        values += np.dot(xyz, a[0]) + a[1]


    return values
