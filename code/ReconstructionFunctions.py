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

### ================================ GENERATE RBF CENTRES  ==========================================


def generate_rbf_centres_targets(surfacePoints, normals, indices, epsilon):
    if indices is None:
        return surfacePoints
    elif len(indices) > 0:
        perturbedSurfacePoints = surfacePoints[indices]
        indexedNormals = normals[indices]
    else:
        perturbedSurfacePoints = surfacePoints
        indexedNormals = normals
        
    
    RBFCentres = np.concatenate ((
        surfacePoints,
        perturbedSurfacePoints + epsilon * indexedNormals,
        perturbedSurfacePoints - epsilon * indexedNormals
    ))


    return RBFCentres

### ================================ RBF DISTANCE MATRICES ==========================================

def basic_RBF_matrix(surfacePoints, normals, kernel, epsilon):
    
    RBFCentres = generate_rbf_centres_targets(surfacePoints, normals, [], epsilon)

    assert kernel == polyharmonic or kernel == Wendland or kernel == biharmonic, "Invalid kernel function"


    RBFMatrix = np.zeros((RBFCentres.shape[0], surfacePoints.shape[0]))
    
    pairwise_distances = distance.cdist(RBFCentres, RBFCentres, 'euclidean')

    RBFMatrix = kernel(pairwise_distances)

    return RBFMatrix, RBFCentres

def reduced_RBF_matrix(surfacePoints, kernel, RBFCentreIndices, normals, epsilon):

    dataPoints = generate_rbf_centres_targets(surfacePoints, normals, [], epsilon)

    RBFCentres = generate_rbf_centres_targets(surfacePoints, normals, RBFCentreIndices, epsilon)

    RBFMatrix = np.zeros(
        (len(dataPoints), len(RBFCentres))
    )
    
    pairwise_distances = distance.cdist( dataPoints, RBFCentres, 'euclidean' )

    RBFMatrix = kernel(pairwise_distances)

    return RBFMatrix, RBFCentres


### ================================ POLYNOMIAL MATRIX GENERATION ======================================

def generate_polynomial_matrix(RBFCentres, degree, SPACE_DIM = 3):
    # Generate all indices up to the degree
    indices = np.arange(degree + 1)
    
    # Generate all combinations of indices where the sum is less than or equal to the degree
    indices_combinations = np.array(np.meshgrid(*[indices] * SPACE_DIM)).T.reshape(-1, SPACE_DIM)
    indices_combinations = indices_combinations[np.sum(indices_combinations, axis=1) <= degree]
    
    # Initialize Q with the correct shape
    Q = np.zeros((RBFCentres.shape[0], indices_combinations.shape[0]))
    
    # Compute the values for Q using broadcasting
    RBFCentres_bc = RBFCentres[:, np.newaxis, :]  # Add new axis for broadcasting
    indices_combinations_bc = indices_combinations[np.newaxis, :, :]  # Add new axis for broadcasting
    
    Q = np.prod(np.power(RBFCentres_bc, indices_combinations_bc), axis=2)
    
    return Q


### ================================ RBF WEIGHTS COMPUTATION ======================================


def compute_basic_weights(surfacePoints, target_vals, kernel, normals, epsilon):
    RBFMatrix, RBFCentres = basic_RBF_matrix(surfacePoints, normals, kernel, epsilon)

    LU_factorization = scipy.linalg.lu_factor(RBFMatrix)

    weights = scipy.linalg.lu_solve(LU_factorization, target_vals)


    return weights, RBFCentres, []


def compute_polynomial_weights(surfacePoints, normals, target_vals, kernel, degree, epsilon):

    RBFMatrix, RBFCentres = basic_RBF_matrix(surfacePoints, normals, kernel, epsilon)

    Q = generate_polynomial_matrix(RBFCentres, degree)
    

    assert RBFMatrix.shape[0] == Q.shape[0], "Q: {} RBFMatrix: {}".format(Q.shape, RBFMatrix.shape)

    # new system of equations:
        
    # | A   Q | w = b
    # | Q^T 0 | a = 0
    
    n_terms = Q.shape[1]
    zeroes = np.zeros((n_terms, n_terms))


    M = np.block([
        [RBFMatrix,       Q],
        [Q.T,        zeroes]
    ])

    target_vals = np.concatenate((
        target_vals,
        np.zeros(n_terms)
    ) , axis=0)


    LU_factorization = scipy.linalg.lu_factor(M)

    Solution = scipy.linalg.lu_solve(LU_factorization, target_vals)


    weights = Solution[:RBFCentres.shape[0] ]
    a       = Solution[ RBFCentres.shape[0]:]
    
    return weights, RBFCentres, a


def compute_reduced_weights(surfacePoints, target_vals, kernel, RBFCentreIndices, normals, epsilon):

    
    RBFMatrix, RBFCentres = reduced_RBF_matrix(surfacePoints, kernel, RBFCentreIndices, normals, epsilon)
    ### System is over-determined, so we use least squares

    weights, _, _, _ = np.linalg.lstsq(RBFMatrix, target_vals, rcond=None)

    return weights, RBFCentres, []


### ================================ RBF EVALUATION ======================================


def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=[], useOffPoints=True,
                        sparsify=False, l=-1):
    SPACE_DIM = 3
    

    target_vals = np.repeat([0, epsilon, -epsilon], inputPoints.shape[0])


    use_polynomial = l != -1

    if useOffPoints:
        if use_polynomial:
            return compute_polynomial_weights(inputPoints, inputNormals, target_vals, RBFFunction, l, epsilon )

        if len( RBFCentreIndices ) == 0:

            return compute_basic_weights(inputPoints, target_vals, RBFFunction, inputNormals, epsilon)

        return compute_reduced_weights(inputPoints, target_vals, RBFFunction, RBFCentreIndices, inputNormals, epsilon)


    return compute_reduced_weights(inputPoints, target_vals, RBFFunction, None, inputNormals, epsilon)
    
    


def evaluate_RBF(xyz, centres, RBFFunction, weights, l=-1, a=[]):
    """
    Evaluate the RBF function at the given points.
    """

    ## F(x) = Q*a + A*w


    ### determine 

    A = distance.cdist(xyz, centres, 'euclidean')

    values = np.dot(RBFFunction(A), weights)

    use_polynomial = l != -1

    if not use_polynomial:
        return values

    Q = generate_polynomial_matrix(xyz, l)

    values = values + np.dot(Q, a)

    return values
