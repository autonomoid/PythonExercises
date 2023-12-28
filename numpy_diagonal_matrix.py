import numpy as np

np.set_printoptions(precision=2)

def matrix_of_cofactors(A):
    n = A.shape[0]
    cofactors = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A, i, 0), j, 1)  # Get the minor matrix.
            cofactors[i, j] = (-1)**(i+j) * np.linalg.det(minor)  # Apply cofactor sign.

    return cofactors

def matrix_inverse(A):
    # Compute the adjugate of the matrix.
    A_adjugate = matrix_of_cofactors(A).T

    # Compute the determinant of the matrix.
    determinant_of_A = np.linalg.det(A)

    # Return the inverse of the matrix.
    return A_adjugate / determinant_of_A

def compute_diagonalized_matrix(eigenvectors, eigenvalues):
    # Construct the diagonal matrix D.
    D = np.diag(eigenvalues)

    # Construct the eigenvector matrix P.
    P = np.column_stack(eigenvectors)

    # Compute the inverse of P.
    P_inv = matrix_inverse(P)

    # Return the diagonalized matrix A from the eigendecomposition.
    A = P @ D @ P_inv
    return A

# Given these eigenvectors
eigenvectors = [(1, 1, 0), (1, -1, 1), (-1, 1, 2)]

# and these eigenvalues
eigenvalues = [4, 1, 2]

# Compute the diagonalized matrix A.
A = compute_diagonalized_matrix(eigenvectors, eigenvalues)
print(f"A = {A}")

# Test the matrix on the eigenvectors.
for index, eigenvector in enumerate(eigenvectors):
    print(f"A x V{index+1} = {A @ eigenvector}")
