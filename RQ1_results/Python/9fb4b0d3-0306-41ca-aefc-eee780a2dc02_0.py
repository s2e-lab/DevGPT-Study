import numpy as np
import scipy.sparse

def max_nonzero_per_row(operators, cutoff):
    max_count = 0
    for matrix in operators:
        # Check if it's possible for this matrix to have a larger count
        if min(matrix.shape) - 1 <= max_count:
            continue
        
        # Check if the matrix is sparse
        if scipy.sparse.issparse(matrix):
            matrix = matrix.copy()  # Ensure we don't modify the original matrix
            matrix.data[np.abs(matrix.data) < cutoff] = 0  # Apply cutoff
            matrix.setdiag(0)  # Set diagonal entries to zero
            row_counts = matrix.getnnz(axis=1)  # Count non-zero entries in each row
        else:
            matrix = matrix.copy()  # Make a copy to avoid modifying the original matrix
            mask = np.abs(matrix) >= cutoff  # Create a mask of entries above the cutoff
            np.fill_diagonal(mask, 0)  # Set diagonal entries to zero
            row_counts = np.count_nonzero(mask, axis=1)  # Count non-zero entries in each row
        max_count = max(max_count, np.max(row_counts))  # Update max count if necessary
    return max_count
