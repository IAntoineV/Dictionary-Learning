import torch

from src.dictionary.utils import proj_C

def dico_update_batched_cv_check(D, A, B, steps=100, epsilon=1e-5, tol=1e-4):
    """
    Algo2 from paper Online Dictionary Learning for Sparse Coding (2009).

    Check for Convergence of the update scheme like the pseudocode provided in the paper.
    """
    m, k = D.shape
    # D (m,k)
    # A (k,k)
    # B (m,k)
    for _ in range(steps):
        D_prev = D.clone()

        # Compute the updates for all columns simultaneously
        U = (B - D @ A) / (torch.diag(A).view(1, -1) +epsilon) + D

        D = proj_C(U)

        # Check for convergence
        if torch.norm(D - D_prev, p='fro').item() < tol:
            break

    return D

def dico_update_batched(D, A, B, steps=100, epsilon=1e-5, tol=1e-4):
    """
    Algo2 from paper Online Dictionary Learning for Sparse Coding (2009).

    Do not check for Convergence of the update scheme to avoid dictionary cloning at each step.
    """
    m, k = D.shape
    # D (m,k)
    # A (k,k)
    # B (m,k)
    for _ in range(steps):
        # Compute the updates for all columns simultaneously
        U = (B - D @ A) / (torch.diag(A).view(1, -1) +epsilon) + D
        D = proj_C(U)

    return D

def dico_update(D, A, B, steps=100, epsilon=1e-5):
    """
    Algo2 from paper Online Dictionary Learning for Sparse Coding (2009).

    basic implementation with loops like in the paper.

    Do not check for Convergence of the update scheme to avoid dictionary cloning at each step.
    """
    for _ in range(steps):
        for j in range(D.size(1)):
            u_j = (1 / (A[j, j] + epsilon)) * (B[:, j] - torch.matmul(D, A[:, j])) + D[
                :, j
            ]
            norm_u_j = torch.norm(u_j, p=2)
            D[:, j] = (1 / max(norm_u_j, 1)) * u_j

    return D