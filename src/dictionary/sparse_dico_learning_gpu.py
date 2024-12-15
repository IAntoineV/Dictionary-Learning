import torch






def algo2_batched_cv_check(D, A, B, steps=100, epsilon=1e-5, tol=1e-4):
    m, k = D.shape
    # D (m,k)
    # A (k,k)
    # B (m,k)
    for _ in range(steps):
        D_prev = D.clone()

        # Compute the updates for all columns simultaneously
        U = (B - D @ A) / (torch.diag(A).view(1, -1) +epsilon) + D

        norms = torch.norm(U, p=2, dim=0, keepdim=True).clamp(min=1)
        D = U / norms

        # Check for convergence
        if torch.norm(D - D_prev, p='fro').item() < tol:
            break

    return D

def algo2_batched(D, A, B, steps=100, epsilon=1e-5, tol=1e-4):
    m, k = D.shape
    # D (m,k)
    # A (k,k)
    # B (m,k)
    for _ in range(steps):

        # Compute the updates for all columns simultaneously
        U = (B - D @ A) / (torch.diag(A).view(1, -1) +epsilon) + D

        norms = torch.norm(U, p=2, dim=0, keepdim=True).clamp(min=1)
        D = U / norms

    return D
