
import torch


def proj_C(D):
    """Projection on the subspace of matrix with columns of l_2 norm <= 1."""
    norms = torch.norm(D, p=2, dim=0, keepdim=True).clamp(min=1)
    D = D / norms
    return D

