import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from src.dictionary.dictionary_update import dico_update_batched


def proj_C(D, k):
    """Projection on the subspace of matrix with columns of l_2 norm <= 1."""
    for j in range(k):
        norm = torch.norm(D[:, j], p=2)
        if norm > 1:
            D[:, j] = D[:, j] / norm
    return D


def algo2(D, A, B, steps, epsilon=1e-5):
    for _ in range(steps):
        for j in range(D.size(1)):
            u_j = (1 / (A[j, j] + epsilon)) * (B[:, j] - torch.matmul(D, A[:, j])) + D[
                :, j
            ]
            norm_u_j = torch.norm(u_j, p=2)
            D[:, j] = (1 / max(norm_u_j, 1)) * u_j

    return D

@ignore_warnings(category=ConvergenceWarning)
def batched_algo1(x_loader, m, k, tmax, steps=3, lbd=0.001):
    A, B = torch.zeros(size=(k, k)), torch.zeros(size=(m, k))
    D = torch.randn(m, k)
    D = proj_C(D, k)

    # scaler = StandardScaler()

    for t in tqdm(range(tmax)):
        x_patches = next(x_loader)  # [c, p_h, p_w]
        eta = len(x_patches)
        delta_A, delta_B = torch.zeros_like(A), torch.zeros_like(B)
        for x in x_patches:
            # x = torch.flatten(x_patch)  # [c*p_h*p_w] = [m]
            # x = torch.tensor(scaler.fit_transform(x.numpy().reshape(-1, 1)).flatten())

            lasso = LassoLars(
                alpha=lbd/m, fit_intercept=False
            )
            lasso.fit(X=D, y=x)
            alpha = torch.tensor(lasso.coef_, dtype=torch.float32)

            delta_A += torch.outer(alpha, alpha)
            delta_B += torch.outer(x, alpha)

        if t < eta:
            theta = t*eta
        else:
            theta = eta**2 + t - eta

        beta = (theta + 1 - eta)/(theta + 1)

        A = beta*A + delta_A
        B = beta*B + delta_B

        # D = algo2(D, A, B, steps)
        D = dico_update_batched(D, A, B, steps)

    return D

@ignore_warnings(category=ConvergenceWarning)
def base_algo1(x_loader, m, k, tmax, steps=3,  lbd=0.001):
    A, B = torch.zeros(size=(k, k)), torch.zeros(size=(m, k))
    D = torch.randn(m, k)
    D = proj_C(D, k)

    scaler = StandardScaler()

    for _ in tqdm(range(tmax)):
        patch_x = next(x_loader)[0]  # [c, p_h, p_w]
        x = torch.flatten(patch_x)  # [c*p_h*p_w] = [m]
        x = torch.tensor(scaler.fit_transform(x.numpy().reshape(-1, 1)).flatten())

        lasso = LassoLars(
            alpha=lbd/m, fit_intercept=False
        ) 
        lasso.fit(X=D, y=x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)

        A += torch.outer(alpha, alpha)
        B += torch.outer(x, alpha)

        # all_coeffs_magnitude = torch.abs(A).mean()
        # diagonal_coeffs = torch.diag(A)
        # diagonal_coeffs_magnitude = torch.abs(diagonal_coeffs).mean()

        # print("Avg norm of A's coeffs:", all_coeffs_magnitude.item())
        # print(
        #     "Avg norm of A's diagonal coeffs:", diagonal_coeffs_magnitude.item(), "\n"
        # )

        D = algo2(D, A, B, steps)

    return D