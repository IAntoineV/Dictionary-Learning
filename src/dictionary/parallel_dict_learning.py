import torch
from torch.multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import StandardScaler


def proj_C(D, k):
    """Projection on the subspace of matrix with columns of l_2 norm <= 1."""
    for j in range(k):
        norm = torch.norm(D[:, j], p=2)
        if norm > 1:
            D[:, j] = D[:, j] / norm
    return D

def process_patches(batch, D, scaler, k, m):
    delta_A, delta_B = torch.zeros(k, k), torch.zeros(m, k)

    for x_patch in batch:
        x = torch.flatten(x_patch)  # [c*p_h*p_w] = [m]
        x = torch.tensor(scaler.fit_transform(x.numpy().reshape(-1, 1)).flatten())

        lasso = LassoLars(
            alpha=0.001, fit_intercept=False
        )  #  TODO: fix the issue of alpha = 0 whis proposed lambda
        lasso.fit(X=D, y=x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)

        delta_A += torch.outer(alpha, alpha)
        delta_B += torch.outer(x, alpha)

    return delta_A, delta_B


def algo2(D, A, B, steps, epsilon=1e-5):
    for _ in range(steps):
        for j in range(D.size(1)):
            u_j = (1 / (A[j, j] + epsilon)) * (B[:, j] - torch.matmul(D, A[:, j])) + D[
                :, j
            ]
            norm_u_j = torch.norm(u_j, p=2)
            D[:, j] = (1 / max(norm_u_j, 1)) * u_j

    return D


def algo1(x_loader, m, k, lbd, tmax, num_workers=2, steps=3):
    A, B = torch.zeros(size=(k, k)), torch.zeros(size=(m, k))
    D = torch.randn(m, k)
    D = proj_C(D, k)

    scaler = StandardScaler()

    for t in tqdm(range(tmax)):
        x_patches = next(x_loader)  # [c, p_h, p_w]
        eta = len(x_patches)
        
        chunks = torch.chunk(x_patches, num_workers)
        with Pool(num_workers) as pool:
            results = pool.starmap(
                process_patches, [(chunk, D, scaler, k, m) for chunk in chunks]
            )
        
        delta_A = sum(res[0] for res in results)
        delta_B = sum(res[1] for res in results)
        
        if t < eta:
            theta = t*eta
        else:
            theta = eta**2 + t - eta

        beta = (theta + 1 - eta)/(theta + 1)

        A = beta*A + delta_A
        B = beta*B + delta_B

        D = algo2(D, A, B, steps)

    return D
