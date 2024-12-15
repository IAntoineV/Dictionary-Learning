import torch

from tqdm import tqdm
from cuml import Lasso
from utils import proj_C

def algo2_batched_cv_check(D, A, B, steps=100, epsilon=1e-5, tol=1e-4):
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

def algo2_batched(D, A, B, steps=100, epsilon=1e-5, tol=1e-4):
    m, k = D.shape
    # D (m,k)
    # A (k,k)
    # B (m,k)
    for _ in range(steps):

        # Compute the updates for all columns simultaneously
        U = (B - D @ A) / (torch.diag(A).view(1, -1) +epsilon) + D
        D = proj_C(U)

    return D




def batched_algo1_gpu(x_loader, m, k, tmax, steps=3,lbd=0.001):
    A, B = torch.zeros(size=(k, k)), torch.zeros(size=(m, k))
    D = torch.randn(m, k)
    D = proj_C(D)

    for t in tqdm(range(tmax)):
        x_batched = next(x_loader)  # [batch_size, m]
        eta = len(x_batched) # eta = batch_size
        delta_A, delta_B = torch.zeros_like(A), torch.zeros_like(B)
        for x in x_batched:
            lasso = Lasso(
                alpha=lbd, fit_intercept=False
            )  # TODO: fix the issue of alpha = 0 whis proposed lambda
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

        D = algo2_batched(D, A, B, steps)

    return D