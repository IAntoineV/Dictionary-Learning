import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit


def patch_reconstruction(
    D, flatten_x, mean, std, l1_penalty, n_nonzero_coefs=10, omp=False
):
    """From a flatten patch, a dictionnary D and the mean and std of the original patch,
    builds an approximation of the patch using D."""
    if omp:
        omp_solver = OrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False
        )
        omp_solver.fit(X=D, y=flatten_x)
        alpha = torch.tensor(omp_solver.coef_, dtype=torch.float32)
    else:
        lasso = LassoLars(alpha=l1_penalty, fit_intercept=False)
        lasso.fit(X=D, y=flatten_x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)

    x_reconstructed_scaled = (
        torch.matmul(D.clone().detach(), alpha).clone().detach().numpy()
    )
    x_reconstructed = x_reconstructed_scaled * std + mean

    reconstructed_x = (
        torch.tensor(x_reconstructed).reshape(3, 12, 12).permute(1, 2, 0)
    )  # [H, W, C]

    return np.clip(np.array(reconstructed_x).astype(np.int32), 0, 255)


def visualize_patch_reconstruction(
    D,
    patch_x,
    flatten_x,
    l1_penalty,
    n_nonzero_coefs=10,
    omp=False,
    mean=None,
    std=None,
):
    """Plots the initial patch, the patch derived from the input, and the reconstruction from the latter.
    Also plots activations."""
    if not mean:
        mean = flatten_x.mean()
    if not std:
        std = flatten_x.std()

    std = std.item() if isinstance(std, torch.Tensor) else std
    mean = mean.item() if isinstance(mean, torch.Tensor) else mean

    if std > 0:
        normalized_x = (flatten_x - mean) / std
    else:
        normalized_x = flatten_x.clone()

    if omp:
        omp_solver = OrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False
        )
        omp_solver.fit(X=D, y=normalized_x)
        alpha = torch.tensor(omp_solver.coef_, dtype=torch.float32)
    else:
        lasso = LassoLars(alpha=l1_penalty, fit_intercept=False)
        lasso.fit(X=D, y=normalized_x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)

    x_reconstructed_scaled = torch.matmul(torch.tensor(D), alpha).numpy()
    x_reconstructed = x_reconstructed_scaled * std + mean  # De-normalize

    original_x = patch_x.permute(1, 2, 0)  # [H, W, C] from [C, H, W]
    scaled_x = normalized_x.reshape(3, 12, 12).permute(1, 2, 0)  # [H, W, C]
    # reconstructed_x = x_reconstructed.reshape(3, 16, 16).permute(1, 2, 0)  # [H, W, C]
    reconstructed_x = (
        torch.tensor(x_reconstructed).reshape(3, 12, 12).permute(1, 2, 0)
    )  # [H, W, C]

    original_rgb = (original_x - torch.min(original_x)) / (
        torch.max(original_x) - torch.min(original_x)
    )
    scaled_rgb = (scaled_x - torch.min(scaled_x)) / (
        torch.max(scaled_x) - torch.min(scaled_x)
    )
    reconstructed_rgb = (reconstructed_x - torch.min(reconstructed_x)) / (
        torch.max(reconstructed_x) - torch.min(reconstructed_x)
    )

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original X (RGB)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(scaled_rgb)
    plt.title("Scaled X (RGB)")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(reconstructed_rgb)
    plt.title("Reconstructed X (RGB)")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.bar(np.arange(len(alpha)), alpha.numpy())
    plt.title("Alpha Activations")
    plt.xlabel("Dictionary Index")
    plt.ylabel("Activation Magnitude")
    plt.tight_layout()

    plt.show()


def visualize_n_reconstruction(
    D, loader, l1_penalty, n=3, n_nonzero_coefs=10, omp=False
):
    for _ in range(n):
        patch_x, flatten_x = next(loader)
        patch_x, flatten_x = patch_x[0], flatten_x[0]

        visualize_patch_reconstruction(
            D, patch_x, flatten_x, l1_penalty, n_nonzero_coefs=n_nonzero_coefs, omp=omp
        )


def reconstrucion_metrics(D, loader, l1_penalty, omps=[1, 2, 5, 10, 20, 50], N=1000):
    """Prints mean squared error distances for reconstructions using given omp parameters and lars."""
    omps_metrics = {omp_val: [] for omp_val in omps}
    lars = []
    for _ in range(N):
        patch_x, flatten_x = next(loader)
        patch_x, flatten_x = patch_x[0], flatten_x[0]
        patch_x = patch_x.permute(1, 2, 0)
        patch_x = np.clip(np.array(255 * patch_x).astype(np.int32), 0, 255)
        mean = patch_x.mean()
        std = patch_x.std()
        for omp in omps_metrics.keys():
            reconstructed_patch = patch_reconstruction(
                D=D,
                flatten_x=flatten_x,
                mean=float(mean),
                std=float(std),
                l1_penalty=l1_penalty,
                omp=True,
                n_nonzero_coefs=omp,
            )
            reconstructed_patch = np.clip(
                np.array(reconstructed_patch).astype(np.int32), 0, 255
            )

            omps_metrics[omp].append(
                (patch_x - reconstructed_patch).astype(np.float64) ** 2
            )
        reconstructed_patch = patch_reconstruction(
            D=D,
            flatten_x=flatten_x,
            mean=mean,
            std=std,
            l1_penalty=l1_penalty,
            omp=False,
        )
        lars.append((patch_x - reconstructed_patch).astype(np.float64) ** 2)

    print("Mean reconstruction error:")
    for omp, errors in omps_metrics.items():
        print(f"OMP with {omp} components: {np.mean(errors)}")
    print(f"LARS: {np.mean(lars)}")
