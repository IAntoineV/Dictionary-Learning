import matplotlib.pyplot as plt

def plot_image_grid(data, k, figsize = (4,4)):
    """
    Plots a k x k grid of images
    """
    fig, axes = plt.subplots(k, k, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(data):
            ax.imshow(data[i])
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')

    plt.show()

