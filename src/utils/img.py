import numpy as np

def get_augmentation():
    def id(x, **kwargs):
        return x

    def flip_h(x):
        return x[:,:,::-1]

    def flip_v(x):
        return x[:,::-1]

    def rotate_90(x):
        return np.rot90(x, k=1, axes = (2,1))

    def rotate_180(x):
        return np.rot90(x, k=-1, axes = (2,1))

    def rotate_270(x):
        return np.rot90(x, k=2, axes = (2,1))

    dico = {"original": id, "flip_h": flip_h, "flip_v": flip_v, "rotate_90": rotate_90, "rotate_180": rotate_180,
            "rotate_270": rotate_270}

    return dico


def augment_image(dataset, augment_list=None):
    """
    Apply augmentation inplace to a dataset.
    """
    if augment_list is None:
        augment_list = ["original", "flip_h", "flip_v", "rotate_90", "rotate_270", "rotate_180"]

    augment_dico_functions = get_augmentation()
    n_data = len(dataset)
    nb_augment = len(augment_list)
    choose_augment = np.random.randint(nb_augment, size=n_data)
    for i in range(nb_augment):
        numpy_batched_fun = augment_dico_functions[augment_list[i]]
        dataset[choose_augment == i] = numpy_batched_fun(dataset[choose_augment == i])
    return dataset