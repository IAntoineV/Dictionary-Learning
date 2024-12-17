import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_augmentation():
    def id(x, **kwargs):
        return x

    def flip_h(x):
        return x[:, :, ::-1]

    def flip_v(x):
        return x[:, ::-1]

    def rotate_90(x):
        return np.rot90(x, k=1, axes=(2, 1))

    def rotate_180(x):
        return np.rot90(x, k=-1, axes=(2, 1))

    def rotate_270(x):
        return np.rot90(x, k=2, axes=(2, 1))

    dico = {
        "original": id,
        "flip_h": flip_h,
        "flip_v": flip_v,
        "rotate_90": rotate_90,
        "rotate_180": rotate_180,
        "rotate_270": rotate_270,
    }

    return dico


def augment_image(dataset, augment_list=None):
    """
    Apply augmentation inplace to a dataset.
    """
    if augment_list is None:
        augment_list = [
            "original",
            "flip_h",
            "flip_v",
            "rotate_90",
            "rotate_270",
            "rotate_180",
        ]

    augment_dico_functions = get_augmentation()
    n_data = len(dataset)
    nb_augment = len(augment_list)
    choose_augment = np.random.randint(nb_augment, size=n_data)
    for i in range(nb_augment):
        numpy_batched_fun = augment_dico_functions[augment_list[i]]
        dataset[choose_augment == i] = numpy_batched_fun(dataset[choose_augment == i])
    return dataset


def add_gaussian_noise(img, mean=0, std=25):
    """
    Adds a gaussian noise to a PIL image
    """
    img_np = np.array(img)
    noise = np.random.normal(mean, std, img_np.shape)
    noisy_img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img_np)


loremipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor. Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi. Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, non fermentum diam nisl sit amet erat. Duis semper. Duis arcu massa, scelerisque vitae, consequat in, pretium a, enim. Pellentesque congue. Ut in risus volutpat libero pharetra tempor. Cras vestibulum bibendum augue. Praesent egestas leo in pede. Praesent blandit odio eu enim. Pellentesque sed dui ut augue blandit sodales. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Aliquam nibh. Mauris ac mauris sed pede pellentesque fermentum. Maecenas adipiscing ante non diam sodales hendrerit."


def add_text_noise(
    img, text=loremipsum, font_size=20, color=(255, 0, 0)
):
    """
    Add text overlay as noise to a PIL image.
    """
    img_with_text = img.copy()
    draw = ImageDraw.Draw(img_with_text)
    font = ImageFont.load_default()

    width, height = img_with_text.size
    
    for y in range(0, height, font_size):
        for x in range(0, width, len(text) * font_size // 2):
            draw.text((x, y), text, font=font, fill=color)
    return img_with_text
