import os
import os.path as osp
import json
from PIL import Image
import numpy as np
from torch.utils.data.dataloader import DataLoader

import xml.etree.ElementTree as ET
from scipy.io import wavfile

from src.dataset.torch_dataset import FlattenedPatchDataset, NormalizationType


IMPLEMENTED_DATASETS = {
    "tuberculosis": "tuberculosis-phonecamera",
    "instruments": "instruments",
    "berkeley": "berkeley",
}


def get_data_path():
    cwd = os.getcwd()
    dir_to_check = cwd
    for _ in range(4):
        for root, dirs, _ in os.walk(dir_to_check):
            if "data" in dirs:
                return os.path.join(root, "data")
            if "dataset" in dirs:
                for subdir_root, subdirs, _ in os.walk(os.path.join(root, "dataset")):
                    if "data" in subdirs:
                        return os.path.join(subdir_root, "data")
                return None
        dir_to_check = os.path.dirname(dir_to_check)
    return None


def get_all_implemented_dataset():
    return list(IMPLEMENTED_DATASETS.keys())


def get_dataset(name, **kwargs):
    data_path = get_data_path()
    if data_path is None:
        assert False, "No data directory found in your repository, please create one under dataset directory"
    (
        os.path.exists(IMPLEMENTED_DATASETS["tuberculosis"]),
        f"download {name} dataset following DOWNLOAD_DATA.md \n\n {IMPLEMENTED_DATASETS['tuberculosis']} dir was not found",
    )
    path_dataset = os.path.join(data_path, IMPLEMENTED_DATASETS[name])
    if name == "tuberculosis":
        return get_tuberculosis_data(path_dataset)
    elif name == "instruments":
        return get_instruments_data(path_dataset)
    elif name == "berkeley":
        return get_berekley_data_loaders(path_dataset, **kwargs)
    assert False, f"No dataset named : {name} implemented"


def get_tuberculosis_data(path_dir):
    path_to_download = path_dir

    def load_jpg(path):
        image = Image.open(path)
        return np.array(image)

    def load_xml(path):
        tree = ET.parse(path)
        root = tree.getroot()
        # Accessing object details

        data = []
        for obj in root.findall("object"):
            label = obj.find("label").text
            pose = obj.find("pose").text
            truncated = obj.find("truncated").text
            occluded = obj.find("occluded").text

            # Access bounding box details
            bndbox = obj.find("bndbox")
            xmin = bndbox.find("xmin").text
            ymin = bndbox.find("ymin").text
            xmax = bndbox.find("xmax").text
            ymax = bndbox.find("ymax").text

            data.append((xmin, xmax, ymin, ymax, label))
        return data

    def get_data(path):
        file_names = os.listdir(path)
        data_jpg = {}
        data_xml = {}
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            file_without_ext = file_name.split(".")[0]
            if file_name.endswith(".jpg"):
                data_jpg[file_without_ext] = load_jpg(file_path)

            if file_name.endswith(".xml"):
                data_xml[file_without_ext] = load_xml(file_path)
            assert True, "Not supported format found in the directory"

        keys = list(data_jpg.keys())
        x = []
        y = []
        for key in keys:
            x.append(data_jpg[key])
            y.append(data_xml[key])
        return x, y, keys

    x, y, file_name = get_data(path_to_download)
    x = np.array(x)
    return x, y, file_name


def get_instruments_data(path_dir):
    audio_data = []
    sampling_rates = []
    files = []

    for root, _, filenames in os.walk(path_dir):
        for file in filenames:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    sr, y = wavfile.read(file_path)
                    audio_data.append(y)
                    sampling_rates.append(sr)
                    files.append(file_path)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return audio_data, sampling_rates, files


def get_berekley_data_loaders(
    path_dir, patch_size=(16, 16), patches_per_image=5, batch_size=32, norm_type=NormalizationType.L2, **kwargs
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dir = os.path.join(path_dir, "train/")
    val_dir = os.path.join(path_dir, "val/")
    test_dir = os.path.join(path_dir, "test/")

    train_dataset = FlattenedPatchDataset(
        train_dir, patch_size=patch_size, patches_per_image=patches_per_image, norm_type=norm_type
    )
    val_dataset = FlattenedPatchDataset(
        val_dir, patch_size=patch_size, patches_per_image=patches_per_image, norm_type=norm_type
    )
    test_dataset = FlattenedPatchDataset(
        test_dir, patch_size=patch_size, patches_per_image=patches_per_image, norm_type=norm_type
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




def parse_img_boat(data_unparsed):
    size = len(data_unparsed)
    assert size == 3 * 80 * 80, "Wrong data feeded"
    r_channels = data_unparsed[:size // 3]
    g_channels = data_unparsed[size // 3: 2*size//3]
    b_channels = data_unparsed[2*size//3:]
    pixels = np.array([r_channels, g_channels, b_channels], dtype=np.uint8).T
    pixels = pixels.reshape((80, 80, 3))
    return pixels


def get_boat_data(path_dir):
    jsondata = json.load(open(osp.join(path_dir, "shipsnet.json")))
    path_scenes = osp.join(path_dir,  "scenes/scenes/")
    file_names = os.listdir(path_scenes)
    images = [ Image.open(osp.join(path_scenes, file_name)).convert("RGB") for file_name in file_names ]

    y_true = np.array(jsondata["labels"])
    x = np.array(list(map(lambda x: parse_img_boat(x), jsondata["data"])))

    return x,y_true, images, jsondata







