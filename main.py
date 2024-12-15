import numpy as np
from tqdm import tqdm
import torch
from itertools import cycle
from sklearn.preprocessing import StandardScaler

from src.dataset.dataloaders import get_dataset
from src.dictionary.basic_dictionary_learning import batched_algo1 as algo1
from src.dictionary.dictionary_algo import DictionaryAlgoBasic, DictionaryAlgoParallel


def main():
    train_loader, val_loader, test_loader = get_dataset(
        "berkeley", patch_size=(16, 16), patches_per_image=5, batch_size=1
    )

    infinite_x_loader = cycle(train_loader)
    m = 16 * 16 * 3

    # algo1(infinite_x_loader, m=m, k=200, lbd=1.2/np.sqrt(m), tmax=int(1e4))

    DictLearner = DictionaryAlgoParallel(
        m=m, k=200, lbd=1.2 / (m * np.sqrt(m)), dic_update_steps=3
    )

    DictLearner.fit(tmax=int(1e4), iterable=infinite_x_loader)


if __name__ == "__main__":
    main()
