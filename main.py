import numpy as np
from itertools import cycle

from src.dataset.dataloaders import get_dataset

from src.dictionary.basic_dictionary_learning import algo1

def main():
    train_loader, val_loader, test_loader = get_dataset("berkeley",  patch_size=(16, 16), patches_per_image=5, batch_size=1)

    infinite_x_loader = cycle(train_loader)
    m = 16*16*3
    algo1(infinite_x_loader, m=m, k=200, lbd=1.2/np.sqrt(m), tmax=100)

if __name__ == "__main__":
    main()