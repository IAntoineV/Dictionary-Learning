import torch
from src.dictionary.dictionary_update import (
    dico_update,
    dico_update_batched,
    dico_update_batched_cv_check,
)

from src.dictionary.utils import proj_C

IMPLEMENTED_DICTIONARY_UPDATE = {
    "basic_update": dico_update,
    "quick_update": dico_update_batched,
    "quick_update_cv_check": dico_update_batched_cv_check,
}


class DictionaryBase:
    def __init__(self, m, k, lbd, dico_update="quick_update", dic_update_steps=100):
        self.D = torch.randn(m, k)
        self.D = proj_C(self.D)
        self.A = torch.zeros(size=(k, k))
        self.B = torch.zeros(size=(m, k))
        self.m = m
        self.k = k
        self.lbd = lbd
        self.dico_update = dico_update
        self.dic_update_steps = dic_update_steps
        self.t = 0

    def fit(self, iterator):
        """
        :param iterator: Iterable containing batch or single example data.
        :return:
        """
        for x in iterator:
            self.fit_data(x)
            self.t += 1

    def fit_data(self, x, **kwargs):
        """
        Fit a data or a batch of data to our dictionary.
        """
        raise NotImplementedError



    def update_dictionary(self, **kwargs):
        """
        Update the dictionary based on the current values of A and B. You may check the 2009's paper :

        Online Dictionary Learning for Sparse Coding (2009).
        """
        update_implementation = IMPLEMENTED_DICTIONARY_UPDATE[self.dico_update]
        return update_implementation(self.D, self.A, self.B, self.dic_update_steps, **kwargs)
