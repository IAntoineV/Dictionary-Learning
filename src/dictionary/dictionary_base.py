import torch
import os
import numpy as np
from typing_extensions import override

from src.dictionary.dictionary_update import (
    dico_update,
    dico_update_batched,
    dico_update_batched_cv_check,
)
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from src.dictionary.utils import proj_C

IMPLEMENTED_DICTIONARY_UPDATE = {
    "basic_update": dico_update,
    "quick_update": dico_update_batched,
    "quick_update_cv_check": dico_update_batched_cv_check,
}


class DictionaryBase:
    def __init__(self, m, k, lbd, dico_update="quick_update", dic_update_steps=100, verbose=False):
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
        self.save_path = "./dicolearning_exp/"
        os.makedirs(self.save_path, exist_ok=True)
        self.nb_data_seen = 0
        self.verbose=verbose
        self.device = self.D.device
    @property
    def components_(self):
        return np.array(self.D).T
    def fit(self, iterator, nb_save = 10, name="exp"):
        """
        :param iterator: Iterable containing batch or single example data.
        :return:
        """
        total_iterations = len(iterator)


        for x in iterator:
            self.training_step(x)
            self.t += 1
            if self.t % (total_iterations // nb_save) == 0:
                self.save(self.save_path + f" {name}_{100 * (self.t // (total_iterations // nb_save)) / nb_save}%")

    def training_step(self, x):
        self.fit_data(x)
        self.update_dictionary()


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
        self.D = update_implementation(self.D, self.A, self.B, self.dic_update_steps, **kwargs)

    def save(self, path):
        torch.save(self.D, path)

    def load(self, path):
        self.D = torch.load(path)

    def transform(self,X, n_nonzero_coefs=50):
        coefs =[]
        for x in X:
            omp_solver = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False)
            omp_solver.fit(X=self.D, y=x)
            alpha = omp_solver.coef_
            coefs.append(alpha)

        return np.array(coefs)



