import torch
from tqdm import tqdm

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from src.dictionary.dictionary_base import IMPLEMENTED_DICTIONARY_UPDATE
from src.dictionary.dictionary_base import DictionaryBase

class DictionaryAlgoBasic(DictionaryBase):

    def __init__(self, m, k, lbd, dico_update = "quick_update", dic_update_steps=100, use_cuda=False):
        super().__init__(m, k, lbd, dico_update = dico_update, dic_update_steps=dic_update_steps)
        if use_cuda:
            from cuml import Lasso
            self.lasso_function = Lasso
        else:
            from sklearn.linear_model import LassoLars
            self.lasso_function = LassoLars

    def fit_data(self, x):

        lasso = self.lasso_function(
            alpha=self.lbd, fit_intercept=False
        )
        lasso.fit(X=self.D, y=x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)
        self.A += torch.outer(alpha, alpha)
        self.B += torch.outer(x, alpha)



class DictionaryAlgoParallel(DictionaryAlgoBasic):
    def __init__(self, m, k, lbd, dico_update = "quick_update", dic_update_steps=100, use_cuda=False, num_workers=4):
        super().__init__(m, k, lbd,dico_update = dico_update, dic_update_steps=dic_update_steps, use_cuda=use_cuda)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, iterable, tmax):
        A, B = self.A, self.B
        D = self.D

        for t in tqdm(range(self.t, self.t + tmax)):
            x_patches = next(iterable)  # [c, p_h, p_w]
            eta = len(x_patches)
            delta_A, delta_B = torch.zeros_like(A), torch.zeros_like(B)

            for x in x_patches:

                lasso = self.lasso_function(
                    alpha=self.lbd, fit_intercept=False
                )
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

            D = IMPLEMENTED_DICTIONARY_UPDATE[self.dico_update](D, A, B, self.dic_update_steps)

        self.D = D
        self.A = A
        self.B = B
