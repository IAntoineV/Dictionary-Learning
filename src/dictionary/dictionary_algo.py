

import torch
from dictionary_base import DictionaryBase

class DictionaryAlgoBasic(DictionaryBase):

    def __init__(self, m, k, dico_update = "quick_update", use_cuda=False):
        super().__init__(m, k, dico_update = dico_update)
        if use_cuda:
            from cuml import Lasso
            self.lasso_function = Lasso
        else:
            from sklearn.linear_model import Lasso
            self.lasso_function = Lasso
    def fit_data(self, x, lbd = 0.001):

        lasso = self.lasso_function(
            alpha=lbd, fit_intercept=False
        )
        lasso.fit(X=self.D, y=x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)
        self.A += torch.outer(alpha, alpha)
        self.B += torch.outer(x, alpha)



class DictionaryAlgoParallel(DictionaryAlgoBasic):
    def __init__(self, m, k, dico_update = "quick_update", use_cuda=False, num_workers=4):
        super().__init__(m, k, dico_update = dico_update, use_cuda=use_cuda)


    def fit_data(self, x_batched, lbd = 0.001):

        eta = len(x_batched)
        delta_A, delta_B = torch.zeros_like(self.A), torch.zeros_like(self.B)
        for x in x_batched:
            lasso = self.lasso_function(
                alpha=lbd, fit_intercept=False
            )
            lasso.fit(X=self.D, y=x)
            alpha = torch.tensor(lasso.coef_, dtype=torch.float32)

            delta_A += torch.outer(alpha, alpha)
            delta_B += torch.outer(x, alpha)

        if self.t < eta:
            theta = self.t * eta
        else:
            theta = eta ** 2 + self.t - eta

        beta = (theta + 1 - eta) / (theta + 1)

        self.A = beta * self.A + delta_A
        self.B = beta * self.B + delta_B

        self.update_dictionary()
