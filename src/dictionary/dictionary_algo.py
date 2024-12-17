import torch
from joblib import Parallel, delayed
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from src.dictionary.dictionary_base import DictionaryBase

class DictionaryAlgoBasic(DictionaryBase):

    def __init__(self, m, k, lbd, dico_update = "quick_update", dic_update_steps=100, use_cuda=False):
        super().__init__(m, k, lbd, dico_update = dico_update, dic_update_steps=dic_update_steps)
        if use_cuda:
            from cuml import Lasso
            self.lasso_function = Lasso
            self.device = "cuda"
            if not torch.cuda.is_available():
                raise "No GPU Found in torch. Turn off GPU using with Dictionary learning."

        else:
            from sklearn.linear_model import LassoLars
            self.lasso_function = LassoLars
            self.device = "cpu"
        self.D = self.D.to(self.device)
        self.A = self.A.to(self.device)
        self.B = self.B.to(self.device)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_data(self, x):

        lasso = self.lasso_function(
            alpha=self.lbd, fit_intercept=False
        )
        lasso.fit(X=self.D, y=x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)
        self.A += torch.outer(alpha, alpha)
        self.B += torch.outer(x, alpha)

    @ignore_warnings(category=ConvergenceWarning)
    def get_A_B(self, x):
        lasso = self.lasso_function(
            alpha=self.lbd, fit_intercept=False
        )
        lasso.fit(X=self.D, y=x)
        alpha = torch.tensor(lasso.coef_, dtype=torch.float32)

        delta_A = torch.outer(alpha, alpha)
        delta_B = torch.outer(x, alpha)
        return delta_A, delta_B


class DictionaryAlgoParallel(DictionaryAlgoBasic):
    def __init__(self, m, k, lbd, dico_update = "quick_update", dic_update_steps=100, use_cuda=False, n_jobs=4):
        super().__init__(m, k, lbd,dico_update = dico_update, dic_update_steps=dic_update_steps, use_cuda=use_cuda)
        self.n_jobs = n_jobs

    def fit_data(self, x_batch, ):
        x_batch = x_batch.to(self.device)
        eta = len(x_batch)

        results = Parallel(n_jobs=self.n_jobs)(delayed(self.get_A_B)(x) for x in x_batch)
        delta_A,delta_B = zip(*results)
        delta_A = sum(delta_A)
        delta_B = sum(delta_B)

        if self.t < eta:
            theta = self.t*eta
        else:
            theta = eta**2 + self.t - eta

        beta = (theta + 1 - eta)/(theta + 1)

        self.A = beta*self.A + delta_A
        self.B = beta*self.B + delta_B

