import sklearn.preprocessing as pre

# axis = 0
class Normalizer(pre.Normalizer):
    def __init__(self, norm="l2", *, copy=True):
        super(Normalizer, self).__init__(norm = norm, copy = copy)

    def transform(self, X, copy=None):

        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        return pre.normalize(X, norm=self.norm, axis=0, copy=copy)



# Normalizer(norm="max")