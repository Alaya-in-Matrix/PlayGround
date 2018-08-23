import autograd.numpy as np
from .GP_model import GP_model


class Bagging_GP_model:
    def __init__(self, num_models, train_x, train_y, layer_sizes, activations, bfgs_iter=100, l1=0, l2=0, debug=True):
        self.num_models = num_models
        self.model = []
        for i in range(self.num_models):
            self.model.append(GP_model(train_x, train_y, layer_sizes, activations, bfgs_iter=bfgs_iter, l1=l1, l2=l2, debug=True))

    def fit(self, scale=1.0):
        for i in range(self.num_models):
            theta0 = self.model[i].rand_theta(scale=scale)
            self.model[i].fit(theta0)

    def predict(self, test_x):
        py = np.zeros((test_x.shape[1]))
        ps2 = np.zeros((test_x.shape[1], test_x.shape[1]))
        for i in range(self.num_models):
            tmp_py, tmp_ps2 = self.model[i].predict(test_x)
            py = py + tmp_py/self.num_models
            ps2 = ps2 + (tmp_ps2 + np.square(tmp_py))/self.num_models
        ps2 -= np.square(py)
        return py, ps2












