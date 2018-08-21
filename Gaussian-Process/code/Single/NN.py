import autograd.numpy as np

## fully-connected neural network

class NN:
    def __init__(self, layer_sizes, activations):
        self.num_layers = np.copy(len(layer_sizes))
        self.layer_sizes = np.copy(layer_sizes)
        self.activations = activations

    def num_param(self, dim):
        '''
        get the parameter number of the neural network
        
        dim: input vector size
        layer_sizes only contains size of weights
        remember to take bias into consideration
        '''
        xs = [dim]
        results = 0
        for l in self.layer_sizes:
            xs.append(l)
        for i in range(self.num_layers):
            results += (1+xs[i])*xs[i+1]
        return results

    def w_nobias(self, w, dim):
        '''
        get the weights matrix for l1/l2 regularization

        w: weights + bias
        dim: input vector size

        w_layer: only contains weights, without bias
        '''
        prev_size = dim
        start_idx = 0
        wnb = np.array([])
        for i in range(self.num_layers):
            layer_size = self.layer_sizes[i]
            w_num_layer = (1+prev_size)*layer_size
            w_layer = np.reshape(w[start_idx:start_idx+w_num_layer],(prev_size+1,layer_size))[:prev_size]
            wnb = np.concatenate((wnb, w_layer.reshape(w_layer.size)))
            start_idx += w_num_layer
            prev_size = layer_size
        return wnb

    def predict(self, w, x):
        '''
        get the prediction results

        x.shape: dim, num_train
        w_layer.shape: dim, next_layer_dim
        out.shape: dim, num_train
        '''
        dim, num_train = x.shape
        out = x
        start_idx = 0
        prev_size = dim
        bias = np.ones((1, num_train))
        for i in range(self.num_layers):
            layer_size = self.layer_sizes[i]
            w_num_layer = (prev_size+1)*layer_size
            w_layer = np.reshape(w[start_idx:start_idx+w_num_layer], (prev_size+1,layer_size))
            out = np.concatenate((out, bias))
            out = self.activations[i](np.dot(w_layer.T, out))
            start_idx += w_num_layer
            prev_size = layer_size
        return out

