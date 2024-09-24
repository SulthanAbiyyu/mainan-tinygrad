from tinygrad import Tensor

PI = 3.141592653589793

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.glorot_uniform(in_features, out_features)
        self.bias = Tensor.glorot_uniform(out_features) if bias else None
    
    def __call__(self, x):
        if self.bias is None:
            return x @ self.weight
        return x @ self.weight + self.bias
    
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor.kaiming_normal(num_embeddings, embedding_dim)
    
    def __call__(self, x):
        return self.weight[x]
    
class GeLU:
    # Tanh approximation
    def __call__(self, x):
        return 0.5 * x * (1 + ((x + 0.044715 * x ** 3) * (2 / PI)**0.5).tanh())

class RMSNorm:
    def __init__(self, dim, epsilon=1e-8):
        self.dim = dim
        self.epsilon = epsilon
        self.scale = Tensor([1.0])
    
    def __call__(self, x):
        return x / ((x**2).mean(self.dim, keepdim=True) + self.epsilon).sqrt() * self.scale
    
class Sequential:
    def __init__(self, modules: list):
        self.modules = modules
    def __call__(self, x):
        for m in self.modules:
            x = m(x)
        return x