import pytest
from tinygrad import Tensor

from nn import Linear, Embedding, GeLU, RMSNorm, Sequential, PI

@pytest.fixture
def setup_linear():
    return Linear(in_features=4, out_features=2)

@pytest.fixture
def setup_embedding():
    return Embedding(num_embeddings=5, embedding_dim=3)

@pytest.fixture
def setup_gelu():
    return GeLU()

@pytest.fixture
def setup_rmsnorm():
    return RMSNorm(dim=1)

@pytest.fixture
def setup_sequential(setup_linear, setup_embedding, setup_gelu):
    return Sequential([setup_embedding, setup_linear, setup_gelu])

def test_linear_forward(setup_linear):
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    output = setup_linear(x)
    assert output.shape == (1, 2), "Linear layer output shape mismatch"

def test_embedding_forward(setup_embedding):
    x = Tensor([0, 1, 2, 3])
    output = setup_embedding(x)
    assert output.shape == (4, 3), "Embedding output shape mismatch"

def test_gelu_forward(setup_gelu):
    x = Tensor([1.0, 2.0, -1.0])
    output = setup_gelu(x)
    expected_output = 0.5 * x * (1 + ((x + 0.044715 * x ** 3) * (2 / PI)**0.5).tanh())
    assert output == expected_output, "GeLU output mismatch"

def test_rmsnorm_forward(setup_rmsnorm):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = setup_rmsnorm(x)
    expected_output = x / ((x**2).mean(1, keepdim=True) + setup_rmsnorm.epsilon).sqrt() * setup_rmsnorm.scale
    assert output == expected_output, "RMSNorm output mismatch"

def test_sequential_forward(setup_sequential):
    x = Tensor([[0, 1, 2, 3]])
    output = setup_sequential(x)
    assert output.shape == (1, 2), "Sequential output shape mismatch"

if __name__ == "__main__":
    pytest.main()
