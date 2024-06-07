from torch import zeros, ones
from torch import sqrt
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Module

class LayerNormalization(Module):
    def __init__(self, model_dimension: int, epsilon: float = 1e-12):
        super().__init__()
        self.gamma = Parameter(data=ones(model_dimension))
        self.beta = Parameter(data=zeros(model_dimension))
        self.epsilon = epsilon

    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        variance = input.var(dim=-1, unbiased=True, keepdim=True)
        return self.gamma * ((input - mean) / sqrt(variance + self.epsilon)) + self.beta