from torch import Tensor
from torch.nn import Module
from torch.nn import Linear, ReLU, Sequential, Dropout

class Autoencoder(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, dropout_rate: float = 0.2):
        super().__init__()
        self.layers = Sequential(
            Linear(model_dimension, hidden_dimension),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dimension, model_dimension)
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)