from model.attention import MultiHeadAttention
from model.autoencoder import Autoencoder
from model.normalization import LayerNormalization

from torch import Tensor
from torch import Module

class Encoder(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, number_of_heads: int, dropout_rate: float):
        super().__init__()
        self.attention = MultiHeadAttention(model_dimension, number_of_heads, dropout_rate)
        self.ffn = Autoencoder(model_dimension, hidden_dimension, dropout_rate)
        self.first_normalization = LayerNormalization(model_dimension)
        self.second_normalization = LayerNormalization(model_dimension)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x, x, x) + x
        x = self.first_normalization(x)
        x = self.ffn(x) + x
        x = self.second_normalization(x)
        return x