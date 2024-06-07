from math import sqrt
from torch import Tensor
from torch.nn import Module
from torch.nn import Linear, Dropout
from torch.nn.functional import softmax

class MultiHeadAttention(Module):
    def __init__(self, model_dimension: int, number_of_heads: int, dropout_rate: float):
        super().__init__()
        self.number_of_heads = number_of_heads

        self.key_projector = Linear(model_dimension, model_dimension, bias=False)
        self.value_projector = Linear(model_dimension, model_dimension, bias=False)
        self.query_projector = Linear(model_dimension, model_dimension, bias=False)
        self.output_projector = Linear(model_dimension, model_dimension, bias=False)

        self.dropout = Dropout(p=dropout_rate)
        self.mask = None
        
    def split(self, tensor: Tensor) -> Tensor:
        batch_size, sequence_lenght, model_dimension = tensor.size()
        tensor_dimension = model_dimension // self.number_of_heads
        return tensor.view(batch_size, sequence_lenght, self.number_of_heads, tensor_dimension).transpose(1, 2)
    
    def concatenate(self, tensor: Tensor) -> Tensor: 
        batch_size, number_of_heads, sequence_lenght, tensor_dimension = tensor.size()
        model_dimension = number_of_heads * tensor_dimension
        return tensor.transpose(1, 2).contiguous().view(batch_size, sequence_lenght, model_dimension)

    def attention(self, key: Tensor, query: Tensor, value: Tensor) -> Tensor:
        scale = sqrt(key.size(-1))
        score = (query @ key.transpose(-2, -1) ) / scale
        if self.mask:
            score = score.masked_fill(self.mask, float('-inf'))
        return softmax(score, dim=-1) @ value
    
    def forward(self, key: Tensor, query: Tensor, value: Tensor) -> Tensor:
        key, query, value = self.key_projector(key), self.query_projector(query), self.value_projector(value)
        key, query, value = self.split(key), self.split(query), self.split(value)
        attention = self.attention(key, query, value)
        attention = self.dropout(attention)
        attention = self.concatenate(attention)
        return self.output_projector(attention)