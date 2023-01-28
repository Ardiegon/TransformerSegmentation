import torch
import torch.nn as nn
import numpy as np
import math

class ScaledDotAttention(nn.Module):
    def __init__(self, keys_dim: int) -> None:
        super().__init__()
        self.keys_dim_sqrt = np.sqrt(keys_dim)
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor):
        x = torch.mm(keys, torch.transpose(queries,0,1))
        x = self.softmax(x/self.keys_dim_sqrt)
        x = torch.mm(x,values)
        return x
        

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_heads = 8) -> None:
        super().__init__()
        
        assert model_dim%n_heads==0
        
        self.model_dim = model_dim
        self.split_dim = model_dim / n_heads
        self.n_heads = n_heads

        self.attention_layer = ScaledDotAttention(self.split_dim)
        self.m_proj = nn.Linear(self.model_dim, self.model_dim)
        self.q_proj = nn.Linear(self.split_dim, self.model_dim)
        self.k_proj = nn.Linear(self.split_dim, self.model_dim)
        self.v_proj = nn.Linear(self.split_dim, self.model_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.m_proj.weight)
        self.m_proj.bias.data.fill_(0)


    def forward(self, x):
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, size_in = 512, size_out=512) -> None:
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out

        weights_0 = torch.Tensor(4*size_in, size_in)
        weights_1 = torch.Tensor(size_out, 4*size_in)
        biases_0 = torch.Tensor(4*size_in)
        biases_1 = torch.Tensor(size_out)  

        self.weights_0 = nn.Parameter(weights_0)
        self.weights_1 = nn.Parameter(weights_1)
        self.biases_0 = nn.Parameter(biases_0)
        self.biases_1 = nn.Parameter(biases_1)

        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.weights_0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_1, a=math.sqrt(5))
        fan_in_0, _ = nn.init._calculate_fan_in_and_fan_out(self.weights_0)
        fan_in_1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights_1)
        bound_0 = 1 / math.sqrt(fan_in_0)
        bound_1 = 1 / math.sqrt(fan_in_1)
        nn.init.uniform_(self.biases_0, -bound_0, bound_0)
        nn.init.uniform_(self.biases_0, -bound_1, bound_1)

    def forward(self, x):
        x = torch.add(torch.mm(x, self.weights_0.t()), self.biases_0.t())
        x = self.relu(x)
        x = torch.add(torch.mm(x, self.weights_1.t()), self.biases_1.t())
        return x

