import torch
import torch.nn as nn
import numpy as np

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

if __name__ == "__main__":
    sda = ScaledDotAttention(5)
    input = torch.tensor([[0.8279],[0.1293],[0.2629],[0.5056],[0.1980]])
    print(input)

    output = sda(input, input, input)
    print(output)