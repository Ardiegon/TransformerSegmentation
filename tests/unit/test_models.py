import pytest 
import torch
from src.models import *

INPUT_TENSOR = torch.tensor([[0.8279,0.1293,0.2629,0.5056,0.1980]])
SIZE_IN = 5
SIZE_OUT = 10


def test_sda():
    sda = ScaledDotAttention(SIZE_IN)
    output = sda(INPUT_TENSOR, INPUT_TENSOR, INPUT_TENSOR)
    assert len(INPUT_TENSOR) == len(output)


def test_ffn():
    ffm = FeedForwardNetwork(SIZE_IN,SIZE_OUT)
    output = ffm(INPUT_TENSOR)
    assert len(output.flatten()) == SIZE_OUT