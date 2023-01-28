import pytest 
import torch
from src.models import *

torch.manual_seed(0)

SIZE_IN_SHORT = 5
SIZE_IN_LONG = 32

INPUT_TENSOR_SHORT = torch.rand((1,SIZE_IN_SHORT))
INPUT_TENSOR_LONG = torch.rand((1,SIZE_IN_LONG))

SIZE_OUT = 10


def test_sda():
    sda = ScaledDotAttention(SIZE_IN_SHORT)
    output = sda(INPUT_TENSOR_SHORT, INPUT_TENSOR_SHORT, INPUT_TENSOR_SHORT)
    assert len(output.flatten()) == SIZE_IN_SHORT


def test_ffn():
    ffm = FeedForwardNetwork(SIZE_IN_SHORT,SIZE_OUT)
    output = ffm(INPUT_TENSOR_SHORT)
    assert len(output.flatten()) == SIZE_OUT


def test_mha():
    mha = MultiHeadAttention(SIZE_IN_LONG)
    output = mha(INPUT_TENSOR_LONG)
    assert len(output.flatten()) == SIZE_IN_LONG
