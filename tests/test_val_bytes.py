import torch

from toolkit.utils import val_bytes


def test_tensor_bytes():
    x = torch.randn(2, 3, dtype=torch.float32)
    assert val_bytes(x) == x.numel() * x.element_size()


def test_symint_handling():
    x = torch.empty(2, 3)
    result = val_bytes(x)
    assert isinstance(result, int)
    assert result == x.numel() * x.element_size()


def test_tuple_input():
    a = torch.randn(2, 3, dtype=torch.float16)
    b = torch.randn(4, dtype=torch.float32)
    expected = a.numel() * a.element_size() + b.numel() * b.element_size()
    assert val_bytes((a, [b])) == expected


def test_empty_tensor():
    x = torch.empty(0, dtype=torch.float32)
    assert val_bytes(x) == 0
