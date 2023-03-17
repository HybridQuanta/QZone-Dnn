import math
import numpy as np
from numpy.testing import assert_almost_equal
import torch
from tqdnn import cc_mul

shape_h_upper_limit = 256
shape_w_upper_limit = 256


def _gen_random(size, dtype="float32"):
    return np.random.random(size=size).astype(dtype)


def _test_cc_mul(device):
    shape = (np.random.randint(1, shape_h_upper_limit + 1), np.random.randint(1, shape_w_upper_limit + 1))
    lhs_real = _gen_random(shape) * math.pi
    lhs_img = _gen_random(shape) * math.pi
    lhs = np.stack((lhs_real, lhs_img), axis=0)
    rhs_real = _gen_random(shape) * math.pi
    rhs_img = _gen_random(shape) * math.pi
    rhs = np.stack((rhs_real, rhs_img), axis=0)

    torch_lhs = torch.from_numpy(lhs).to(device=device)
    torch_rhs = torch.from_numpy(rhs).to(device=device)

    torch_res = cc_mul(torch_lhs, torch_rhs).cpu().detach().numpy()
    numpy_ref = cc_mul(lhs, rhs)

    assert_almost_equal(torch_res, numpy_ref)


def test_cc_mul_cpu():
    device = torch.device("cpu")
    _test_cc_mul(device)


def test_cc_mul_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _test_cc_mul(device)
