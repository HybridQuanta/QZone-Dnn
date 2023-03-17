import math
import numpy as np
from numpy.testing import assert_almost_equal
import torch
from tqdnn import cc_mul, rc_mul, s_mul

shape_h_upper_limit = 256
shape_w_upper_limit = 256


################ neurophox reference implementation ############################
def ref_rc_mul(real: torch.Tensor, comp: torch.Tensor):
    return real.unsqueeze(dim=0) * comp


def ref_cc_mul(comp1: torch.Tensor, comp2: torch.Tensor) -> torch.Tensor:
    real = comp1[0] * comp2[0] - comp1[1] * comp2[1]
    comp = comp1[0] * comp2[1] + comp1[1] * comp2[0]
    return torch.stack((real, comp), dim=0)


def ref_s_mul(s: np.complex, comp: torch.Tensor):
    return s.real * comp + torch.stack((-s.imag * comp[1], s.imag * comp[0]))


################################################################################


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

    torch_out = cc_mul(torch_lhs, torch_rhs).cpu().detach().numpy()
    numpy_out = cc_mul(lhs, rhs)
    ref_out = ref_cc_mul(torch_lhs, torch_rhs).cpu().numpy()

    assert_almost_equal(torch_out, ref_out)
    assert_almost_equal(numpy_out, ref_out)


def test_cc_mul_cpu():
    device = torch.device("cpu")
    _test_cc_mul(device)


def test_cc_mul_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _test_cc_mul(device)


def _test_rc_mul(device):
    shape = (np.random.randint(1, shape_h_upper_limit + 1), np.random.randint(1, shape_w_upper_limit + 1))
    lhs = _gen_random(shape) * math.pi
    rhs_real = _gen_random(shape) * math.pi
    rhs_img = _gen_random(shape) * math.pi
    rhs = np.stack((rhs_real, rhs_img), axis=0)

    torch_lhs = torch.from_numpy(lhs).to(device=device)
    torch_rhs = torch.from_numpy(rhs).to(device=device)

    torch_out = rc_mul(torch_lhs, torch_rhs).cpu().detach().numpy()
    numpy_out = rc_mul(lhs, rhs)
    ref_out = ref_rc_mul(torch_lhs, torch_rhs).cpu().numpy()

    assert_almost_equal(torch_out, ref_out)
    assert_almost_equal(numpy_out, ref_out)


def test_rc_mul_cpu():
    device = torch.device("cpu")
    _test_rc_mul(device)


def test_rc_mul_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _test_rc_mul(device)


def _test_s_mul(device):
    shape = (np.random.randint(1, shape_h_upper_limit + 1), np.random.randint(1, shape_w_upper_limit + 1))
    lhs = np.complex(np.random.rand(), np.random.rand()) * math.pi
    rhs_real = _gen_random(shape) * math.pi
    rhs_img = _gen_random(shape) * math.pi
    rhs = np.stack((rhs_real, rhs_img), axis=0)

    torch_rhs = torch.from_numpy(rhs).to(device=device)

    torch_out = s_mul(lhs, torch_rhs).cpu().detach().numpy()
    numpy_out = s_mul(lhs, rhs)
    ref_out = ref_s_mul(lhs, torch_rhs).cpu().numpy()

    assert_almost_equal(torch_out, ref_out)
    assert_almost_equal(numpy_out, ref_out)


def test_s_mul_cpu():
    device = torch.device("cpu")
    _test_s_mul(device)


def test_s_mul_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _test_s_mul(device)
