import functools
import numpy as np
import torch


def torch_tensor_to_numpy_ndarray(torch_tensors):
    if len(torch_tensors) == 0:
        return []
    device = torch_tensors[0].device
    np_arrays = []
    for tt in torch_tensors:
        assert tt.device == device
        np_arrays.append(tt.cpu().detach().numpy())
    return np_arrays


def params_compatible_with_torch_tensor(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        indexed_torch_tensors = [(i, x) for i, x in enumerate(args) if isinstance(x, torch.Tensor)]
        if len(indexed_torch_tensors) == 0:
            # numpy ndarray args
            return f(*args, **kwargs)

        # torch tensor args
        indices, torch_tensors = list(zip(*indexed_torch_tensors))
        device = torch_tensors[0].device
        numpy_arrays = torch_tensor_to_numpy_ndarray(torch_tensors)
        new_args = list(args)
        for idx, np_array in zip(indices, numpy_arrays):
            new_args[idx] = np_array
        np_res = f(*new_args, **kwargs)
        return torch.from_numpy(np_res).to(device=device)

    return wrapper


@params_compatible_with_torch_tensor
def cc_mul(comp1: np.ndarray, comp2: np.ndarray) -> np.ndarray:
    """
    complex * complex op
    If both of the inputs' types are torch.Tensor, the output's type will also be torch.Tensor.
    """
    real = comp1[0] * comp2[0] - comp1[1] * comp2[1]
    comp = comp1[0] * comp2[1] + comp1[1] * comp2[0]
    return np.stack((real, comp), axis=0)


@params_compatible_with_torch_tensor
def rc_mul(real: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """
    real * complex op
    If inputs' type is torch.Tensor, the output's type will also be torch.Tensor.
    """
    return np.expand_dims(real, axis=0) * comp


@params_compatible_with_torch_tensor
def s_mul(s: np.complex, comp: np.ndarray) -> np.ndarray:
    """
    complex scalar * complex op
    If inputs' type is torch.Tensor, the output's type will also be torch.Tensor.
    """
    return s.real * comp + np.stack((-s.imag * comp[1], s.imag * comp[0]))
