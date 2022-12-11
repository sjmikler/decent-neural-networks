import jax.numpy as jnp
import torch
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict


def convert(pyt_w):
    arr = jnp.array(pyt_w)
    if len(pyt_w.shape) == 4:
        return jnp.transpose(arr, (2, 3, 1, 0))
    elif len(pyt_w.shape) == 2:
        return jnp.transpose(arr, (1, 0))
    return arr


def load_and_convert_all(pyt_weights_path, jax_weights):
    pyt_weights = torch.load(pyt_weights_path)

    new_jax_params = {}
    flat_jax_weights = flatten_dict(jax_weights["params"])
    for (k1, v1), (k2, v2) in zip(pyt_weights.items(), flat_jax_weights.items()):
        new_jax_params[k2] = convert(v1)
        assert new_jax_params[k2].shape == v2.shape

    new_jax_params = unflatten_dict(new_jax_params)
    params = FrozenDict({**jax_weights, "params": new_jax_params})
