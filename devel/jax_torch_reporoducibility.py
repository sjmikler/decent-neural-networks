import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict
import flax.linen as nn
import flax

import torch

# from jax_.model_like_lua import ResNet as JaxResNet
# from pytorch_.model_like_lua import ResNet as PytResNet

from jax_.model_simple import Simple as JaxResNet
from pytorch_.model_simple import Simple as PytResNet

# %%

jax_x = jax.random.normal(jax.random.PRNGKey(0), (128, 32, 32, 3))
pyt_x = torch.tensor(jax_x.tolist()).permute((0, 3, 1, 2))

jax_y = jnp.array([0] * 128)
pyt_y = torch.tensor(jax_y.tolist())

# %%

pyt_model = PytResNet(
    # num_classes=10,
    # block_sizes=(2, 2, 2),
    # block_channels=(2, 2, 2),
    # block_strides=(1, 2, 2),
)
pyt_model.eval()
pyt_model(torch.randn(1, 3, 32, 32))
pyt_weights = pyt_model.state_dict()
pyt_weights = {k: v for k, v in pyt_weights.items() if "num_batches_tracked" not in k and "running" not in k}


def convert(pyt_w):
    arr = jnp.array(pyt_w)
    if len(pyt_w.shape) == 4:
        return jnp.transpose(arr, (2, 3, 1, 0))
    elif len(pyt_w.shape) == 2:
        return jnp.transpose(arr, (1, 0))
    return arr


jax_model = JaxResNet(
    # num_classes=10,
    # block_sizes=(2, 2, 2),
    # block_channels=(2, 2, 2),
    # block_strides=(1, 2, 2),
)
params = jax_model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
jax_weights = params["params"]
jax_weights = flatten_dict(jax_weights)

for k, v in jax_weights.items():
    print(k, v.shape)

for k, v in pyt_weights.items():
    print(k, v.shape)

print()

new_jax_weights = {}
for (k1, v1), (k2, v2) in zip(pyt_weights.items(), jax_weights.items()):
    # print(k1, v1.shape, k2, v2.shape)
    new_jax_weights[k2] = convert(v1)
    assert new_jax_weights[k2].shape == v2.shape

new_jax_weights = unflatten_dict(new_jax_weights)

# %%

pyt_model.train()

optimizer = torch.optim.SGD(
    pyt_model.parameters(),
    lr=0.1,
    momentum=0.9,
    nesterov=True,
    weight_decay=2e-3,
)
loss_fn = torch.nn.CrossEntropyLoss()


def pyt_train_step(x, y):
    optimizer.zero_grad()

    # with torch.autocast("cuda"):
    logits = pyt_model(x)
    loss = loss_fn(logits, y)

    loss.backward()
    optimizer.step()

    accuracy = (logits.argmax(dim=-1) == y).float().mean()
    return loss.item(), accuracy.item()


pyt_model.eval()
pyt_outs = pyt_model(pyt_x)
print(float(torch.sum(pyt_outs)))

pyt_model.train()
for _ in range(5):
    loss, _ = pyt_train_step(pyt_x, pyt_y)
    print(float(loss))

pyt_model.eval()
pyt_outs = pyt_model(pyt_x)
print(float(torch.sum(pyt_outs)))

pyt_model.train()
pyt_outs = pyt_model(pyt_x)
print(float(torch.sum(pyt_outs)))
print()

# %%


optimizer = optax.sgd(
    learning_rate=0.1,
    momentum=0.9,
    nesterov=True,
)
use_l2 = True
params = FrozenDict({"params": new_jax_weights, "batch_stats": params["batch_stats"]})

optimizer_state = optimizer.init(params)


def weight_decay(updates, params, beta):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_updates = flax.traverse_util.flatten_dict(updates)
    for k, v in flat_params.items():
        # if "kernel" in k:
        flat_updates[k] += v * beta
    return FrozenDict(flax.traverse_util.unflatten_dict(flat_updates))


def get_l2(params, alpha):
    l2 = 0.0
    for k, w in flax.traverse_util.flatten_dict(params).items():
        l2 += jnp.sum(w ** 2) * alpha
    return l2


ALPHA = 1e-3
BETA = 2e-3
USE_L2 = False
USE_WD = True


def get_loss(params, x, y, train):
    logits, params = jax_model.apply(params, x, train=train, mutable=True)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    main_loss = jnp.mean(losses)
    if USE_L2:
        reg_loss = get_l2(params["params"], alpha=ALPHA)
    else:
        reg_loss = 0
    return main_loss + reg_loss, (main_loss, logits, params)


get_value_and_grad = jax.value_and_grad(get_loss, has_aux=True)


def train_step(params, optimizer_state, x, y):
    (full_loss, (main_loss, logits, params)), grads = get_value_and_grad(params, x, y, train=True)

    if USE_WD:
        trainable = params["params"]
        g = grads["params"]
        g = weight_decay(g, trainable, BETA)
        grads = FrozenDict({**grads, "params": g})

    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)

    accuracy = jnp.mean(logits.argmax(-1) == y)
    return params, optimizer_state, full_loss, main_loss, accuracy


jax_outs, _ = jax_model.apply(params, jax_x, mutable=True, train=False)
print(float(jnp.sum(jax_outs)))

for _ in range(5):
    params, optimizer_state, reg_loss, loss, _ = train_step(params, optimizer_state, jax_x, jax_y)
    print(float(loss))

jax_outs, _ = jax_model.apply(params, jax_x, mutable=True, train=False)
print(float(jnp.sum(jax_outs)))

jax_outs, _ = jax_model.apply(params, jax_x, mutable=True, train=True)
print(float(jnp.sum(jax_outs)))
