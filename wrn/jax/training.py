import os
import time

import click
import flax
import jax
import jax.numpy as jnp
import optax
from clu import parameter_overview
from flax.core import FrozenDict
from jax.random import PRNGKey
from progress_table import ProgressTable

from .dataset import load_cifar10
from .model import ResNet
from .weight_conversion import load_and_convert_all


@click.command()
@click.option("--dtype", default="float16")
@click.option("--block-sizes", nargs=3, default=(2, 2, 2))
@click.option("--block-channels", nargs=3, default=(128, 256, 512))
@click.option("--block-strides", nargs=3, default=(1, 2, 2))
@click.option("--batch-size", default=128)
@click.option("--initial-lr", default=0.1)
@click.option("--total-epochs", default=200)
@click.option("--schedule-boundaries", nargs=3, default=(60, 120, 180))
@click.option("--schedule-decay", default=0.2)
@click.option("--momentum", default=0.9)
@click.option("--nesterov", default=True)
@click.option("--weight-decay-alpha", default=5e-4)
@click.option("--l2-loss-alpha", default=0)
@click.option("--amp-scaling", default=128)
@click.option("--save-csv-path", default=None)
@click.option("--gpu", default=0)
@click.option("--load-pyt-weights", default=False)
def run(
    dtype,
    block_sizes,
    block_channels,
    block_strides,
    batch_size,
    initial_lr,
    total_epochs,
    schedule_boundaries,
    schedule_decay,
    momentum,
    nesterov,
    weight_decay_alpha,
    l2_loss_alpha,
    amp_scaling,
    save_csv_path,
    gpu,
    load_pyt_weights,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if dtype == "float32":
        dtype = jnp.float32
    elif dtype == "float16":
        dtype = jnp.float16
    else:
        raise KeyError

    device = jax.devices()[0]
    rng = PRNGKey(0)

    model = ResNet(
        num_classes=10,
        dtype=dtype,
        block_sizes=block_sizes,
        block_channels=block_channels,
        block_strides=block_strides,
    )

    jax_weights = model.init(rng, jnp.ones((1, 32, 32, 3)), train=True)

    if load_pyt_weights:
        jax_weights = load_and_convert_all(load_pyt_weights, jax_weights)

    print(parameter_overview.get_parameter_overview(jax_weights["batch_stats"]))
    print(parameter_overview.get_parameter_overview(jax_weights["params"]))
    params = jax_weights

    train_loader, valid_loader = load_cifar10(batch_size)
    epoch_steps = len(train_loader)

    schedule = optax.piecewise_constant_schedule(
        initial_lr,
        boundaries_and_scales={epoch_steps * boundary: schedule_decay for boundary in schedule_boundaries},
    )

    optimizer = optax.sgd(learning_rate=schedule, momentum=momentum, nesterov=nesterov)
    optimizer_state = optimizer.init(params["params"])

    def get_l2(params, alpha):
        l2 = 0.0
        for k, w in flax.traverse_util.flatten_dict(params).items():
            l2 += jnp.sum(w ** 2) * alpha
        return l2

    def weight_decay(updates, params, beta):
        flat_params = flax.traverse_util.flatten_dict(params)
        flat_updates = flax.traverse_util.flatten_dict(updates)
        for k, v in flat_params.items():
            # if "kernel" in k:
            flat_updates[k] += v * beta
        return FrozenDict(flax.traverse_util.unflatten_dict(flat_updates))

    def get_loss(params, x, y, train):
        logits, params = model.apply(params, x, train=train, mutable=True)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        main_loss = jnp.mean(losses)
        if l2_loss_alpha:
            reg_loss = get_l2(params["params"], alpha=l2_loss_alpha)
        else:
            reg_loss = 0
        return amp_scaling * (main_loss + reg_loss), (main_loss, logits, params)

    get_value_and_grad = jax.value_and_grad(get_loss, has_aux=True)

    def train_step(params, optimizer_state, x, y):
        (full_loss, (main_loss, logits, params)), grads = get_value_and_grad(params, x, y, train=True)

        full_loss = full_loss / amp_scaling
        grads = jax.tree_util.tree_map(lambda x: x / amp_scaling, grads)
        trainable = params["params"]
        grads = grads["params"]
        if weight_decay_alpha:
            grads = weight_decay(grads, trainable, weight_decay_alpha)
        grads = FrozenDict(grads)

        updates, optimizer_state = optimizer.update(grads, optimizer_state, trainable)
        trainable = optax.apply_updates(trainable, updates)
        params = FrozenDict({**params, "params": trainable})

        accuracy = jnp.mean(logits.argmax(-1) == y)
        return params, optimizer_state, full_loss, main_loss, accuracy

    train_step = jax.jit(train_step, device=device)

    def valid_step(params, x, y):
        loss, (main_loss, logits, params) = get_loss(params, x, y, train=False)
        accuracy = jnp.mean(logits.argmax(-1) == y)
        return main_loss, accuracy

    valid_step = jax.jit(valid_step, device=device)

    table = ProgressTable(embedded_progress_bar=True, table_style="round")

    table.add_column("epoch", color="blue")
    table.add_column("updates", color="blue")
    table.add_column("full train loss", aggregate="mean")
    table.add_column("train loss", aggregate="mean")
    table.add_column("train accuracy", aggregate="mean")
    table.add_column("valid loss", aggregate="mean")
    table.add_column("valid accuracy", aggregate="mean")
    table.add_column("learning rate", color="blue")
    table.add_column("epoch time", color="blue")

    training_steps = 0
    for epoch in range(total_epochs):
        table["epoch"] = epoch
        table["learning rate"] = schedule(training_steps)

        t0 = time.perf_counter()

        for x, y in table(train_loader):
            x = jnp.array(x)
            y = jnp.array(y)
            training_steps += 1
            params, optimizer_state, full_loss, loss, accuracy = train_step(params, optimizer_state, x, y)

            table["full train loss"] = full_loss
            table["train loss"] = loss
            table["train accuracy"] = accuracy
        table["epoch time"] = time.perf_counter() - t0

        for x, y in table(valid_loader):
            x = jnp.array(x)
            y = jnp.array(y)
            loss, accuracy = valid_step(params, x, y)

            table["valid loss"] = loss
            table["valid accuracy"] = accuracy
        table["updates"] = int(training_steps)
        table.next_row()
    table.close()

    if save_csv_path:
        table.to_df().to_csv(save_csv_path)


if __name__ == "__main__":
    run()
