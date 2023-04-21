import argparse
import os
import time

import flax
import gin
import jax
import jax.numpy as jnp
import optax
from rich import print
import rich.traceback
from clu import parameter_overview
from flax.core import FrozenDict
from jax.random import PRNGKey
from progress_table import ProgressTable

from .dataset import load_cifar10
from .model import ResNet
from .weight_conversion import load_and_convert_all

rich.traceback.install()

# REGISTER CONSTANTS
gin.constant("jnp.float16", jnp.float16)
gin.constant("jnp.float32", jnp.float32)
gin.external_configurable(optax.sgd)


@gin.configurable
def training(
    initial_lr,
    total_epochs,
    schedule_boundaries,
    schedule_decay,
    weight_decay_alpha,
    l2_loss_alpha,
    amp_scaling,
    save_csv_path,
    load_pyt_weights,
):
    print(dict(
        initial_lr=initial_lr,
        total_epochs=total_epochs,
        schedule_boundaries=schedule_boundaries,
        schedule_decay=schedule_decay,
        weight_decay_alpha=weight_decay_alpha,
        l2_loss_alpha=l2_loss_alpha,
        amp_scaling=amp_scaling,
        save_csv_path=save_csv_path,
        load_pyt_weights=load_pyt_weights,
    ))

    rng = PRNGKey(0)
    device = jax.devices()[0]
    model = ResNet(num_classes=10)

    jax_weights = model.init(rng, jnp.ones((1, 32, 32, 3)), train=True)

    if load_pyt_weights:
        jax_weights = load_and_convert_all(load_pyt_weights, jax_weights)

    print(parameter_overview.get_parameter_overview(jax_weights["batch_stats"]))
    print(parameter_overview.get_parameter_overview(jax_weights["params"]))
    params = jax_weights

    train_loader, valid_loader = load_cifar10()
    epoch_steps = len(train_loader)

    schedule = optax.piecewise_constant_schedule(
        initial_lr,
        boundaries_and_scales={epoch_steps * boundary: schedule_decay for boundary in schedule_boundaries},
    )

    optimizer = optax.sgd(learning_rate=schedule)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", required=False, default=0)

    args = parser.parse_args()

    path = args.config
    if not path.endswith(".gin"):
        path = path + ".gin"
    if not path.startswith("config"):
        path = "config-" + path

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    gin.parse_config_file(f"wide_resnets/jax/{path}")
    training()
