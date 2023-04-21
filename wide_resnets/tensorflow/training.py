import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import tensorflow.keras as keras

from progress_table import ProgressTable
import datasets
from wrn.tensorflow.model import ResNet

tf.keras.mixed_precision.set_global_policy('mixed_float16')

ds = datasets.cifar(
    train_batch_size=128,
    valid_batch_size=128,
    repeat_train=False,
)

model = ResNet(
    num_classes=10,
    block_sizes=(2, 2, 2),
    block_channels=(128, 256, 512),
    block_strides=(1, 2, 2),
)

outs = model(tf.random.normal((1, 32, 32, 3)), training=True)
model.summary()

# %%

schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=(
        391 * 60,
        391 * 120,
        391 * 160,
    ),
    values=(
        0.1,
        0.02,
        0.004,
        0.0008,
    ),
)
optimizer = keras.optimizers.SGD(
    learning_rate=schedule,
    momentum=0.9,
    nesterov=True,
)

optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        logits = tf.cast(logits, tf.float32)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, logits)
        loss += sum(model.losses)
        scaled_loss = model.optimizer.get_scaled_loss(loss)
    grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = model.optimizer.get_unscaled_gradients(grads)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, logits


@tf.function
def valid_step(x, y):
    logits = model(x, training=False)
    logits = tf.cast(logits, tf.float32)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, logits)
    return loss, logits


# %%


table = ProgressTable(embedded_progress_bar=True)
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
for epoch in range(200):
    table["epoch"] = epoch
    table["learning rate"] = schedule(training_steps)

    t0 = time.perf_counter()

    for x, y in table(ds["train"]):
        x = tf.cast(x, tf.float16)
        training_steps += 1
        full_loss, logits = train_step(x, y)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, logits)
        accuracy = keras.metrics.SparseCategoricalAccuracy()(y, logits)

        table["full train loss"] = full_loss
        table["train loss"] = loss
        table["train accuracy"] = accuracy

    table["epoch time"] = time.perf_counter() - t0
    for x, y in table(ds["validation"], prefix="VALIDATION: "):
        x = tf.cast(x, tf.float16)
        loss, logits = valid_step(x, y)
        accuracy = keras.metrics.SparseCategoricalAccuracy()(y, logits)

        table["valid loss"] = loss
        table["valid accuracy"] = accuracy
    table["updates"] = int(training_steps)
    table.next_row()
table.close()
