import click
import time
import os
import torchinfo

import torch
import torch.utils.data

from progress_table import ProgressTable

from wrn.pytorch.model import ResNet

from . import model, dataset


@click.command()
@click.option("--dtype", default="float32")
@click.option("--block-sizes", nargs=3, default=(2, 2, 2))
@click.option("--block-channels", nargs=3, default=(128, 256, 512))
@click.option("--block-strides", nargs=3, default=(1, 2, 2))
@click.option("--batch-size", default=128)
@click.option("--initial-lr", default=0.1)
@click.option("--total-epochs", default=200)
@click.option("--schedule-boundaries", nargs=3, default=(60, 120, 180))
@click.option("--schedule-decay", default=0.2)
@click.option("--weight-decay-alpha", default=5e-4)
@click.option("--nesterov", default=True)
@click.option("--save-csv-path", default=None)
@click.option("--gpu", default=0)
@click.option("--save-pyt-weights", default=False)
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
    weight_decay_alpha,
    nesterov,
    save_csv_path,
    gpu,
    save_pyt_weights,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    train_loader, valid_loader = dataset.load_cifar10()

    model = ResNet(
        num_classes=10,
        block_sizes=block_sizes,
        block_channels=block_channels,
        block_strides=block_strides,
    )

    torchinfo.summary(model)

    if save_pyt_weights:
        pyt_weights = model.state_dict()
        pyt_weights = {
            k: v for k, v in pyt_weights.items() if "num_batches_tracked" not in k and "running" not in k
        }
        torch.save(pyt_weights, save_pyt_weights)

    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=schedule_decay,
        weight_decay=weight_decay_alpha,
        nesterov=nesterov,
    )
    schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=schedule_boundaries,
        gamma=schedule_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    def train_step(x, y):
        optimizer.zero_grad()

        with torch.autocast("cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        accuracy = (logits.argmax(dim=-1) == y).float().mean()
        return loss.item(), accuracy.item()

    def valid_step(x, y):
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y)
            accuracy = (logits.argmax(dim=-1) == y).float().mean()
        return loss.item(), accuracy.item()

    table = ProgressTable(embedded_progress_bar=True)
    table.add_column("epoch", color="blue")
    table.add_column("updates", color="blue")
    # table.add_column("full train loss", aggregate="mean")
    table.add_column("train loss", aggregate="mean")
    table.add_column("train accuracy", aggregate="mean")
    table.add_column("valid loss", aggregate="mean")
    table.add_column("valid accuracy", aggregate="mean")
    table.add_column("learning rate", color="blue")
    table.add_column("epoch time", color="blue")

    training_steps = 0

    for epoch in range(total_epochs):
        table["epoch"] = epoch
        table["learning rate"] = optimizer.state_dict()["param_groups"][0]["lr"]

        t0 = time.perf_counter()

        model.train()
        for x, y in table(train_loader):
            x = x.cuda()
            y = y.cuda()
            training_steps += 1
            loss, accuracy = train_step(x, y)
            table["train loss"] = loss
            table["train accuracy"] = accuracy
        table["epoch time"] = time.perf_counter() - t0

        model.eval()
        for x, y in table(valid_loader):
            x = x.cuda()
            y = y.cuda()
            loss, accuracy = valid_step(x, y)
            table["valid loss"] = loss
            table["valid accuracy"] = accuracy
        table["updates"] = int(training_steps)
        table.next_row()
        schedule.step()
    table.close()

    if save_csv_path:
        table.to_df().to_csv(save_csv_path)


if __name__ == "__main__":
    run()
