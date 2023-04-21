import click
import torch
from .model import ResNet
import torchinfo
import torchview


@click.command()
@click.option("--path")
@click.option("--depth", default=float("inf"))
def run(path, depth):
    model = ResNet(
        num_classes=10,
        block_sizes=(2, 2, 2),
        block_channels=(128, 256, 512),
        block_strides=(1, 2, 2)
    )

    print(torchinfo.summary(model, depth=4))

    x = torch.randn(1, 3, 32, 32)
    gr = torchview.draw_graph(model, x, expand_nested=True, depth=depth)
    gr.visual_graph.render(filename=path, format="pdf")


if __name__ == "__main__":
    run()
