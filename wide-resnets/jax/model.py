import flax.linen as nn
import jax.numpy as jnp


class ResidualBlock(nn.Module):
    size: int
    channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train):
        stride = self.stride

        for i in range(self.size):
            inputs = x

            x = nn.BatchNorm(
                dtype=self.dtype,
            )(x, use_running_average=not train)
            x = nn.relu(x)

            activ = x

            x = nn.Conv(self.channels, (3, 3), stride, dtype=self.dtype, use_bias=False)(x)
            x = nn.BatchNorm(
                dtype=self.dtype,
            )(x, use_running_average=not train)
            x = nn.relu(x)
            x = nn.Conv(self.channels, (3, 3), dtype=self.dtype, use_bias=False)(x)
            if self.channels != activ.shape[-1] or stride != 1:
                shortcut = nn.Conv(
                    self.channels,
                    (1, 1),
                    strides=(stride, stride),
                    dtype=self.dtype,
                    use_bias=False,
                )(activ)
            else:
                shortcut = inputs
            x = x + shortcut
            stride = 1  # do not use stride for next blocks

        return x


class ResNet(nn.Module):
    num_classes: int
    dtype: jnp.dtype = jnp.float32
    block_sizes: tuple = (2, 2, 2, 2)
    block_channels: tuple = (64, 128, 256, 512)
    block_strides: tuple = (1, 2, 2, 2)

    @nn.compact
    def __call__(self, x, train):
        first_channels = 16
        x = nn.Conv(
            features=first_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )(x)

        for i, (size, channels, stride) in enumerate(
            zip(self.block_sizes, self.block_channels, self.block_strides)
        ):
            x = ResidualBlock(
                size=size,
                channels=channels,
                stride=stride,
                dtype=self.dtype,
            )(x, train=train)

        x = nn.BatchNorm(
            dtype=self.dtype,
        )(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(x.shape[1], x.shape[2])).squeeze()
        x = nn.Dense(features=self.num_classes, dtype=self.dtype)(x)
        return x
