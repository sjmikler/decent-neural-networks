import tensorflow as tf

ALPHA = 2.5e-4
initializer = tf.keras.initializers.LecunNormal()


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, start_with_bn, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.start_with_bn = start_with_bn

    def build(self, input_shape):
        self.relu = tf.keras.layers.ReLU()

        if self.start_with_bn:
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.start = tf.keras.Sequential([
                self.bn1,
                self.relu,
            ])
        else:
            self.start = tf.keras.Sequential()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=self.strides,
            padding="same",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(ALPHA),
            kernel_initializer=initializer,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(ALPHA),
            kernel_initializer=initializer,
        )
        if input_shape[-1] != self.filters or self.strides != 1:
            self.shortcut = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                strides=self.strides,
                padding="same",
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(ALPHA),
                kernel_initializer=initializer,
            )
        else:
            self.shortcut = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs, training):
        x = inputs
        x = self.start(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        shortcut = self.shortcut(inputs)
        return x + shortcut


class ResidualGroup(tf.keras.Model):
    def __init__(self, filters, strides, num_blocks, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.num_blocks = num_blocks

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.num_blocks):
            if i == 0:
                block = ResidualBlock(self.filters, self.strides, start_with_bn=False)
            else:
                block = ResidualBlock(self.filters, 1, start_with_bn=True)
            self.blocks.append(block)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet(tf.keras.Model):
    def __init__(self, num_classes, block_sizes, block_channels, block_strides, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.block_sizes = block_sizes
        self.block_channels = block_channels
        self.block_strides = block_strides

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.block_channels[0],
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(ALPHA),
            kernel_initializer=initializer,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.groups = []
        for size, channels, strides in zip(self.block_sizes, self.block_channels, self.block_strides):
            self.groups.append(ResidualGroup(channels, strides, size))

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(self.num_classes,
                                        kernel_regularizer=tf.keras.regularizers.l2(ALPHA),
                                        kernel_initializer=initializer)

    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        for block in self.groups:
            x = block(x, training=training)
        x = self.avg_pool(x)
        return self.fc(x)
