import tensorflow as tf
import tensorflow_datasets as tfds
import time

AUTOTUNE = tf.data.AUTOTUNE


def cifar(
    train_batch_size=128,
    valid_batch_size=512,
    padding="reflect",
    dtype=tf.float32,
    shuffle_train=20000,
    repeat_train=True,
    version=10,
    data_dir=None,
):
    subtract = tf.constant([0.49139968, 0.48215841, 0.44653091], dtype=dtype)
    divide = tf.constant([0.24703223, 0.24348513, 0.26158784], dtype=dtype)

    def train_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], mode=padding)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = (x - subtract) / divide
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = (x - subtract) / divide
        return x, y

    if version == 10 or version == 100:
        ds = tfds.load(name=f"cifar{version}", as_supervised=True, data_dir=data_dir)
    else:
        raise Exception(f"version = {version}, but should be either 10 or 100!")
    ds["validation"] = ds["test"]

    if repeat_train:
        ds["train"] = ds["train"].repeat()
    if shuffle_train:
        ds["train"] = ds["train"].shuffle(shuffle_train)
    ds["train"] = ds["train"].map(train_prep, num_parallel_calls=AUTOTUNE)
    ds["train"] = ds["train"].batch(train_batch_size)
    ds["train"] = ds["train"].prefetch(AUTOTUNE)

    ds["validation"] = ds["validation"].map(valid_prep, num_parallel_calls=AUTOTUNE)
    ds["validation"] = ds["validation"].batch(valid_batch_size)
    ds["validation"] = ds["validation"].prefetch(AUTOTUNE)
    return ds


def mnist(
    train_batch_size=100,
    valid_batch_size=400,
    dtype=tf.float32,
    shuffle_train=10000,
    data_dir=None,
):
    def preprocess(x, y):
        x = tf.cast(x, dtype)
        x /= 255
        return x, y

    ds = tfds.load(name="mnist", as_supervised=True, data_dir=data_dir)
    ds["validation"] = ds["test"]

    ds["train"] = ds["train"].repeat()
    ds["train"] = ds["train"].shuffle(shuffle_train)
    ds["train"] = ds["train"].map(preprocess, num_parallel_calls=AUTOTUNE)
    ds["train"] = ds["train"].batch(train_batch_size)
    ds["train"] = ds["train"].prefetch(AUTOTUNE)

    ds["validation"] = ds["validation"].map(preprocess, num_parallel_calls=AUTOTUNE)
    ds["validation"] = ds["validation"].batch(valid_batch_size)
    ds["validation"] = ds["validation"].prefetch(AUTOTUNE)

    ds["input_shape"] = (28, 28, 1)
    ds["n_classes"] = 10
    return ds


def imagenet(
    train_batch_size=128,
    valid_batch_size=128,
    padding="reflect",
    dtype=tf.float32,
    shuffle_train=20000,
    repeat_train=True,
    data_dir=None,
):

    subtract = tf.constant([0.485, 0.456, 0.406], dtype=dtype)
    divide = tf.constant([0.229, 0.224, 0.225], dtype=dtype)

    def train_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = tf.image.random_flip_left_right(x)
        x = tf.image.resize(x, (224, 224))
        x = tf.pad(x, [[16, 16], [16, 16], [0, 0]], mode=padding)
        x = tf.image.random_crop(x, (224, 224, 3))
        x = (x - subtract) / divide
        return x, y

    def valid_prep(x, y):
        x = tf.cast(x, dtype) / 255.0
        x = tf.image.resize(x, (224, 224))
        x = (x - subtract) / divide
        return x, y

    input_context = tf.distribute.InputContext(
        input_pipeline_id=0,  # Worker id
        num_input_pipelines=8,  # Total number of workers
    )
    read_config = tfds.ReadConfig(
        input_context=input_context,
    )
    ds = tfds.load(
        name=f"imagenet2012",
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=True,
        read_config=read_config,
    )

    if repeat_train:
        ds["train"] = ds["train"].repeat()
    if shuffle_train:
        ds["train"] = ds["train"].shuffle(shuffle_train)
    ds["train"] = ds["train"].map(train_prep, num_parallel_calls=AUTOTUNE)
    ds["train"] = ds["train"].batch(train_batch_size)
    ds["train"] = ds["train"].prefetch(AUTOTUNE)

    ds["validation"] = ds["validation"].map(valid_prep, num_parallel_calls=AUTOTUNE)
    ds["validation"] = ds["validation"].batch(valid_batch_size)
    ds["validation"] = ds["validation"].prefetch(AUTOTUNE)
    return ds


def placeholder(train_batch_size=100, image_shape=(32, 32, 3), dtype=tf.float32):
    images = tf.ones([2, *image_shape])
    target = tf.constant([0, 1])

    def preprocess(x, y):
        x = tf.cast(x, dtype)
        return x, y

    ds = {}
    ds["train"] = tf.data.Dataset.from_tensor_slices((images, target))
    ds["train"] = ds["train"].map(preprocess).repeat().batch(train_batch_size)
    ds["validation"] = tf.data.Dataset.from_tensor_slices((images, target))
    ds["validation"] = ds["validation"].map(preprocess).batch(2)
    return ds


def figure_input_shape(ds):
    for x, y in ds["validation"]:
        break
    else:
        raise RuntimeError("Dataset is empty!")
    return x.shape[1:]


def figure_n_classes(ds):
    classes = set()
    for x, y in ds["validation"]:
        classes.update(y.numpy())
    return len(classes)


def benchmark(ds, time_limit=2):
    batch_size = None
    t0 = time.time()
    time_delta = None
    it = -1

    for x, y in ds["train"]:
        batch_size = x.shape[0]
        it += 1

        time_delta = time.time() - t0
        if time_delta > time_limit:
            break

        if it == 0:  # warmup iteration
            t0 = time.time()

    print("DATASET BENCHMARKING")
    print(f"Iterations per second: {it / time_delta:6.3f}")
    print(f"Examples per second: {it * batch_size / time_delta:6.3f}")
