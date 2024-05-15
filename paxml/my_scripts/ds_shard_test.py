import tensorflow as tf


tf.random.set_seed(42)


ds = tf.data.Dataset.range(24)

workers = 4
rank = 0
ds = ds.batch(4)
ds = ds.shard(workers, rank)

ds = ds.as_numpy_iterator()