import tensorflow as tf
import numpy as np

# # Load the training data into two NumPy arrays, for example using `np.load()`.
# features = np.load(X)
# labels = np.load(Y)
# # Assume that each row of `features` corresponds to the same row as `labels`.
# assert features.shape[0] == labels.shape[0]

# features_placeholder = tf.placeholder(features.dtype, features.shape)
# labels_placeholder = tf.placeholder(labels.dtype, labels.shape)


# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))



def read_npy_file(item):
    data = np.load(item.numpy().decode())
    return data
file_list = ['/npy_data/case_00001_x.npy','/npy_data/case_00001_y.npy']

dataset = tf.data.Dataset.from_tensor_slices(file_list)
dataset = dataset.map(lambda item: tf.py_func(read_npy_file, [item], [tf.float32,]))
print(dataset.output_shapes)