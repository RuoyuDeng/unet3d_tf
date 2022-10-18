import tensorflow as tf

def parse(serialized):
    """ Parse TFRecord

    :param serialized: Serialized record for a particular example
    :return: sample, label, mean and std of intensities
    """
    
    features = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
        'mean': tf.io.FixedLenFeature([], tf.float32),
        'stdev': tf.io.FixedLenFeature([], tf.float32),
        'shape_x': tf.io.FixedLenFeature([], tf.int64),
        'shape_y': tf.io.FixedLenFeature([], tf.int64),
        'shape_z': tf.io.FixedLenFeature([], tf.int64)
    }


    parsed_example = tf.io.parse_single_example(serialized=serialized,features=features)

    def get_val(x):
        zero = tf.constant(0,dtype=tf.int32)
        sess = tf.Session()
        result = sess.run(tf.add(x,zero))
        sess.close()
        return result

    
    print("The class of parsed_example['shape_x'] is :", type(parsed_example['shape_x']))    
    x = tf.cast(parsed_example['shape_x'],tf.int32)
    y = tf.cast(parsed_example['shape_y'],tf.int32)
    z = tf.cast(parsed_example['shape_z'],tf.int32)
    print("Type of shape_x:", type(x))
    print("shape_x is: ", x)
    print("shape_x value is: ", tf.get_static_value(x))

    
    # with sess.as_default():
    #     print("The fucking x value is:",x.eval())
    # sess.close()
    # print("The fucking x value is:", sess.run(op))
    # print(tf.print(x))

    tf.config.experimental_run_functions_eagerly(True)
    g = tf.function(get_val)
    func_result = g(x)
    print("Result of executing function eagerly: ",func_result)
    tf.config.experimental_run_functions_eagerly(False)

    # x = tf.constant(1)
    # print("Type of shape_x:", type(x))
    # print("shape_x is: ", x)
    # print("shape_x value is: ", tf.get_static_value(x, partial=True))
    
        
    x_shape = (x, y, z, 1)
    y_shape = (x, y, z)
    
    print("The whole x_shape is:",x_shape)
    sample = tf.io.decode_raw(parsed_example['X'], tf.float32)
    # sample = tf.cast(tf.reshape(sample, x_shape), tf.float32)
    
    label = tf.io.decode_raw(parsed_example['Y'], tf.uint8)
    # label = tf.cast(tf.reshape(label, y_shape), tf.uint8)

    mean = parsed_example['mean']
    stdev = parsed_example['stdev']

    return sample, label, mean, stdev

def train_fn(train_files):
    dataset = tf.data.TFRecordDataset(filenames=train_files)
    dataset = dataset.map(parse)
    
    return dataset


if __name__ == "__main__":
    # tf.compat.v1.enable_eager_execution()
    # train_files = ["/data/volume-0.tfrecord", "/data/volume-1.tfrecord"]
    train_files = ["/dl-bench/ruoyudeng/unet3d_tf_data/29gb-tf/volume-0.tfrecord",
                    "/dl-bench/ruoyudeng/unet3d_tf_data/29gb-tf/volume-1.tfrecord"]
    data = train_fn(train_files)
    
    data = data.take(1)

    data_iterator = tf.contrib.eager.Iterator(data)
    tensors = next(data_iterator) 
    # for i,tensor in enumerate(tensors):
    #     print(f"Type of tensor {i}:",type(tensor))
    #     print(f"Tensor tensor {i}:", tensor)