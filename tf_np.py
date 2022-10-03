import tensorflow as tf
import numpy as np
def parse_single_image(image, label):
  
  #define the dictionary -- the structure -- of our single example
  features = {'X': tf.io.FixedLenFeature([], tf.string),
                    'Y': tf.io.FixedLenFeature([], tf.string),
                    'mean': tf.io.FixedLenFeature([], tf.float32),
                    'stdev': tf.io.FixedLenFeature([], tf.float32)}
  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=features))

  return out


def parse(serialized):
    """ Parse TFRecord

    :param serialized: Serialized record for a particular example
    :return: sample, label, mean and std of intensities
    """
    features = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
        'mean': tf.io.FixedLenFeature([], tf.float32),
        'stdev': tf.io.FixedLenFeature([], tf.float32)
    }

    parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                features=features)

    sample = tf.io.decode_raw(parsed_example['X'], tf.uint8)
    sample = tf.cast(tf.reshape(sample, (190, 392, 392, 1)), tf.uint8)
    label = tf.io.decode_raw(parsed_example['Y'], tf.uint8)
    label = tf.cast(tf.reshape(label, (190, 392, 392)), tf.uint8)

    mean = parsed_example['mean']
    stdev = parsed_example['stdev']

    return sample, label, mean, stdev



if __name__ == "__main__":
  print("Running")
  filename = "/data/volume-0.tfrecord"
  dataset = tf.data.TFRecordDataset(filename)

  #pass every single feature through our mapping function
  new_dataset = dataset.map(parse)
  
  new_dataset = new_dataset.take(1)
  print(new_dataset)

  data_iterator = tf.contrib.eager.Iterator(new_dataset)
  tensor = next(data_iterator)  
  # see_tensor = tensor[0,:5,:5,0]
  
  # print(see_tensor.shape)
  new_tens = tf.constant([[45, 26, 16], [22, 18, 29], [13, 28, 90]])
  new_output = tf.get_static_value(new_tens)

  print(type(new_tens))
  print("Static value:",new_output)

  # print(type(tensor))
  # print(tensor)
  # print(tf.get_static_value(tensor,partial=True))

  for t in tensor:
    print(tf.get_static_value(t,partial=True))


  # sess = tf.Session()
  # a = tf.constant(1.5)
  # print(see_tensor.numpy())
  # print(type(a))
  # print(type(tensor))

  # print(sess.run(a))
  # print(sess.run(tensor))
  