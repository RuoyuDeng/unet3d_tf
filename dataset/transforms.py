# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Transforms for 3D data augmentation """
import random
import tensorflow as tf
import numpy as np


def apply_transforms(samples, labels, transforms):
    """ Apply a chain of transforms to a pair of samples and labels """
    for _t in transforms:
        if _t is not None:
            samples, labels = _t(samples, labels)
    return samples, labels


def apply_test_transforms(samples, mean, stdev, transforms):
    """ Apply a chain of transforms to a samples using during test """
    for _t in transforms:
        if _t is not None:
            samples = _t(samples, labels=None, mean=mean, stdev=stdev)
    return samples


class PadXYZ: # pylint: disable=R0903
    """ Pad volume in three dimensiosn """
    def __init__(self, shape=None):
        """ Add padding

        :param shape: Target shape
        """
        self.shape = shape

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Padded samples and labels
        """
        paddings = tf.constant([[0, 0], [0, 0], [0, 5], [0, 0]])
        samples = tf.pad(samples, paddings, "CONSTANT")
        if labels is None:
            return samples
        labels = tf.pad(labels, paddings, "CONSTANT")
        return samples, labels


class CenterCrop: # pylint: disable=R0903
    """ Produce a central crop in 3D """
    def __init__(self, shape):
        """ Create op

        :param shape: Target shape for crop
        """
        self.shape = shape

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Cropped samples and labels
        """
        
        shape = samples.get_shape()
        delta = [(shape[i].value - self.shape[i]) // 2 for i in range(len(self.shape))]
        
        samples = samples[
            delta[0]:delta[0] + self.shape[0],
            delta[1]:delta[1] + self.shape[1],
            delta[2]:delta[2] + self.shape[2]]
        if labels is None:
            return samples
        labels = labels[
            delta[0]:delta[0] + self.shape[0],
            delta[1]:delta[1] + self.shape[1],
            delta[2]:delta[2] + self.shape[2]]
        return samples, labels


class RandomCrop3D: # pylint: disable=R0903
    """ Produce a random 3D crop """
    def __init__(self, shape, margins=(0, 0, 0)):
        """ Create op

        :param shape: Target shape
        :param margins: Margins within to perform the crop
        """
        self.shape = shape
        self.margins = margins

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Cropped samples and labels
        """
        
        shape = samples.shape()
        # print("RandomCrop3D initial shape:", shape)
        # print("Type of sample's shape:", type(shape))
        # print("shape[0].value:", shape[0].value)
        # min_ = tf.constant(self.margins, dtype=tf.float32)
        # max_ = tf.constant([shape[0].value - self.shape[0] - self.margins[0],
        #                     shape[1].value - self.shape[1] - self.margins[1],
        #                     shape[2].value - self.shape[2] - self.margins[2]],
        #                    dtype=tf.float32)

        print("sample shape: ", shape)
        index0 = np.random.randint(low=self.margins[0], high=(shape[0].value - self.shape[0] - self.margins[0]+1))
        index1 = np.random.randint(low=self.margins[1], high=(shape[1].value - self.shape[1] - self.margins[1]+1))
        index2 = np.random.randint(low=self.margins[2], high=(shape[2].value - self.shape[2] - self.margins[2]+1))
        center = tf.constant([index0,index1,index2],dtype=tf.float32)
        # center = np.random.randint()
        # center = tf.random.uniform((len(self.shape),), minval=min_, maxval=max_, dtype=tf.float32)
        # print("Content of the first element of center:", tf.get_static_value(center,partial=True))
        # print("Type of center:", type(center))
        # print("Shape of center:", center.shape)
        center = tf.cast(center, dtype=tf.int32)
        # print("value of center:", tf.get_static_value(center))
        
        # added by us
        # max_ = tf.cast(max_,dtype=tf.int32)
        samples = samples[center[0]:center[0] + self.shape[0],
                          center[1]:center[1] + self.shape[1],
                          center[2]:center[2] + self.shape[2]]
        # samples = samples[max_[0]:max_[0] + self.shape[0],
        #                   max_[1]:max_[1] + self.shape[1],
        #                   max_[2]:max_[2] + self.shape[2]]
        if labels is None:
            return samples
        labels = labels[center[0]:center[0] + self.shape[0],
                        center[1]:center[1] + self.shape[1],
                        center[2]:center[2] + self.shape[2]]

        # labels = labels[max_[0]:max_[0] + self.shape[0],
        #                   max_[1]:max_[1] + self.shape[1],
        #                   max_[2]:max_[2] + self.shape[2]]
        # print("RandomCrop3D input shape [0]:", self.shape[0])
        # print("RandomCrop3D input shape [0] type:", type(self.shape[0]))
        # print("RandomCrop3D shape after crop:", samples.shape)
        return samples, labels


class RandBalancedCrop:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, images, labels):
        #print("images shape:", images.shape())
        print("images type:", type(images))
        #data_tuples = [self._rand_crop(image,label) for image,label in zip(images,labels)]
        #images = np.array([data[0] for data in data_tuples])
        #labels = np.array([data[1] for data in data_tuples])
        return images, labels

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):
        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(scipy.ndimage.measurements.label(label==cl)[0])
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]


class NormalizeImages: # pylint: disable=R0903
    """ Run zscore normalization """
    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean
        :param stdev:  Std
        :return: Normalized samples and labels
        """
        mask = tf.math.greater(samples, 0)
        samples = tf.where(mask, (samples - tf.cast(mean, samples.dtype)) / (tf.cast(stdev + 1e-8, samples.dtype)),
                           samples)

        if labels is None:
            return samples
        return samples, labels


class Cast:
    def __init__(self, types=(tf.float32, tf.int8)):
        self.types = types

    def __call__(self, samples, labels, mean, stdev):
        if labels is None:
            return tf.cast(samples, dtype=self.types[0])
        return tf.cast(samples, dtype=self.types[0]), tf.cast(labels, dtype=self.types[1])


class RandFlip:
    def __init__(self):
        self.axis = [0, 1, 2]
        self.prob = 1 / len(self.axis)

    def __call__(self, samples, labels, mean, stdev):
        for axis in self.axis:
            h_flip = tf.random_uniform([]) < self.prob
            
            for axis in self.axis:
                samples = tf.cond(h_flip, lambda: tf.reverse(samples, axis=[axis]), lambda: samples)
                labels = tf.cond(h_flip, lambda: tf.reverse(labels, axis=[axis]), lambda: labels)

        return samples, labels


class RandomGammaCorrection: # pylint: disable=R0903
    """ Random gamma correction over samples """
    def __init__(self, gamma_range=(0.8, 1.5), keep_stats=False, threshold=0.5, epsilon=1e-8):
        self._gamma_range = gamma_range
        self._keep_stats = keep_stats
        self._eps = epsilon
        self._threshold = threshold

    def __call__(self, samples, labels, mean, stdev):
        """ Run op

        :param samples: Sample arrays
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: Gamma corrected samples
        """
        augment = tf.random_uniform([]) > self._threshold
        gamma = tf.random_uniform([], minval=self._gamma_range[0], maxval=self._gamma_range[1])

        x_min = tf.math.reduce_min(samples)
        x_range = tf.math.reduce_max(samples) - x_min

        samples = tf.cond(augment,
                          lambda: tf.math.pow(((samples - x_min) / float(x_range + self._eps)),
                                              gamma) * x_range + x_min,
                          lambda: samples)
        return samples, labels


class RandomBrightnessAugmentation:
    def __init__(self, factor, prob):
        self.prob = prob  # threshold
        self.factor = factor  # alpha

    def __call__(self, samples, labels, mean, stdev):
        augment =tf.random_uniform([])  < self.prob
        # mask = tf.math.greater(sample, 0)
        augmentation = tf.random_uniform([1],
                                         minval = 1.0 - self.factor,
                                         maxval = 1.0 + self.factor,
                                         dtype = samples.dtype)
        samples = tf.cond(augment,
                          lambda: samples * (1 + augmentation),
                          lambda: samples)
        
        return samples, labels


class GaussianNoise:
    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, samples, labels, mean, stdev):
        h_gauss = tf.random_uniform([]) < self.prob

        scale = tf.random_uniform([],
                                  minval = 0.0,
                                  maxval = self.std,
                                  dtype = samples.dtype)
        noise = tf.random.normal(tf.shape(samples),
                                 mean = self.mean,
                                 stddev = scale,
                                 dtype = samples.dtype)

        samples = tf.cond(h_gauss,
                          lambda: samples + noise,
                          lambda: samples)
        
        return samples, labels


class OneHot:
    '''One hot (new ver)'''
    def __init__(self, n_classes=1):
        self._n_classes = n_classes
    
    def __call__(self, samples, labels):
        def one_hot(a, num_classes):
            return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
        return samples, one_hot(labels, self._n_classes)


class OneHotLabels: # pylint: disable=R0903
    """ One hot encoding of labels """
    def __init__(self, n_classes=1):
        self._n_classes = n_classes

    def __call__(self, samples, labels):
        """ Run op

        :param samples: Sample arrays (unused)
        :param labels: Label arrays
        :param mean: Mean (unused)
        :param stdev:  Std (unused)
        :return: One hot encoded labels
        """
        print("Type of items in labels:", labels.dtype)
        #print("Type of self._n_classes:", type(self._n_classes))
        return samples, tf.one_hot(labels, self._n_classes)
