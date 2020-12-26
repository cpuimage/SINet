from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def data_term_loss(y_true, y_pred):
    y_true = tf.concat((tf.ones_like(y_true) - y_true, y_true), -1)
    matching_indicator = tf.math.reduce_sum(y_true, axis=3)
    matching_penalty = tf.math.square(tf.math.subtract(y_pred[:, :, :, 0], y_true[:, :, :, 1]))
    data_term = tf.math.multiply(matching_indicator, matching_penalty)
    data_term = tf.math.reduce_sum(data_term, axis=2)
    data_term = tf.math.reduce_sum(data_term, axis=1)
    return data_term


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output_resolution = 224
    y_true = tf.ones(shape=(1, output_resolution, output_resolution, 1))
    y_pred = tf.ones(shape=(1, output_resolution, output_resolution, 1)) * 2.0

    output = data_term_loss(y_true=y_true, y_pred=y_pred)
    print(output)


if __name__ == "__main__":
    main()
