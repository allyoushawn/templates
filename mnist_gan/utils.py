import tensorflow as tf
import numpy as np




# Build a layer of dnn, that is , a fully-connected (fc) layer.
def single_layer_fc(x, input_dim, output_dim, activation=None, scope='fc'):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', [input_dim, output_dim],
               initializer=tf.random_uniform_initializer(minval=-0.08,
                                                         maxval=0.08))
        b = tf.get_variable('bias', [output_dim],
               initializer=tf.random_uniform_initializer(minval=-0.08,
                                                         maxval=0.08))

        # activation = None -> Linear
        if activation == None:
            return tf.matmul(x, w) + b
        else:
            return activation(tf.matmul(x, w) + b)


