import tensorflow as tf


'''
Build a layer of dnn, that is , a fully-connected (fc) layer.
'''
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


'''
Build a layer of rnn
'''
def single_layer_rnn(cell, x, sequence_len, init_state=None):
    outputs, state = tf.nn.dynamic_rnn(cell, x,
        sequence_length=sequence_len,
        dtype=tf.float32,
        initial_state=init_state)

    return outputs, state


'''
Generate rnn cells
'''
def gen_rnn_cell(rnn_cell_num, rnn_type, dropout_keep_prob):

    cell = tf.contrib.rnn.GRUCell(rnn_cell_num)
    #cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_keep_prob)
    return cell
