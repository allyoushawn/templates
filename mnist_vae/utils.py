import tensorflow as tf
import numpy as np




# Build a layer of dnn, that is , a fully-connected (fc) layer.
def single_layer_fc(x, input_dim, output_dim, activation=None, scope='fc'):
    with tf.variable_scope(scope):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        w = tf.get_variable('weights', [input_dim, output_dim],
                                                         initializer=w_init)
        b = tf.get_variable('bias', [output_dim],
               initializer=b_init)
        '''
        w = tf.get_variable('weights', [input_dim, output_dim],
               initializer=tf.random_uniform_initializer(minval=-0.08,
                                                         maxval=0.08))
        b = tf.get_variable('bias', [output_dim],
               initializer=tf.random_uniform_initializer(minval=-0.08,
                                                         maxval=0.08))
        '''

        # activation = None -> Linear
        if activation == None:
            return tf.matmul(x, w) + b
        else:
            return activation(tf.matmul(x, w) + b)


# Perform reparametrization
def reparametrize(mu, log_var, eps=None):
    eps = tf.random_normal(tf.shape(mu), 0, 1,
      dtype=tf.float32)
    return tf.add(mu, tf.multiply(tf.sqrt(tf.exp(log_var)), eps))


# Compute point-wise log prob of Gaussian
def log_gauss(mu, logvar, x):
    """compute point-wise log prob of Gaussian"""
    x_shape = x.get_shape().as_list()

    if isinstance(mu, tf.Tensor):
        mu_shape = mu.get_shape().as_list()
    else:
        mu_shape = list(np.asarray(mu).shape)

    if isinstance(logvar, tf.Tensor):
        logvar_shape = logvar.get_shape().as_list()
    else:
        logvar_shape = list(np.asarray(logvar).shape)

    return -0.5 * (np.log(2 * np.pi) + logvar + tf.pow((x - mu), 2) / tf.exp(logvar))


def kld(mu, logvar, q_mu=None, q_logvar=None):
    """compute dimension-wise KL-divergence
    -0.5 (1 + logvar - q_logvar - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar))
    q_mu, q_logvar assumed 0 is set to None
    """
    if q_mu is None:
        q_mu = tf.zeros_like(mu)
    else:
        print("using non-default q_mu %s" % q_mu)

    if q_logvar is None:
        q_logvar = tf.zeros_like(logvar)
    else:
        print("using non-default q_logvar %s" % q_logvar)

    if isinstance(mu, tf.Tensor):
        mu_shape = mu.get_shape().as_list()
    else:
        mu_shape = list(np.asarray(mu).shape)

    if isinstance(q_mu, tf.Tensor):
        q_mu_shape = q_mu.get_shape().as_list()
    else:
        q_mu_shape = list(np.asarray(q_mu).shape)

    if isinstance(logvar, tf.Tensor):
        logvar_shape = logvar.get_shape().as_list()
    else:
        logvar_shape = list(np.asarray(logvar).shape)

    if isinstance(q_logvar, tf.Tensor):
        q_logvar_shape = q_logvar.get_shape().as_list()
    else:
        q_logvar_shape = list(np.asarray(q_logvar).shape)

    if not np.all(mu_shape == logvar_shape):
        raise ValueError("mu_shape (%s) and logvar_shape (%s) does not match" % (
            mu_shape, logvar_shape))
    if not np.all(mu_shape == q_mu_shape):
        raise ValueError("mu_shape (%s) and q_mu_shape (%s) does not match" % (
            mu_shape, q_mu_shape))
    if not np.all(mu_shape == q_logvar_shape):
        raise ValueError("mu_shape (%s) and q_logvar_shape (%s) does not match" % (
            mu_shape, q_logvar_shape))

    return -0.5 * (1 + logvar - q_logvar - \
            (tf.pow(mu - q_mu, 2) + tf.exp(logvar)) / tf.exp(q_logvar))
