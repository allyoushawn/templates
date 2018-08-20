import tensorflow as tf
from utils import single_layer_fc



class TrainModel(object):
    def __init__(self):
        mode = tf.contrib.learn.ModeKeys.TRAIN
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = GAN(mode)


class GAN(object):
    def __init__(self, mode):
        self.mode = mode
        self.code_dim = 100
        self.feature_dim = 784
        lr = 0.0002

        # Input tensors
        self.x = tf.placeholder(tf.float32, [None, self.feature_dim])
        self.z = tf.placeholder(tf.float32, [None, self.code_dim])

        with tf.variable_scope('generator'):
             h0 = single_layer_fc(self.z, self.z.get_shape()[1], 256,
                                  activation=tf.nn.relu, scope='h0')
             h1 = single_layer_fc(h0, h0.get_shape()[1], 512,
                                  activation=tf.nn.relu, scope='h1')
             h2 = single_layer_fc(h1, h1.get_shape()[1], 1024,
                                  activation=tf.nn.relu, scope='h2')
             G_z = single_layer_fc(h2, h2.get_shape()[1], 784,
                                  activation=tf.nn.tanh, scope='G_z')
        with tf.variable_scope('discriminator') as scope:
             h0 = single_layer_fc(self.x, self.x.get_shape()[1], 1024,
                                  activation=tf.nn.relu, scope='h0')
             h1 = single_layer_fc(h0, h0.get_shape()[1], 512,
                                  activation=tf.nn.relu, scope='h1')
             h2 = single_layer_fc(h1, h1.get_shape()[1], 256,
                                  activation=tf.nn.relu, scope='h2')
             D_real = single_layer_fc(h2, h2.get_shape()[1], 1,
                                  activation=tf.nn.sigmoid, scope='D')
             scope.reuse_variables()
             h0 = single_layer_fc(G_z, G_z.get_shape()[1], 1024,
                                  activation=tf.nn.relu, scope='h0')
             h1 = single_layer_fc(h0, h0.get_shape()[1], 512,
                                  activation=tf.nn.relu, scope='h1')
             h2 = single_layer_fc(h1, h1.get_shape()[1], 256,
                                  activation=tf.nn.relu, scope='h2')
             D_fake = single_layer_fc(h2, h2.get_shape()[1], 1,
                                  activation=tf.nn.sigmoid, scope='D')
        with tf.variable_scope('loss'):
            eps = 1e-2
            D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
            G_loss = tf.reduce_mean(-tf.log(D_fake + eps))
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_loss', G_loss)
        with tf.variable_scope('train'):

            # trainable variables for each network
            t_vars = tf.trainable_variables()
            D_vars = [var for var in t_vars if 'discriminator' in var.name]
            G_vars = [var for var in t_vars if 'generator' in var.name]

            # optimizer for each network
            self.D_train_step = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
            self.G_train_step = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)
        self.merged_summary = tf.summary.merge_all()
        self.step = 0
        self.G_z = G_z


    def setup_tensorboard(self, sess):
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            dir_name = 'train'
        self.writer = tf.summary.FileWriter('tensorboard/' + dir_name,
                                             sess.graph)


    def train(self, sess, x,  z_):
        assert(self.mode == tf.contrib.learn.ModeKeys.TRAIN)
        feed={self.x: x, self.z: z_}
        sess.run([self.G_train_step, self.D_train_step], feed_dict=feed)
        self.step += 1


    def summary(self, sess, x, z_):
        feed={self.x: x, self.z: z_}
        summary = sess.run(self.merged_summary,
                              feed_dict=feed)
        self.writer.add_summary(summary, self.step)


    def sample_img(self, sess, z_):
        assert(self.mode == tf.contrib.learn.ModeKeys.TRAIN)
        feed={self.z: z_}
        return sess.run(self.G_z, feed_dict=feed)
