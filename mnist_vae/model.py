import tensorflow as tf
from utils import single_layer_fc, reparametrize, log_gauss, kld



class TrainModel(object):
    def __init__(self):
        mode = tf.contrib.learn.ModeKeys.TRAIN
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = VAE(mode)


class VAE(object):
    def __init__(self, mode):
        self.mode = mode

        # Config
        self.hid_dim = 500
        self.code_dim = 20
        self.init_lr = 1e-3
        self.feature_dim = 784


        # Input tensors
        self.x = tf.placeholder(tf.float32, [None, self.feature_dim])
        self.y_ = tf.placeholder(tf.float32, [None, self.feature_dim])

        with tf.variable_scope('inference_model'):
            hid1 = single_layer_fc(self.x, self.feature_dim, self.hid_dim,
                                  activation=tf.nn.relu, scope='hid1')
            hid2 = single_layer_fc(hid1, self.hid_dim, self.hid_dim,
                                  activation=tf.nn.relu, scope='hid2')
            z_dist = single_layer_fc(hid2, self.hid_dim, self.code_dim * 2,
                                    scope='z_dist')
            z_mean = z_dist[:, :self.code_dim]
            z_logvar = z_dist[:, self.code_dim:]
            #z_stddev = 1e-6 + tf.nn.softplus(z_dist[:, self.code_dim:])
            '''
            mean_hid = single_layer_fc(hid2, self.hid_dim, self.code_dim,
                                   activation=tf.nn.relu, scope='mean_hid')
            logvar_hid = single_layer_fc(hid2, self.hid_dim, self.code_dim,
                                   activation=tf.nn.relu, scope='logvar_hid')
            z_mean = single_layer_fc(mean_hid, self.code_dim, self.code_dim,
                                   scope='z_mean')
            z_logvar = single_layer_fc(logvar_hid, self.code_dim,
                                   self.code_dim, scope='z_logvar')
            '''
        z = reparametrize(z_mean, z_logvar)
        #z = z_mean + tf.sqrt(tf.exp(z_stddev)) * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)

        with tf.variable_scope('generative_model'):
            x_hid1 = single_layer_fc(z, self.code_dim, self.hid_dim,
                                  activation=tf.nn.relu, scope='x_hid1')
            x_hid2 = single_layer_fc(x_hid1, self.hid_dim, self.hid_dim,
                                  activation=tf.nn.relu, scope='x_hid2')
            x_dist = single_layer_fc(hid2, self.hid_dim, self.feature_dim,
                                    scope='x_dist')

            x_mean = tf.nn.sigmoid(x_dist)
            x_mean = tf.clip_by_value(x_mean, 1e-8, 1 - 1e-8)
            '''
            x_mean_hid = single_layer_fc(x_hid2, self.hid_dim,
                                   self.feature_dim,
                                   activation=tf.nn.relu, scope='x_mean_hid')
            x_logvar_hid = single_layer_fc(x_hid2, self.hid_dim,
                                   self.feature_dim,
                                   activation=tf.nn.relu, scope='x_logvar_hid')
            x_mean = single_layer_fc(x_mean_hid, self.feature_dim,
                                   self.feature_dim, activation=tf.nn.sigmoid, scope='x_mean')
            x_logvar = single_layer_fc(x_logvar_hid, self.feature_dim,
                                   self.feature_dim, scope='x_logvar')
            '''

        with tf.variable_scope('loss'):
            with tf.variable_scope('vae_lower_bound'):
                '''
                # Use the following log gauss will failed
                log_px_z = tf.reduce_mean(
                    tf.reduce_sum(log_gauss(x_mean, x_logvar, self.y_), axis=1))
                '''
                log_px_z = tf.reduce_sum(self.y_ * tf.log(x_mean) + (1 - self.y_) * tf.log(1 - x_mean), 1)
                log_px_z = tf.reduce_mean(log_px_z)
                tf.summary.scalar('log_Px_z', log_px_z)

                neg_kld = -1 * tf.reduce_sum(kld(z_mean, z_logvar),
                                                axis=1)
                neg_kld = tf.reduce_mean(neg_kld)
                #kld = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(1e-8 + tf.square(z_stddev)) - 1, 1)
                #neg_kld = -1 * tf.reduce_mean(kld)

                tf.summary.scalar('KL-Divergence', -1 * neg_kld)
                lower_bound = neg_kld + log_px_z
                tf.summary.scalar('lower bound', lower_bound)
            with tf.variable_scope('reg_loss'):
                inf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='inference_model')
                gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generative_model')
                total_vars = inf_vars + gen_vars

                weights = [ item for item in total_vars if 'bias' not in item.name]

                reg_loss = \
                 tf.reduce_mean([tf.reduce_mean(tf.square(x)) for x in weights])

                tf.summary.scalar('regularization_loss', reg_loss)
            loss = -1 * lower_bound + 0.001 * reg_loss


        #self.train_step = tf.train.AdamOptimizer(learning_rate=self.init_lr).minimize(loss)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
            capped_gvs = []
            with tf.variable_scope('gradient_computing'):
                gvs = optimizer.compute_gradients(loss)
                for grad, var in gvs:
                    if grad == None:
                        capped_gvs.append((tf.zeros_like(var), var))
                    else:
                        capped_gvs.append(
                         (tf.clip_by_value(grad, -1., 1.), var))
            self.train_step = optimizer.apply_gradients(capped_gvs)

        self.merged_summary = tf.summary.merge_all()
        self.step = 0
        self.x_mean = x_mean


    def setup_tensorboard(self, sess):
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            dir_name = 'train'
        self.writer = tf.summary.FileWriter('tensorboard/' + dir_name,
                                             sess.graph)


    def train(self, sess, x, y_):
        assert(self.mode == tf.contrib.learn.ModeKeys.TRAIN)
        feed={self.x: x, self.y_: y_}
        _, summary = sess.run([self.train_step, self.merged_summary],
                              feed_dict=feed)
        self.step += 1
        self.writer.add_summary(summary, self.step)


    def reconstruct(self, sess, x,):
        feed={self.x: x}
        return sess.run(self.x_mean,
                              feed_dict=feed)

