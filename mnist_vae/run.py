#!/usr/bin/env python3
import tensorflow as tf
from model import TrainModel
from tensorflow.examples.tutorials.mnist import input_data
import pdb


# Init vars, summaries
def initialize(train_model, train_sess):
    with train_model.graph.as_default():
        train_sess.run(tf.global_variables_initializer())
        train_model.model.setup_tensorboard(train_sess)


def run_training(train_model, train_sess, x, y_):
    with train_model.graph.as_default():
        train_model.model.train(train_sess, x, y_)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples
    batch_size = 500
    epoch_num = 5000

    tr_model = TrainModel()
    tr_sess = tf.Session(target='', graph=tr_model.graph)
    initialize(tr_model, tr_sess)
    n_batches = int(n_samples / batch_size)

    for epoch in range(epoch_num):
        for _ in range(n_batches):
            x, y = mnist.train.next_batch(batch_size)
            run_training(tr_model, tr_sess, x, x)
