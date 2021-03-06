#!/usr/bin/env python3
import tensorflow as tf
from model import TrainModel
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import pdb


# Init vars, summaries
def initialize(train_model, train_sess):
    with train_model.graph.as_default():
        train_sess.run(tf.global_variables_initializer())
        train_model.model.setup_tensorboard(train_sess)


def run_training(train_model, train_sess, x, y_):
    z_ = np.random.normal(0, 1, (batch_size, 100))
    with train_model.graph.as_default():
        train_model.model.train(train_sess, x, z_)
        train_model.model.summary(train_sess, x, z_)


def sample_img(train_model, train_sess):
    z_ = np.random.normal(0, 1, (batch_size, 100))
    with train_model.graph.as_default():
        return train_model.model.sample_img(train_sess, z_)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples
    batch_size = 500
    epoch_num = 5000

    tr_model = TrainModel()
    tr_sess = tf.Session(target='', graph=tr_model.graph)
    initialize(tr_model, tr_sess)
    n_batches = int(n_samples / batch_size)

    x_sample = mnist.test.next_batch(100)[0]

    plt.figure(figsize=(8, 12))
    step = 0

    for epoch in range(epoch_num):
        for _ in range(n_batches):
            x, y = mnist.train.next_batch(batch_size)
            run_training(tr_model, tr_sess, x, x)
            step += 1
            if step % 10 != 0: continue

            x_sampled = sample_img(tr_model, tr_sess)
            for i in range(5):
                plt.subplot(5, 2, 2 * i + 1)
                plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                plt.title("Test input")
                plt.subplot(5, 2, 2 * i + 2)
                plt.imshow(x_sampled[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                plt.title("Reconstruction")
            plt.pause(1e-17)
