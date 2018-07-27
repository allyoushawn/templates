#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import pdb

# Load MNIST data in a format suited for tensorflow.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
batch_size = 50

n_batches = int(n_samples / batch_size)
for _ in range(n_batches):
    x, y = mnist.train.next_batch(batch_size)

# Test data
x_sample, y_sample = mnist.test.next_batch(5000)


