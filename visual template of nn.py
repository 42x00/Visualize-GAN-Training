import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = tf.placeholder(dtype=tf.float32, shape=[None, 2])


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


W = weight_variable([2, 1])
b = bias_variable([1])

# y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x1 = np.arange(-100, 100, 2.5)
x2 = np.arange(-100, 100, 2.5)
x1, x2 = np.meshgrid(x1, x2)

xx1 = np.reshape(x1, (6400, 1))
xx2 = np.reshape(x2, (6400, 1))
X = np.concatenate((xx1, xx2), axis=1)

print(sess.run(W))
Z = sess.run(y, feed_dict={x: X})


def plot_surface_nn(x, y, v):
    z = np.reshape(v, (80, 80))
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()


plot_surface_nn(x1, x2, Z)