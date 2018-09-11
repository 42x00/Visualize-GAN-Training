import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -- plot parameter -- #
plt.style.use("seaborn-whitegrid")

fig = plt.figure()
ax = Axes3D(fig)

cnt_interval = 80
x_axis_min = -1
x_axis_max = 1
y_axis_min = -1
y_axis_max = 1

# -- prepare plot axis -- #
x1 = np.linspace(x_axis_min, x_axis_max, cnt_interval)
x2 = np.linspace(y_axis_min, y_axis_max, cnt_interval)
x1, x2 = np.meshgrid(x1, x2)
x1_vec = np.reshape(x1, (cnt_interval ** 2, 1))
x2_vec = np.reshape(x2, (cnt_interval ** 2, 1))

X_visual = np.concatenate((x1_vec, x2_vec), axis=1)


def plot_surface_nn(x, y, value):
    plt.cla()

    # draw surface
    z = np.reshape(value, (cnt_interval, cnt_interval))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='coolwarm', alpha=0.7)

    # set lim
    plt.xlim(x_axis_min * 1.5, x_axis_max * 1.5)
    plt.ylim(y_axis_min * 1.5, y_axis_max * 1.5)

    plt.pause(0.3)


# -- build nn -- #
x = tf.placeholder(dtype=tf.float32, shape=[None, 2])


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


W = weight_variable([2, 1])
b = bias_variable([1])

y = tf.nn.softmax(tf.matmul(x, W) + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# calc surface data and plot
Z = sess.run(y, feed_dict={x: X_visual})
plot_surface_nn(x1, x2, Z)
