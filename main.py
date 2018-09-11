import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mb_size = 10
X_dim = 2
z_dim = 2
h_dim = 40

plt.style.use("seaborn-whitegrid")
fig = plt.figure()
ax = Axes3D(fig)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# -- D -- #
X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

# -- G -- #
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-2)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-2)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

X_mb = sample_z(10, 2)
z_fix = sample_z(10, 2)
X_fake = sess.run(G_sample, feed_dict={z: z_fix})
x1 = np.arange(-1, 1, 0.025)
x2 = np.arange(-1, 1, 0.025)
x1, x2 = np.meshgrid(x1, x2)

xx1 = np.reshape(x1, (6400, 1))
xx2 = np.reshape(x2, (6400, 1))
X_visual = np.concatenate((xx1, xx2), axis=1)


def plot_surface_nn(x, y, v):
    plt.cla()

    z = np.reshape(v, (80, 80))

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='coolwarm', alpha = 0.7)
    ax.scatter(X_mb[:, 0], X_mb[:, 1], -5, color='r')
    ax.scatter(X_fake[:, 0], X_fake[:, 1], -5, color='b')

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    ax.set_zlim(-5, 5)

    plt.pause(0.3)


for it in range(1):
    for _ in range(100):
        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, z: z_fix}
            # feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )
        V_visual = sess.run(D_real, feed_dict={X: X_visual})
        plot_surface_nn(x1, x2, V_visual)

    # _, G_loss_curr = sess.run(
    #     [G_solver, G_loss],
    #     feed_dict={z: sample_z(mb_size, z_dim)}
    # )

    # print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
    #       .format(it, D_loss_curr, G_loss_curr))

