import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl

# -- nn parameter -- #
X_dim = 2
z_dim = 2
h_dim = 40

# -- WGAN parameter -- #
cnt_point = 10
iter_G = 1
iter_D = 100

# -- plot parameter -- #
fig3D = plt.figure(1)
fig2D = plt.figure(2)
ax = Axes3D(fig3D)
cnt_interval = 80
# plot arrange
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
# to calc points where X_visual.shape = [None, X_dim]
X_visual = np.concatenate((x1_vec, x2_vec), axis=1)


# calc "value = f(X_visual)" then function can draw
def plot_surface_nn(x, y, value, real_point, fake_point, grad_visual):
    z = np.reshape(value, (cnt_interval, cnt_interval))

    # -- 3D plot -- #
    with plt.style.context("seaborn-whitegrid"):
        pl.figure(1)
        plt.cla()
        plt.title('3D View')

        # draw surface
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='coolwarm', alpha=0.7)

        # draw points
        ax.scatter(real_point[:, 0], real_point[:, 1], 0, color='#D0252D')
        ax.scatter(fake_point[:, 0], fake_point[:, 1], 0, color='#1057AA')

        # draw gradients
        ax.quiver(fake_point[:, 0], fake_point[:, 1], np.zeros((cnt_point, 1)), grad_visual[:, 0], grad_visual[:, 1],
                  np.zeros((cnt_point, 1)), length=0.1, color='black', normalize=True, arrow_length_ratio=0.4,
                  lw=1)

        # set lim
        plt.xlim(x_axis_min * 1.5, x_axis_max * 1.5)
        plt.ylim(y_axis_min * 1.5, y_axis_max * 1.5)

    # -- 2D plot -- #
    pl.figure(2)
    plt.cla()
    plt.title('2D View')

    # draw projection
    # plt.contourf(x, y, z, 400, cmap='coolwarm', alpha=0.7)
    plt.imshow(z, extent=[-1, 1, -1, 1], cmap='coolwarm', origin='lower')

    # draw points
    plt.scatter(real_point[:, 0], real_point[:, 1], color='#D0252D', marker='+')
    plt.scatter(fake_point[:, 0], fake_point[:, 1], color='#1057AA', marker='+')

    # draw gradients
    plt.quiver(fake_point[:, 0], fake_point[:, 1], grad_visual[:, 0], grad_visual[:, 1],
               color='black', units='width')

    # set lim
    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)

    plt.pause(0.3)


# initialize nn weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# -- set D -- #
X_toView = tf.placeholder(tf.float32, shape=[None, X_dim])
X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W3 = tf.Variable(xavier_init([h_dim, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

# -- set G -- #
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


# -- set WGAN -- #
def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2 + D_b2))
    out = tf.matmul(D_h2, D_W3) + D_b3
    return out


G_sample = generator(z)
D_value = discriminator(X_toView)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

Grad_fake = tf.gradients(D_value, X_toView)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-2)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-2)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# -- prepare data -- #
X_real = sample_z(10, 2)
z_fix = sample_z(10, 2)
X_fake = sess.run(G_sample, feed_dict={z: z_fix})

for iter_g in range(iter_G):
    # train D
    for iter_d in range(iter_D):
        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_real, z: z_fix}
            # feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

        # calc surface and gradient data to plot
        Value_visual = sess.run(D_value, feed_dict={X_toView: X_visual})
        Grad_visual = sess.run(Grad_fake, feed_dict={X_toView: X_fake})
        plot_surface_nn(x1, x2, Value_visual, X_real, X_fake, Grad_visual[0])

    # update G
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(cnt_point, z_dim)}
    )

    # print loss
    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
          .format(iter_g, D_loss_curr, G_loss_curr))
