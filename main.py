import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl

# -- nn parameter -- #
X_dim = 2
z_dim = 2
h_dim = 512
D_layers = 7
G_layers = 5

# -- WGAN parameter -- #
cnt_point = 10
noise_min = -1.
noise_max = 1.
iter_G = 100
iter_D = 120
D_learning_rate = 1e-5
G_learning_rate = 1e-3
lam = 0.1

# -- plot parameter -- #
visual_delay = 0.01
fig3D = plt.figure(1)
fig2D = plt.figure(2)
figloss = plt.figure(3)
ax = Axes3D(fig3D)
cnt_interval = 80
# plot arrange
x_axis_min = -1.5
x_axis_max = 1.5
y_axis_min = -1.5
y_axis_max = 1.5

# -- prepare plot axis -- #
x1 = np.linspace(x_axis_min, x_axis_max, cnt_interval)
x2 = np.linspace(y_axis_min, y_axis_max, cnt_interval)
x1, x2 = np.meshgrid(x1, x2)
x1_vec = np.reshape(x1, (cnt_interval ** 2, 1))
x2_vec = np.reshape(x2, (cnt_interval ** 2, 1))
# to calc points where X_visual.shape = [None, X_dim]
X_visual = np.concatenate((x1_vec, x2_vec), axis=1)


# calc "value = f(X_visual)" then function can draw
def plot_surface_nn(x, y, value, real_point, real_value, fake_point, fake_value, grad_visual, iter):
    z = np.reshape(value, (cnt_interval, cnt_interval))

    # -- 3D plot -- #
    with plt.style.context("seaborn-whitegrid"):
        pl.figure(1)
        plt.cla()
        plt.title('3D View of ' + str(iter) + ' Iter')

        # draw surface
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='coolwarm', alpha=0.7)

        # draw points
        ax.scatter(real_point[:, 0], real_point[:, 1], real_value, color='#D0252D')
        ax.scatter(fake_point[:, 0], fake_point[:, 1], fake_value, color='#1057AA')

        # draw gradients
        ax.quiver(fake_point[:, 0], fake_point[:, 1], fake_value.T, grad_visual[:, 0], grad_visual[:, 1],
                  np.zeros((cnt_point, 1)), color='black', normalize=True, lw=1, length=0.1)

        # set lim
        plt.xlim(x_axis_min * 1.5, x_axis_max * 1.5)
        plt.ylim(y_axis_min * 1.5, y_axis_max * 1.5)

    # -- 2D plot -- #
    pl.figure(2)
    plt.cla()
    plt.title('2D View ' + str(iter) + ' Iter')

    # draw projection
    # plt.contourf(x, y, z, 400, cmap='coolwarm', alpha=0.7)
    plt.imshow(z, extent=[x_axis_min, x_axis_max, y_axis_min, y_axis_max], cmap='coolwarm', origin='lower')

    # draw points
    plt.scatter(real_point[:, 0], real_point[:, 1], color='#D0252D', marker='+')
    plt.scatter(fake_point[:, 0], fake_point[:, 1], color='#1057AA', marker='+')

    # draw gradients
    plt.quiver(fake_point[:, 0], fake_point[:, 1], grad_visual[:, 0], grad_visual[:, 1],
               color='black', units='width')

    # set lim
    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)

    plt.pause(visual_delay)


def plot_loss_change(iter, D_fake_loss, D_real_loss, grad_loss):
    # add data to history
    D_fake_loss_rec.append(D_fake_loss)
    D_real_loss_rec.append(-D_real_loss)
    GAN_loss_rec.append(-(D_fake_loss - D_real_loss))
    grad_loss_rec.append(grad_loss)

    with plt.style.context("seaborn-whitegrid"):
        # -- draw stacked plot -- #
        pl.figure(3)
        plt.cla()
        plt.title('Loss Stack View')

        # draw loss change proportion
        draw_labels = ['GAN', 'Grad']
        draw_pal = ['gold', 'dimgray']
        plt.stackplot(range(max(0, iter - iter_D), iter), GAN_loss_rec[max(0, iter - iter_D): iter],
                      grad_loss_rec[max(0, iter - iter_D): iter], labels=draw_labels,
                      colors=draw_pal, alpha=0.7)
        plt.legend(loc=2)

        # -- draw fake and real expect -- #
        pl.figure(4)
        plt.cla()
        plt.title('Value Expect View')

        # darw value expect
        plt.fill_between(range(max(0, iter - iter_D), iter), 0, D_fake_loss_rec[max(0, iter - iter_D): iter],
                         color='#1057AA', alpha=0.7)

        plt.fill_between(range(max(0, iter - iter_D), iter), 0, D_real_loss_rec[max(0, iter - iter_D): iter],
                         color='#D0252D', alpha=0.7)

        plt.legend(labels=['Fake', 'Real'], loc=2)

        plt.pause(visual_delay)


# initialize nn weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev, mean=0)


# -- set D -- #
X_toView = tf.placeholder(tf.float32, shape=[None, X_dim])
X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W = []
D_b = []

D_W.append(tf.Variable(xavier_init([X_dim, h_dim])))
D_b.append(tf.Variable(tf.zeros(shape=[h_dim])))

for i in range(D_layers - 2):
    D_W.append(tf.Variable(xavier_init([h_dim, h_dim])))
    D_b.append(tf.Variable(tf.zeros(shape=[h_dim])))

D_W.append(tf.Variable(xavier_init([h_dim, 1])))
D_b.append(tf.Variable(tf.zeros(shape=[1])))

theta_D = []

for i in range(D_layers):
    theta_D.append(D_W[i])

for i in range(D_layers):
    theta_D.append(D_b[i])

# -- set G -- #
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W = []
G_b = []

G_W.append(tf.Variable(xavier_init([z_dim, h_dim])))
G_b.append(tf.Variable(tf.zeros(shape=[h_dim])))

for i in range(G_layers - 2):
    G_W.append(tf.Variable(xavier_init([h_dim, h_dim])))
    G_b.append(tf.Variable(tf.zeros(shape=[h_dim])))

G_W.append(tf.Variable(xavier_init([h_dim, X_dim])))
G_b.append(tf.Variable(tf.zeros(shape=[X_dim])))

theta_G = []

for i in range(G_layers):
    theta_G.append(G_W[i])

for i in range(G_layers):
    theta_G.append(G_b[i])


# -- set WGAN -- #
def sample_z(m, n):
    return np.random.uniform(noise_min, noise_max, size=[m, n])


def generator(z):
    G_last = z

    for i in range(G_layers - 1):
        G_last = tf.nn.relu(tf.matmul(G_last, G_W[i]) + G_b[i])

    G_last = tf.matmul(G_last, G_W[G_layers - 1]) + G_b[G_layers - 1]

    G_out = tf.nn.sigmoid(G_last)
    return G_out


def discriminator(x):
    D_last = x

    for i in range(D_layers - 1):
        D_last = tf.nn.relu(tf.matmul(D_last, D_W[i]) + D_b[i])

    D_last = tf.matmul(D_last, D_W[D_layers - 1]) + D_b[D_layers - 1]

    D_out = tf.log(D_last)
    return D_out


G_sample = generator(z)
D_value = discriminator(X_toView)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

Grad_fake = tf.gradients(D_value, X_toView)

eps = tf.random_uniform([cnt_point, 1], minval=0., maxval=1.)
X_inter = eps * X + (1. - eps) * G_sample
grad = tf.gradients(discriminator(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1) ** 2)

D_fake_mean = tf.reduce_mean(D_fake)
D_real_mean = tf.reduce_mean(D_real)

D_loss = D_fake_mean - D_real_mean + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=D_learning_rate)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=G_learning_rate)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# -- prepare data -- #
X_real = sample_z(cnt_point, X_dim)
z_fix = sample_z(cnt_point, X_dim)
D_real_loss_rec = []
D_fake_loss_rec = []
GAN_loss_rec = []
grad_loss_rec = []

for iter_g in range(iter_G):
    # train D
    # z_fix = sample_z(cnt_point, X_dim)
    X_fake = sess.run(G_sample, feed_dict={z: z_fix})
    for iter_d in range(iter_D):
        _, D_loss_curr, D_fake_mean_curr, D_real_mean_curr, grad_pen_curr = sess.run(
            [D_solver, D_loss, D_fake_mean, D_real_mean, grad_pen],
            feed_dict={X: X_real, z: z_fix}
            # feed_dict={X: X_real, z: sample_z(cnt_point, z_dim)}
        )

        # calc surface and gradient data to plot
        Value_visual = sess.run(D_value, feed_dict={X_toView: X_visual})
        Real_value_visual = sess.run(D_value, feed_dict={X_toView: X_real})
        Fake_value_visual = sess.run(D_value, feed_dict={X_toView: X_fake})
        Grad_visual = sess.run(Grad_fake, feed_dict={X_toView: X_fake})
        plot_surface_nn(x1, x2, Value_visual, X_real, Real_value_visual, X_fake, Fake_value_visual, Grad_visual[0],
                        iter_d)
        plot_loss_change(iter_g * iter_D + iter_d + 1, D_fake_mean_curr, D_real_mean_curr, grad_pen_curr)

    # update G
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: z_fix}
        # feed_dict={z: sample_z(cnt_point, z_dim)}
    )

    # print loss
    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
          .format(iter_g, D_loss_curr, G_loss_curr))
