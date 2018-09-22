import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import cholesky
import pylab as pl

# -- control -- #
to_debug = False
to_plot = True
to_test = False

# -- nn parameter -- #
X_dim = 2
z_dim = 2
h_dim = 512
D_layers = 7
G_layers = 5

# -- WGAN parameter -- #
cnt_point = 8
noise_min = -1.
noise_max = 1.
iter_G = 100
iter_D = 10
D_learning_rate = 1e-4
G_learning_rate = 1e-4
lam_grad_direction = 1
lam_grad_norm = 0.1

# -- plot parameter -- #
visual_delay = 0.1
fig3D = plt.figure(1)
fig2D = plt.figure(2)
figLoss = plt.figure(3)
ax = Axes3D(fig3D)
cnt_draw_along_axis = 80
# plot arrange
x_axis_min = -1.5
x_axis_max = 1.5
y_axis_min = -1.5
y_axis_max = 1.5

# -- prepare plot axis basis -- #
x1 = np.linspace(x_axis_min, x_axis_max, cnt_draw_along_axis)
x2 = np.linspace(y_axis_min, y_axis_max, cnt_draw_along_axis)
x1, x2 = np.meshgrid(x1, x2)
x1_vec = np.reshape(x1, (cnt_draw_along_axis ** 2, 1))
x2_vec = np.reshape(x2, (cnt_draw_along_axis ** 2, 1))
# to calc points where X_visual.shape = [None, X_dim]
X_visual = np.concatenate((x1_vec, x2_vec), axis=1)


# generator gauss
def gauss_2d(mu_1, mu_2, cnt):
    mu = np.array([[mu_1, mu_2]])
    Sigma = np.array([[0.2, 0], [0, 0.2]])
    R = cholesky(Sigma)
    s = np.dot(np.random.randn(cnt, 2), R) + mu
    return s


# calc "value = f(X_visual)" then function can draw
def plot_surface_nn(x, y, value, real_point, real_value, fake_point, fake_value, grad_visual, iter):
    z = np.reshape(value, (cnt_draw_along_axis, cnt_draw_along_axis))

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


def plot_loss_change(iter, D_fake_loss, D_real_loss, grad_norm_loss, grad_direction_loss):
    # add data to history
    D_fake_loss_rec.append(D_fake_loss)
    D_real_loss_rec.append(D_real_loss)
    GAN_loss_rec.append(D_fake_loss - D_real_loss)
    grad_norm_loss_rec.append(grad_norm_loss)
    grad_direction_loss_rec.append(grad_direction_loss)

    with plt.style.context("seaborn-whitegrid"):
        # -- draw stacked plot -- #
        pl.figure(3)
        plt.cla()
        plt.title('Current Loss View')

        # draw loss change proportion
        draw_pal = ['gold', 'dimgray', 'saddlebrown']
        plt.stackplot(range(max(0, iter - iter_D), iter), GAN_loss_rec[max(0, iter - iter_D): iter],
                      grad_direction_loss_rec[max(0, iter - iter_D): iter],
                      grad_norm_loss_rec[max(0, iter - iter_D): iter], colors=draw_pal, alpha=0.7)

        # -- draw fake and real expect -- #
        # draw value expect
        plt.plot(range(max(0, iter - iter_D), iter), D_fake_loss_rec[max(0, iter - iter_D): iter],
                 color='#1057AA', alpha=0.7)

        plt.plot(range(max(0, iter - iter_D), iter), D_real_loss_rec[max(0, iter - iter_D): iter],
                 color='#D0252D', alpha=0.7)

        plt.legend(labels=['Fake', 'Real', 'GAN', 'Grad_Direct', 'Grad_Norm'], loc=2)

        plt.pause(visual_delay)


# for debug : print every layer's mean output
def print_layer_mean_value():
    D_layer_toview = sess.run(D_layer_mean_rec, feed_dict={X_toView: X_real})
    for i in range(D_layers):
        print(i, D_layer_toview[i])


# initialize nn weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev, mean=0)


# -- set D -- #
X = tf.placeholder(tf.float32, shape=[None, X_dim])
X_toView = tf.placeholder(tf.float32, shape=[None, X_dim])
X_fake_fix = tf.placeholder(tf.float32, shape=[None, X_dim])

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
    return (np.float32)(np.random.uniform(noise_min, noise_max, size=[m, n]))


def generator(z):
    G_last = z

    for i in range(G_layers - 1):
        G_last = tf.nn.relu(tf.matmul(G_last, G_W[i]) + G_b[i])

    G_last = tf.matmul(G_last, G_W[G_layers - 1]) + G_b[G_layers - 1]

    G_out = G_last
    # G_out = tf.nn.sigmoid(G_last)
    return G_out


def discriminator(x):
    D_last = x

    for i in range(D_layers - 1):
        D_last = tf.nn.relu(tf.matmul(D_last, D_W[i]) + D_b[i])

    D_last = tf.matmul(D_last, D_W[D_layers - 1]) + D_b[D_layers - 1]

    D_out = D_last
    # D_out = tf.sigmoid(D_last)

    return D_out


def discriminator_rec(x):
    D_layer_value_rec = []
    D_last = x

    for i in range(D_layers - 1):
        D_last = tf.nn.relu(tf.matmul(D_last, D_W[i]) + D_b[i])
        D_layer_value_rec.append(tf.reduce_mean(D_last))

    D_last = tf.matmul(D_last, D_W[D_layers - 1]) + D_b[D_layers - 1]
    D_layer_value_rec.append(tf.reduce_mean(D_last))

    return D_layer_value_rec


# WGAN's G & D
if to_test:
    G_sample = X_fake_fix
else:
    G_sample = generator(z)
D_value = discriminator(X_toView)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

# for debug
D_layer_mean_rec = discriminator_rec(X_toView)

# for grad visualize
Grad_tovisual = tf.gradients(D_value, X_toView)[0]
Grad_fake = tf.gradients(D_value, X_toView)

# -- WGAN optimizer --
# X_real[j] - X_fake[i] & norm(*)
X_fake_mat = tf.reshape(G_sample, (cnt_point, 1, 2))
X_real_mat = tf.reshape(X, (1, cnt_point, 2))
X_distance = X_real_mat - X_fake_mat
X_distance_norm = tf.norm(X_distance, axis=-1)

# D_real[j] - D_fake[i]
D_diff = tf.transpose(D_real) - D_fake

# inner loop penalty
grad_pen_inner_scale = D_diff / tf.square(X_distance_norm)
grad_pen_inner_scale_mat = tf.reshape(grad_pen_inner_scale, (cnt_point, cnt_point, 1))
grad_pen_inner_mat = grad_pen_inner_scale_mat * X_distance

# grad(X_fake[i]) & norm(*)
grad = tf.gradients(D_fake, G_sample)[0]
grad_norm = tf.norm(grad, axis=1)
grad_norm_mat = tf.reshape(grad_norm, (cnt_point, 1))

# external loop penalty
grad_external = grad / grad_norm_mat
grad_external_mat = tf.reshape(grad_external, (cnt_point, 1, 2))

# two kind of grad penalty
grad_direction_pen = lam_grad_direction * tf.reduce_sum(grad_external_mat * grad_pen_inner_mat) / cnt_point ** 2
grad_norm_pen = lam_grad_norm * tf.reduce_mean(grad_norm ** 2)

# final grad penalty
grad_pen = - grad_direction_pen + grad_norm_pen

# wgan basic loss
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

# to visualize
D_real_loss_rec = []
D_fake_loss_rec = []
GAN_loss_rec = []
grad_norm_loss_rec = []
grad_direction_loss_rec = []

# for test
if (to_test):
    iter_G = 1
    iter_D = 1

# -- training -- #
for iter_g in range(iter_G):
    # train D
    X_fake = sess.run(G_sample, feed_dict={z: z_fix})
    for iter_d in range(iter_D):
        if to_test:
            X_fake = gauss_2d(0, 1, cnt_point)
            _, D_loss_curr, D_fake_mean_curr, D_real_mean_curr, grad_norm_pen_curr, grad_direction_pen_curr = sess.run(
                [D_solver, D_loss, D_fake_mean, D_real_mean, grad_norm_pen, grad_direction_pen],
                feed_dict={X: X_real, X_fake_fix: X_fake}
            )

        if not to_test:
            _, D_loss_curr, D_fake_mean_curr, D_real_mean_curr, grad_norm_pen_curr, grad_direction_pen_curr = sess.run(
                [D_solver, D_loss, D_fake_mean, D_real_mean, grad_norm_pen, grad_direction_pen],
                feed_dict={X: X_real, z: z_fix}
            )

        # for debug
        if to_debug:
            print_layer_mean_value()

        # for plot
        if to_plot:
            # calc surface and gradient data to plot
            # surface value
            Value_visual = sess.run(D_value, feed_dict={X_toView: X_visual})
            # point value
            Real_value_visual = sess.run(D_value, feed_dict={X_toView: X_real})
            Fake_value_visual = sess.run(D_value, feed_dict={X_toView: X_fake})
            # grad value
            Grad_visual = sess.run(Grad_tovisual, feed_dict={X_toView: X_fake})

            # draw the plots
            plot_surface_nn(x1, x2, Value_visual, X_real, Real_value_visual, X_fake, Fake_value_visual, Grad_visual,
                            iter_d)
            plot_loss_change(iter_g * iter_D + iter_d + 1, D_fake_mean_curr, D_real_mean_curr, grad_norm_pen_curr,
                             grad_direction_pen_curr)

            # print loss
            print('Iter:' + str(iter_d) + '; D_loss:' + str(D_loss_curr))

    if not to_test:
        # update G
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={z: z_fix}
        )

        # print loss
        print('Iter:' + str(iter_g) + '; G_loss:' + str(G_loss_curr))
