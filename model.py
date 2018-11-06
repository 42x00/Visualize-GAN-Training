import tensorflow as tf
from parameters import *


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
        tmp = tf.matmul(D_last, D_W[i]) + D_b[i]
        D_last = tf.nn.softplus(2.0 * tmp + 2.0) / 2.0 - 1.0
        # D_last = tf.nn.selu(tmp)

    D_last = tf.matmul(D_last, D_W[D_layers - 1]) + D_b[D_layers - 1]

    D_out = D_last
    # D_out = tf.sigmoid(D_last)

    return D_out


# WGAN's G & D
if to_disable_G:
    G_sample = X_fake_fix
else:
    G_sample = generator(z)
D_value = discriminator(X_toView)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

# for grad visualize
Grad_tovisual = tf.gradients(D_value, X_toView)[0]

# -- WGAN optimizer --
X_fake_mat = tf.reshape(G_sample, (cnt_point, 1, 2))
X_fake_transpose_mat = tf.reshape(G_sample, (1, cnt_point, 2))
X_real_mat = tf.reshape(X, (1, cnt_point, 2))

# grad_direction_penalty : real -> fake
# X_real[j] - X_fake[i] & norm(*)
X_distance_rf = X_real_mat - X_fake_mat
X_distance_norm_rf = tf.norm(X_distance_rf, axis=-1)

# D_real[j] - D_fake[i]
D_diff_rf = tf.transpose(D_real) - D_fake
if not add_fake_guide:
    D_diff_rf = tf.maximum(D_diff_rf, 0)

# inner loop penalty : real -> fake
if use_slope:
    grad_pen_inner_scale_rf = D_diff_rf / tf.square(X_distance_norm_rf)
else:
    grad_pen_inner_scale_rf = 1.0 / tf.pow(X_distance_norm_rf, 3)
grad_pen_inner_scale_mat_rf = tf.reshape(grad_pen_inner_scale_rf, (cnt_point, cnt_point, 1))
grad_pen_inner_mat_rf = grad_pen_inner_scale_mat_rf * X_distance_rf

# grad(X_real[i]) & norm(*)
grad_real = tf.gradients(D_real, X)[0]
grad_real_norm = tf.norm(grad_real, axis=1)

# grad(X_fake[i]) & norm(*)
grad_fake = tf.gradients(D_fake, G_sample)[0]
grad_fake_norm = tf.norm(grad_fake, axis=1)
grad_fake_norm_mat = tf.reshape(grad_fake_norm, (cnt_point, 1))

# grad_direction_penalty : fake -> fake
# X_fake[j] - X_fake[i] & norm(*)
X_distance_ff = X_fake_transpose_mat - X_fake_mat
X_distance_norm_ff = tf.norm(X_distance_ff, axis=-1) + tf.eye(cnt_point)

# D_fake[j] - D_fake[i]
D_diff_ff = tf.transpose(D_fake) - D_fake

# inner loop penalty : fake -> fake
if use_slope:
    grad_pen_inner_scale_ff = D_diff_ff / tf.square(X_distance_norm_ff)
else:
    grad_pen_inner_scale_ff = 1.0 / tf.pow(X_distance_norm_ff, 3)
grad_pen_inner_scale_mat_ff = tf.reshape(grad_pen_inner_scale_ff, (cnt_point, cnt_point, 1))
grad_pen_inner_mat_ff = grad_pen_inner_scale_mat_ff * X_distance_ff

# external loop penalty
grad_external = grad_fake / grad_fake_norm_mat
grad_external_mat = tf.reshape(grad_external, (cnt_point, 1, 2))

# two kind of grad penalty
if add_fake_guide:
    grad_expected_direction = tf.reduce_sum(grad_pen_inner_mat_rf - grad_pen_inner_mat_ff, axis=1)
    grad_direction_pen = lam_grad_direction * tf.reduce_sum(
        grad_external * grad_expected_direction
    ) / cnt_point ** 2
else:
    grad_direction_pen = lam_grad_direction * (
        tf.reduce_sum(grad_external_mat * grad_pen_inner_mat_rf)) / cnt_point ** 2

if add_real_norm:
    grad_norm_pen = lam_grad_norm * (
            tf.reduce_mean(grad_fake_norm ** 2) + tf.reduce_mean(grad_real_norm ** 2)
    )
else:
    grad_norm_pen = lam_grad_norm * tf.reduce_mean(grad_fake_norm ** 2)

# final grad penalty
grad_pen = - grad_direction_pen + grad_norm_pen

# wgan basic loss
D_fake_mean = tf.reduce_mean(D_fake)
D_real_mean = tf.reduce_mean(D_real)

D_loss = D_fake_mean - D_real_mean + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdadeltaOptimizer(learning_rate=D_learning_rate)
            .minimize(D_loss, var_list=theta_D))
if not to_disable_G:
    G_solver = (tf.train.AdamOptimizer(learning_rate=G_learning_rate)
                .minimize(G_loss, var_list=theta_G))


# -- for debug -- #
def discriminator_rec(x):
    D_layer_value_rec = []
    D_last = x

    for i in range(D_layers - 1):
        D_last = tf.nn.relu(tf.matmul(D_last, D_W[i]) + D_b[i])
        D_layer_value_rec.append(tf.reduce_mean(D_last))

    D_last = tf.matmul(D_last, D_W[D_layers - 1]) + D_b[D_layers - 1]
    D_layer_value_rec.append(tf.reduce_mean(D_last))

    return D_layer_value_rec


if to_debug:
    D_layer_mean_rec = discriminator_rec(X_toView)
