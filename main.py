import numpy as np
from numpy.linalg import cholesky
from numpy.linalg import norm
from VisualNN import VisualNN
from VisualLoss import VisualLoss
from model import *
import tensorflow as tf
from parameters import *

# -- prepare for surface plot -- #
myVisualNN = VisualNN()
myVisualNN.set_plot_arrange(x_axis_min, x_axis_max, y_axis_min, y_axis_max, cnt_draw_along_axis)
X_visual = myVisualNN.generate_nn_input()

# -- prepare for loss plot -- #
myVisualLoss = VisualLoss()
myVisualLoss.set_visual_times(iter_D)

# -- prepare tensorflow -- #
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.Session()
sess.run(tf.global_variables_initializer())


# -- prepare data -- #
# generator gauss
def gauss_2d(mu_1, mu_2, cnt):
    mu = np.array([[mu_1, mu_2]])
    Sigma = np.array([[2, 0], [0, 2]])
    R = cholesky(Sigma)
    gauss = np.dot(np.random.randn(cnt, 2), R) + mu
    return gauss


#    case 0: random
# X_real = sample_z(cnt_point, X_dim)

#    case 1: single gauss
X_real = gauss_2d(18, 0, int(cnt_point))

#    case 2:  o  x       oxoxox
# X_fake = gauss_2d(0.7, 0, cnt_point - 1)
# X_fake = np.append(X_fake, [[-0.8, 0]], axis=0)
# X_real = gauss_2d(0.7, 0, cnt_point - 1)
# X_real = np.append(X_real, [[-0.5, 0]], axis=0)

#    case 3:  oooo    xxxx    oooo
# X_real_1 = gauss_2d(18, 0, int(cnt_point / 2))
# X_real_2 = gauss_2d(18, 0, int(cnt_point / 2))
# X_real_3 = gauss_2d(0, 2, int(cnt_point / 3))
# X_real = np.concatenate(X_real_1, X_real_2)


def sample_z(m, n):
    return np.float32(np.random.uniform(noise_z_min, noise_z_max, size=[m, n]))


z_fix = sample_z(cnt_point, X_dim)


# -- to debug -- #
# print every layer's mean output
def print_layer_mean_value():
    D_layer_toview = sess.run(D_layer_mean_rec, feed_dict={X_toView: X_real})
    for i in range(D_layers):
        print(i, D_layer_toview[i])


# for test
if to_disable_G:
    if not to_move_fake_manually:
        iter_G = 1
        iter_D = 1000
        X_fake = gauss_2d(-8, 0, int(cnt_point))
    else:
        # X_fake = sample_z(cnt_point, X_dim)
        X_fake = gauss_2d(-18, 0, int(cnt_point))

# -- training -- #
for iter_g in range(iter_G):
    if not to_move_fake_manually:
        X_fake = sess.run(G_sample, feed_dict={z: z_fix})

    print('iter', iter_g)

    # train D
    for iter_d in range(iter_D):
        try:
            if not to_disable_G:
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_real, z: z_fix})
            else:
                _, D_loss_curr, = sess.run([D_solver, D_loss], feed_dict={X: X_real, X_fake_fix: X_fake})
        except:
            myVisualLoss.save_data()
            myVisualNN.save_data()

        # for debug
        if to_debug:
            print_layer_mean_value()

        # for plot
        if to_plot:
            if fast_plot:
                if iter_d != 0:
                    continue
            # calc surface and gradient data to plot
            # surface value
            Value_visual = sess.run(D_value, feed_dict={X_toView: X_visual})
            # point & grad value
            Real_value_visual = sess.run(D_value, feed_dict={X_toView: X_real})
            Fake_value_visual, Grad_visual = sess.run([D_value, Grad_tovisual], feed_dict={X_toView: X_fake})

            # loss
            Grad_expected, D_fake_mean_curr, D_real_mean_curr, grad_norm_pen_curr, grad_direction_pen_curr = sess.run(
                [grad_expected_direction, D_fake_mean, D_real_mean, grad_norm_pen, grad_direction_pen],
                feed_dict={X: X_real, X_fake_fix: X_fake}
            )

            # draw the plots
            tuple_plot_NN = {'surface_value': Value_visual,
                             'real_points_location': X_real,
                             'real_points_value': Real_value_visual,
                             'fake_points_location': X_fake,
                             'fake_points_value': Fake_value_visual,
                             'gradient_direction': Grad_visual,
                             'expected_direction': Grad_expected
                             }
            myVisualNN.add_elements(tuple_plot_NN)
            'Support Elements: surface_value, real_distribution_location, real_points_location, real_points_value, fake_distribution_location, fake_points_location, fake_points_value, gradient_direction, expected_direction'


            tuple_plot_Loss = {'fake_points_loss': D_fake_mean_curr,
                               'real_points_loss': D_real_mean_curr,
                               'gradient_norm_loss': grad_norm_pen_curr,
                               'gradient_direction_loss': grad_direction_pen_curr
                               }
            myVisualLoss.add_elements(tuple_plot_Loss)

            # if iter_d % 5 == 0:
            # myVisualNN.plot()
            # myVisualLoss.plot()

            print loss
            print('Iter:' + str(iter_d) + '; D_loss:' + str(D_loss_curr))

    if not to_disable_G:
        # update G
        _, G_loss_curr = sess.run(
            [G_solver, G_loss],
            feed_dict={z: z_fix}
        )

        # print loss
        # print('Iter:' + str(iter_g) + '; G_loss:' + str(G_loss_curr))
    else:
        if to_move_fake_manually:
            # Grad_visual = sess.run(Grad_tovisual, feed_dict={X_toView: X_fake})
            Grad_expected = sess.run(grad_expected_direction, feed_dict={X: X_real, X_fake_fix: X_fake})
            X_fake = X_fake + G_learning_rate * Grad_expected / (
                    np.linalg.norm(Grad_expected, axis=1, keepdims=True) + epsilon)

myVisualNN.save_data()
myVisualLoss.save_data()
