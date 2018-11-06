
# -- control -- #
to_debug = False
to_plot = True
to_disable_G = True
to_move_fake_manually = True
add_fake_guide = True
add_real_norm = True
little_test = True
fast_plot = True
use_slope = False

# -- nn parameter -- #
X_dim = 2
z_dim = 2
h_dim = 512
D_layers = 10
G_layers = 7

# -- WGAN parameter -- #
cnt_point = 20
if little_test:
    iter_G = 150
    iter_D = 1
else:
    iter_G = 30
    iter_D = 10
D_learning_rate = 1e-2
G_learning_rate = 1
noise_z_min = -10.
noise_z_max = 10.
lam_grad_direction = 3
lam_grad_norm = 0.01

# plot arrange
x_axis_min = -25
x_axis_max = 25
y_axis_min = -25
y_axis_max = 25
cnt_draw_along_axis = 80

epsilon = 1e-8
