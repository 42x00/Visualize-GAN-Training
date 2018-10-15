# -- control -- #
to_debug = False
to_plot = True
to_disable_G = True
to_move_fake_manually = True
add_fake_guide = True
add_real_norm = True

# -- nn parameter -- #
X_dim = 2
z_dim = 2
h_dim = 512
D_layers = 10
G_layers = 7

# -- WGAN parameter -- #
cnt_point = 10
iter_G = 5
iter_D = 2
D_learning_rate = 1e-4
G_learning_rate = 0.8
noise_z_min = -10.
noise_z_max = 10.
lam_grad_direction = 2.0
lam_grad_norm = 0.01

# plot arrange
x_axis_min = -10
x_axis_max = 10
y_axis_min = -10
y_axis_max = 10
cnt_draw_along_axis = 80

epsilon = 1e-8
