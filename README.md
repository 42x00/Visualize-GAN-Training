# Visualize-of-GAN-Training
A tool well packaged to visualize:
- (VisualNN)   3D and 2D projection of Neural Network's surface 
- (VisualLoss) Stackplot of different part of GAN LOSS

### Files

| Name | Description |
| - | - |
| [main.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/main.py) | Prepare $real$ and $fake$ data, calling tensorflow model and the tool|
| [VisualNN.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/VisualNN.py) | A **CLASS** used to save data and plot the 3D and 2D projection of Neural Network's surface|
| [VisualLoss.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/VisualLoss.py) | A **CLASS** used to plot stackplot of different part of GAN LOSS|
| [VisualHistory.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/VisualHistory.py) | Load history records and visualize|
| [parameters.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/parameters.py) | Parameters for running control, Neural Network, GAN and plot|
| [model.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/model.py) | Implementation of **GAN** model and set calculation graph

### Demo
![conv_ops](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/demo/3D.gif)
![conv_ops](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/demo/2D.gif)
![conv_ops](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/demo/Stackplot.gif)


### Usage Guide
- main.py
```python
import ...

# -- prepare for surface plot -- #
myVisualNN = VisualNN()
myVisualNN.set_plot_arrange(x_axis_min, x_axis_max, y_axis_min, y_axis_max, cnt_draw_along_axis)
X_visual = myVisualNN.generate_nn_input()

# -- prepare for loss plot -- #
myVisualLoss = VisualLoss()
myVisualLoss.set_visual_times(iter_D)

# -- prepare tensorflow -- #
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# -- prepare data -- #
...

# -- training -- #
for iter_g in range(iter_G):
    for iter_d in range(iter_D):
    try:
        # -- update D -- #
        ...
    except:
        myVisualLoss.save_data()
        myVisualNN.save_data()
    
    # add data to history by tuple formation
    tuple_plot_NN = {'surface_value': Value_visual,
                     'real_points_location': X_real,
                     'real_points_value': Real_value_visual,
                     'fake_points_location': X_fake,
                     'fake_points_value': Fake_value_visual,
                     'gradient_direction': Grad_visual}
    myVisualNN.add_elements(tuple_plot_NN)

    tuple_plot_Loss = {'fake_points_loss': D_fake_mean_curr,
                        'real_points_loss': D_real_mean_curr,
                        'gradient_norm_loss': grad_norm_pen_curr,
                        'gradient_direction_loss': grad_direction_pen_curr
                        }
    myVisualLoss.add_elements(tuple_plot_Loss)
        
    # -- update G -- #
    ...

# -- save model -- #
myVisualNN.save_data()
myVisualLoss.save_data()
```

