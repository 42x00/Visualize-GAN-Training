# Visualize-of-GAN-Training
A tool well packaged to visualize:
- (VisualNN)   3D and 2D projection of Neural Network's surface 
- (VisualLoss) Stackplot of different part of GAN LOSS

### Files

| Name | Description |
| - | - |
| [main.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/main.py) | Prepare $real$ and $fake$ data, calling tensorflow model and the tool to train GAN|
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
When training GAN model, we collect data like surface value, gradient and so on. Then we add them to the class tool, which saves data for us and allows us to plot on time or later from the history. In this way, we can easily train GAN on GPU, receiving and saving data for visualization.

### Code Structure
- [main.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/main.py)
```python
import ...

# -- prepare for surface plot -- #
myVisualNN = VisualNN()  # default: 3D - figure(1), 2D - figure(2)
myVisualNN.set_plot_arrange(x_axis_min, x_axis_max, y_axis_min, y_axis_max, cnt_draw_along_axis)  # plot range and density
X_visual = myVisualNN.generate_nn_input()  # generate input to feed the model getting surface value

# -- prepare for loss plot -- #
myVisualLoss = VisualLoss()  # default: figure(3)
myVisualLoss.set_visual_times(iter_D)  # width of sequence visual

# -- prepare tensorflow -- #
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# -- prepare real and fake data -- #
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
        
        # add model output to class by tuple formation, omit that you do not need
        tuple_plot_NN = {'surface_value': ...,  # shape: (N*N, 1)
                         'real_points_location': ...,  # shape: (N, 2)
                         'real_points_value': ...,  # shape: (N, 1)
                         'fake_points_location': ...,  # shape: (N, 2)
                         'fake_points_value': ...,  # shape: (N, 1)
                         'gradient_direction': ...  # shape: (N, 2)
                        }
        myVisualNN.add_elements(tuple_plot_NN)

        tuple_plot_Loss = {'fake_points_loss': ...,  # shape: (1, )
                           'real_points_loss': ...,  # shape: (1, )
                           'gradient_norm_loss': ...,  # shape: (1, )
                           'gradient_direction_loss': ...  # shape: (1, )
                        }
        myVisualLoss.add_elements(tuple_plot_Loss)

        # plot here if you want or in VisualHistory.py
        myVisualNN.plot()
        myVisualLoss.plot()
        
    # -- update G -- #
    ...

# -- save model -- #
# Automatically save in destructor, so omit if you run in terminal
myVisualNN.save_data()
myVisualLoss.save_data()
```

- [VisualHistory.py](https://github.com/Lyk98/Visualize-of-GAN-Training/blob/master/VisualHistory.py)
```python
import pickle

# load data from ./history
name = '...'
with open('./history/' + name + '.NN', 'rb') as fr:
    myVisualNN = pickle.load(fr)
with open('./history/' + name + '.LOSS', 'rb') as fr:
    myVisualLoss = pickle.load(fr)

# visual as you want
for i in range(myVisualNN.cnt_history):
    myVisualNN.plot(i)
    myVisualLoss.plot(i)

```

