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


