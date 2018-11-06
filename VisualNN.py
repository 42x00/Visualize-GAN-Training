import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import time
import pickle
import os


class VisualNN(object):
    'Support Elements: surface_value, real_points_location, real_points_value, fake_points_location, fake_points_value, gradient_direction, expected_direction'
    # figures to plot
    index_3d_figure = None
    index_2d_figure = None
    fig3D = None
    fig2D = None
    ax = None
    # parameters for generating base to calc and plot
    x_axis_min = None
    x_axis_max = None
    y_axis_min = None
    y_axis_max = None
    cnt_draw_along_axis = None
    x_axis = None
    y_axis = None
    # parameters for plot
    visual_delay = None
    # records to plot
    cnt_history = None
    surface_value_history = None
    real_points_location_history = None
    real_points_value_history = None
    fake_points_location_history = None
    fake_points_value_history = None
    gradient_direction_history = None
    expected_direction_history = None

    def __init__(self, index_3d_figure=1, index_2d_figure=2, obj=None):
        self.index_3d_figure = index_3d_figure
        self.index_2d_figure = index_2d_figure
        self.fig3D = plt.figure(index_3d_figure)
        self.fig2D = plt.figure(index_2d_figure)
        self.ax = Axes3D(self.fig3D)
        self.visual_delay = 0.1
        if obj is None:
            self.cnt_history = 0
        else:
            self.x_axis_min = obj.x_axis_min
            self.x_axis_max = obj.x_axis_max
            self.y_axis_min = obj.y_axis_min
            self.y_axis_max = obj.y_axis_max
            self.cnt_draw_along_axis = obj.cnt_draw_along_axis
            self.x_axis = obj.x_axis
            self.y_axis = obj.y_axis
            self.visual_delay = obj.visual_delay
            self.cnt_history = obj.cnt_history
            self.surface_value_history = obj.surface_value_history
            self.real_points_location_history = obj.real_points_location_history
            self.real_points_value_history = obj.real_points_value_history
            self.fake_points_location_history = obj.fake_points_location_history
            self.fake_points_value_history = obj.fake_points_value_history
            self.gradient_direction_history = obj.gradient_direction_history
            self.expected_direction_history = obj.expected_direction_history

    # easy to view the result
    # def __del__(self):
    #     self.save_data()
    #     if self.surface_value_history is not None:
    #         del self.surface_value_history
    #     if self.real_points_location_history is not None:
    #         del self.real_points_location_history
    #     if self.real_points_value_history is not None:
    #         del self.real_points_value_history
    #     if self.fake_points_location_history is not None:
    #         del self.fake_points_location_history
    #     if self.fake_points_value_history is not None:
    #         del self.fake_points_value_history
    #     if self.gradient_direction_history is not None:
    #         del self.gradient_direction_history
    #     if self.expected_direction_history is not None:
    #         del self.expected_direction_history

    def set_plot_arrange(self, x_axis_min, x_axis_max, y_axis_min, y_axis_max, cnt_draw_along_axis):
        self.x_axis_min = x_axis_min
        self.x_axis_max = x_axis_max
        self.y_axis_min = y_axis_min
        self.y_axis_max = y_axis_max
        self.cnt_draw_along_axis = cnt_draw_along_axis

    def set_visual_delay(self, visual_delay):
        self.visual_delay = visual_delay

    def reset_plot_location(self, index_3d_figure=1, index_2d_figure=2):
        self.index_3d_figure = index_3d_figure
        self.index_2d_figure = index_2d_figure
        self.fig3D = plt.figure(index_3d_figure)
        self.fig2D = plt.figure(index_2d_figure)
        self.ax = Axes3D(self.fig3D)
        self.visual_delay = 0.1

    def generate_nn_input(self):
        # prepare plot axis basis #
        tmp_x = np.linspace(self.x_axis_min, self.x_axis_max, self.cnt_draw_along_axis)
        tmp_y = np.linspace(self.y_axis_min, self.y_axis_max, self.cnt_draw_along_axis)
        self.x_axis, self.y_axis = np.meshgrid(tmp_x, tmp_y)
        x1_vec = np.reshape(self.x_axis, (self.cnt_draw_along_axis ** 2, 1))
        x2_vec = np.reshape(self.y_axis, (self.cnt_draw_along_axis ** 2, 1))
        # to calc points where X_visual.shape = [None, X_dim]
        x_visual = np.concatenate((x1_vec, x2_vec), axis=1)
        return x_visual

    def add_elements(self, tuple_plot):
        self.cnt_history = self.cnt_history + 1
        tuple_keys = tuple_plot.keys()
        for tuple_key in tuple_keys:
            if tuple_key == 'surface_value':
                if self.surface_value_history is None:
                    self.surface_value_history = []
                self.surface_value_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'real_points_location':
                if self.real_points_location_history is None:
                    self.real_points_location_history = []
                self.real_points_location_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'real_points_value':
                if self.real_points_value_history is None:
                    self.real_points_value_history = []
                self.real_points_value_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'fake_points_location':
                if self.fake_points_location_history is None:
                    self.fake_points_location_history = []
                self.fake_points_location_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'fake_points_value':
                if self.fake_points_value_history is None:
                    self.fake_points_value_history = []
                self.fake_points_value_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'gradient_direction':
                if self.gradient_direction_history is None:
                    self.gradient_direction_history = []
                self.gradient_direction_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'expected_direction':
                if self.expected_direction_history is None:
                    self.expected_direction_history = []
                self.expected_direction_history.append(tuple_plot.get(tuple_key))
            assert 'To know elements to plot, please refer to VisualNN.__doc__'

    def plot(self, index=-1):
        # plot the newest surface if index if omit
        if index == -1:
            index = self.cnt_history - 1

        if self.surface_value_history is not None:
            value = self.surface_value_history[index]
            surface_value = np.reshape(value, (self.cnt_draw_along_axis, self.cnt_draw_along_axis))

        if self.real_points_location_history is not None:
            real_point = self.real_points_location_history[index]

        if self.real_points_value_history is not None:
            real_value = self.real_points_value_history[index]

        if self.fake_points_location_history is not None:
            fake_point = self.fake_points_location_history[index]
            cnt_point = len(fake_point)

        if self.fake_points_value_history is not None:
            fake_value = self.fake_points_value_history[index]

        if self.gradient_direction_history is not None:
            grad_visual = self.gradient_direction_history[index]

        if self.expected_direction_history is not None:
            gradient_direction_expected = self.expected_direction_history[index]

        # -- 3D plot -- #
        with plt.style.context("seaborn-whitegrid"):
            pl.figure(self.index_3d_figure)
            plt.cla()
            plt.title('3D View of ' + str(index) + ' Iter')

            # draw surface
            if self.surface_value_history is not None:
                self.ax.plot_surface(self.x_axis, self.y_axis, surface_value, rstride=1, cstride=1, cmap='coolwarm',
                                     alpha=0.7)

            # draw points
            if self.real_points_value_history is not None:
                self.ax.scatter(real_point[:, 0], real_point[:, 1], real_value, color='#D0252D')
            if self.fake_points_value_history is not None:
                self.ax.scatter(fake_point[:, 0], fake_point[:, 1], fake_value, color='#1057AA')
                cnt_point = len(fake_point)
                all_zero = np.zeros((cnt_point, 1))

            # draw gradients
            if self.gradient_direction_history is not None and self.fake_points_value_history is not None:
                self.ax.quiver(fake_point[:, 0], fake_point[:, 1], fake_value.T, grad_visual[:, 0], grad_visual[:, 1],
                               all_zero, color='black', normalize=True, lw=1, length=0.1)

            # set lim
            plt.xlim(self.x_axis_min * 1.5, self.x_axis_max * 1.5)
            plt.ylim(self.y_axis_min * 1.5, self.y_axis_max * 1.5)

        # -- 2D plot -- #
        pl.figure(self.index_2d_figure)
        plt.cla()
        plt.title('2D View ' + str(index) + ' Iter')

        # draw projection
        # plt.contourf(x, y, surface_value, 400, cmap='coolwarm', alpha=0.7)
        if self.surface_value_history is not None:
            plt.imshow(surface_value, extent=[self.x_axis_min, self.x_axis_max, self.y_axis_min, self.y_axis_max],
                       cmap='coolwarm', origin='lower')

        # draw points
        if self.real_points_location_history is not None:
            plt.scatter(real_point[:, 0], real_point[:, 1], cnt_point, color='#D0252D', marker='+')
        if self.fake_points_location_history is not None:
            plt.scatter(fake_point[:, 0], fake_point[:, 1], cnt_point, color='#1057AA', marker='+')

        # draw gradients
        if self.expected_direction_history is not None:
            plt.quiver(fake_point[:, 0], fake_point[:, 1], gradient_direction_expected[:, 0],
                       gradient_direction_expected[:, 1], color='dimgray', units='width', alpha=0.6)
        if self.gradient_direction_history is not None:
            plt.quiver(fake_point[:, 0], fake_point[:, 1], grad_visual[:, 0], grad_visual[:, 1],
                       color='black', units='width')

        # set lim
        plt.xlim(self.x_axis_min, self.x_axis_max)
        plt.ylim(self.y_axis_min, self.y_axis_max)

        plt.pause(self.visual_delay)

    def save_data(self):
        if not os.path.exists('./history'):
            os.mkdir('./history')
        name = time.strftime("%Y-%m-%d %H-%M", time.localtime()) + '.NN'
        with open('./history/' + name, 'wb') as fw:
            pickle.dump(self, fw, -1)
