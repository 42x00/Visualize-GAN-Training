import matplotlib.pyplot as plt
import pylab as pl
import time
import pickle
import os


class VisualLoss(object):
    'Support Elements: fake_points_loss, real_points_loss, gradient_norm_loss, gradient_direction_loss'
    # figures to plot
    index_loss_figure = None
    figLoss = None
    # parameters for plot
    visual_times = None
    visual_delay = None
    # records to plot
    cnt_history = None
    fake_points_loss_history = None
    real_points_loss_history = None
    gradient_norm_loss_history = None
    gradient_direction_loss_history = None
    WGAN_loss_history = None

    def __init__(self, index_loss_figure=3, obj=None):
        self.index_loss_figure = index_loss_figure
        self.figLoss = plt.figure(index_loss_figure)
        self.visual_delay = 0.1
        if obj is None:
            self.cnt_history = 0
        else:
            self.index_loss_figure = obj.index_loss_figure
            self.figLoss = obj.figLoss
            self.visual_times = obj.visual_times
            self.visual_delay = obj.visual_delay
            self.cnt_history = obj.cnt_history
            self.fake_points_loss_history = obj.fake_points_loss_history
            self.real_points_loss_history = obj.real_points_loss_history
            self.gradient_norm_loss_history = obj.gradient_norm_loss_history
            self.gradient_direction_loss_history = obj.gradient_direction_loss_history
            self.WGAN_loss_history = obj.WGAN_loss_history

    # easy to view the result
    # def __del__(self):
    #     self.save_data()
    #     if self.fake_points_loss_history is not None:
    #         del self.fake_points_loss_history
    #     if self.real_points_loss_history is not None:
    #         del self.real_points_loss_history
    #     if self.gradient_norm_loss_history is not None:
    #         del self.gradient_norm_loss_history
    #     if self.gradient_direction_loss_history is not None:
    #         del self.gradient_direction_loss_history
    #     if self.WGAN_loss_history is not None:
    #         del self.WGAN_loss_history

    def set_visual_times(self, visual_times):
        self.visual_times = visual_times

    def set_visual_delay(self, visual_delay):
        self.visual_delay = visual_delay

    def reset_plot_location(self, index_loss_figure=3):
        self.index_loss_figure = index_loss_figure
        self.figLoss = plt.figure(index_loss_figure)
        self.visual_delay = 0.1

    def add_elements(self, tuple_plot):
        self.cnt_history = self.cnt_history + 1
        tuple_keys = tuple_plot.keys()
        for tuple_key in tuple_keys:
            if tuple_key == 'fake_points_loss':
                if self.fake_points_loss_history is None:
                    self.fake_points_loss_history = []
                self.fake_points_loss_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'real_points_loss':
                if self.real_points_loss_history is None:
                    self.real_points_loss_history = []
                    self.WGAN_loss_history = []
                self.real_points_loss_history.append(tuple_plot.get(tuple_key))
                self.WGAN_loss_history.append(tuple_plot.get('fake_points_loss') - tuple_plot.get('real_points_loss'))
                continue
            if tuple_key == 'gradient_norm_loss':
                if self.gradient_norm_loss_history is None:
                    self.gradient_norm_loss_history = []
                self.gradient_norm_loss_history.append(tuple_plot.get(tuple_key))
                continue
            if tuple_key == 'gradient_direction_loss':
                if self.gradient_direction_loss_history is None:
                    self.gradient_direction_loss_history = []
                self.gradient_direction_loss_history.append(tuple_plot.get(tuple_key))
                continue
            assert 'To know elements to plot, please refer to VisualLoss.__doc__'

    def plot(self, index=-1):
        # plot the newest surface if index if omit
        if index == -1:
            index = self.cnt_history - 1

        with plt.style.context("seaborn-whitegrid"):
            # -- draw stacked plot -- #
            pl.figure(self.index_loss_figure)
            plt.cla()
            plt.title('Current Loss View')

            # draw loss change proportion
            if self.gradient_norm_loss_history is not None and self.gradient_direction_loss_history is not None:
                draw_pal = ['gold', 'saddlebrown', 'dimgray']
                plt.stackplot(range(max(0, index + 1 - self.visual_times), index + 1),
                              self.WGAN_loss_history[max(0, index + 1 - self.visual_times): index + 1],
                              self.gradient_norm_loss_history[max(0, index + 1 - self.visual_times): index + 1],
                              self.gradient_direction_loss_history[max(0, index + 1 - self.visual_times): index + 1],
                              colors=draw_pal, alpha=0.7)

            # -- draw fake and real expect -- #
            # draw value expect
            if self.fake_points_loss_history is not None:
                plt.plot(range(max(0, index + 1 - self.visual_times), index + 1),
                         self.fake_points_loss_history[max(0, index + 1 - self.visual_times): index + 1],
                         color='#1057AA', alpha=0.7)

            if self.real_points_loss_history is not None:
                plt.plot(range(max(0, index + 1 - self.visual_times), index + 1),
                         self.real_points_loss_history[max(0, index + 1 - self.visual_times): index + 1],
                         color='#D0252D', alpha=0.7)

            plt.legend(labels=['Fake', 'Real', 'GAN', 'Grad_Norm', 'Grad_Direct'], loc=2)

            plt.pause(self.visual_delay)

    def save_data(self):
        if not os.path.exists('./history'):
            os.mkdir('./history')
        name = time.strftime("%Y-%m-%d %H-%M", time.localtime()) + '.LOSS'
        with open('./history/' + name, 'wb') as fw:
            pickle.dump(self, fw, -1)
