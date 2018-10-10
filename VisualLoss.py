import matplotlib.pyplot as plt
import pylab as pl
import time
import pickle


class VisualLoss(object):
    'Support Elements: fake_points_loss, real_points_loss, gradient_norm_loss, gradient_direction_loss'
    # figures to plot
    index_loss_figure = None
    figLoss = None
    # parameters for plot
    visual_times = None
    # records to plot
    cnt_history = None
    fake_points_loss_history = None
    real_points_loss_history = None
    gradient_norm_loss_history = None
    gradient_direction_loss_history = None
    WGAN_loss_history = None

    def __init__(self, index_loss_figure=3):
        self.index_loss_figure = index_loss_figure
        self.figLoss = plt.figure(index_loss_figure)
        self.cnt_history = 0

    def set_visual_times(self, visual_times):
        self.visual_times = visual_times

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
            draw_pal = ['gold', 'saddlebrown', 'dimgray']
            plt.stackplot(range(max(0, index - self.visual_times), index),
                          self.WGAN_loss_history[max(0, index - self.visual_times): index],
                          self.gradient_norm_loss_history[max(0, index - self.visual_times): index],
                          self.gradient_direction_loss_history[max(0, index - self.visual_times): index],
                          colors=draw_pal, alpha=0.7)

            # -- draw fake and real expect -- #
            # draw value expect
            plt.plot(range(max(0, index - self.visual_times), index),
                     self.fake_points_loss_history[max(0, index - self.visual_times): index],
                     color='#1057AA', alpha=0.7)

            plt.plot(range(max(0, index - self.visual_times), index),
                     self.real_points_loss_history[max(0, index - self.visual_times): index],
                     color='#D0252D', alpha=0.7)

            plt.legend(labels=['Fake', 'Real', 'GAN', 'Grad_Norm', 'Grad_Direct'], loc=2)

            plt.pause(0.1)

    def save_data(self):
        name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.LOSS'
        with open('./history/' + name, 'wb') as fw:
            pickle.dump(self, fw, -1)
