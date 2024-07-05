# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from tinybig.visual.visualizer import visualizer
import matplotlib.pyplot as plt


class plot_visualizer(visualizer):

    def __init__(self, name='plot_visualizer', *args, **kwargs):
        super().__init__(self, name=name, *args, **kwargs)

    def plot(self, data: dict | list = None, function_data: dict | list = None,
             x_label: str = None, y_label: str = None, *args, **kwargs):
        legends = []
        if data is not None:
            x, y, legend = [], [], None
            if type(data) is dict:
                x = data['x']
                y = data['y']
                legend = data['legend'] if 'legend' in data else None
            elif type(data) is list:
                x = data[0]
                y = data[1]
                legend = data[2] if len(data) == 3 else None
            plt.plot(x, y, 'bo')
            legends.append(legend)
        if function_data is not None:
            x, y, legend = [], [], None
            if type(function_data) is dict:
                x = function_data['x']
                y = function_data['y']
                legend = function_data['legend'] if 'legend' in function_data else None
            elif type(function_data) is list:
                x = function_data[0]
                y = function_data[1]
                legend = function_data[2] if len(function_data) == 3 else None
            plt.plot(x, y, 'r-')
            legends.append(legend)
        plt.xlabel(xlabel=x_label)
        plt.ylabel(ylabel=y_label)
        plt.legend(legends)
        plt.show()
