# coding: utf-8

import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skimage.io as ski_io
from IPython.display import HTML

img_court = ski_io.imread('fullcourt.png')

class Episode():
    def __init__(self, data, length, info=None, FPS=25):
        self.data = data
        self.length = length
        self.info = info
        self.FPS = FPS
    
    @property
    def ani(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        def func(i):
            ax.clear()
            ax.imshow(img_court, extent=[0,94,0,50])
            if self.info is not None:
                ax.text(1, 1, str(self.info[i]), fontsize=12)
            ax.scatter(self.data[i, 1:6, 0], self.data[i, 1:6, 1], c='r', s=100)
            ax.scatter(self.data[i, 6:11, 0], self.data[i, 6:11, 1], c='b', s=100)
            ax.scatter(self.data[i, 0, 0], self.data[i, 0, 1], c='g')
            ax.set_xlim(-10, 104)
            ax.set_ylim(-10, 60)
            return _,

        return animation.FuncAnimation(fig, func, frames=self.length, interval=1000/self.FPS)
    
    def show_ani(self, display_type='js'):
        if display_type == 'js':
            return HTML(self.ani.to_jshtml())
        if display_type == 'html5':
            return HTML(self.ani.to_html5_video())
        else:
            raise ValueError("display_type should be 'js' or 'html5'.")
        
    def output_ani(self, filename):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.FPS, metadata=dict(artist='Me'), bitrate=1800)
        self.ani.save(filename, writer=writer)
