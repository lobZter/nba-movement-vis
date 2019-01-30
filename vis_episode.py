# coding: utf-8

# %%
import os
import datetime
import traceback
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skimage.io as ski_io
from scipy.spatial import Voronoi, voronoi_plot_2d
from IPython.display import HTML
from shapely.geometry import Polygon
from my_voronoi import voronoi_plot_2d_inside_convex_polygon
plt.style.use('default')
img_court = ski_io.imread('fullcourt.png')

# %%
import h5py
with h5py.File('eval_result_shallow_0127_mse.hdf5', 'r') as f:
    pos_ = f['pos'][()]
    len_ = f['len'][()]
    epv_ = f['epv'][()]
    one_hot_ = f['one_hot'][()]
    val_ = np.argmax(one_hot_, axis=1)
    pos_ = pos_[:, :, -1]
    pos_ = np.reshape(pos_, [pos_.shape[0], pos_.shape[1], 11, 2])
    

# %%
# cal shot chart
shots_df = pd.read_csv('shots.csv')
shots_df = shots_df.loc[shots_df['SHOT_ZONE_RANGE'] != 'Back Court Shot']
# dataframe containing SportVU movement data that is converted to 
# the single shooting court that follows shot log dimensions (-250-250, -47.5-422.5)
shots_df['LOC_Y_'] = shots_df['LOC_X']
shots_df['LOC_X_'] = shots_df['LOC_Y']
shots_df['LOC_X'] = shots_df['LOC_X_'].apply(lambda x: (x + 47.5) / 10)
shots_df['LOC_Y'] = shots_df['LOC_Y_'].apply(lambda y: (y + 250) / 10)
shots_df['LOC_X'] = shots_df['LOC_X'].apply(lambda x: 94 - x)
shots_df['LOC_Y'] = shots_df['LOC_Y'].apply(lambda x: 50 - x)
plt.gca().set_aspect('equal', adjustable='box')
xedges = list(range(46, 96, 2))
yedges = list(range(0, 52, 2))
bins = [xedges, yedges]
h, xedges, yedges, image = plt.hist2d(shots_df.LOC_X, shots_df.LOC_Y, bins=bins)
#%%
log_h = np.log(h)
log_h = np.where(np.isnan(log_h), 0, log_h)
log_h = np.where(np.isinf(log_h), 0, log_h)
norm_log_h = log_h / np.max(log_h)
fig = plt.figure()
ax = fig.add_subplot(111)
hist = ax.imshow(norm_log_h.T, interpolation='nearest', origin='lower')
fig.colorbar(hist)
plt.show()

#%%
# init cell x_loc and y_loc
x_cell = np.arange(47, 94, 2)
y_cell = np.arange(1, 50, 2)
x_cell = np.tile(x_cell[:, None], [1, 25])
y_cell = np.tile(y_cell[None, :], [24, 1])
cell = np.concatenate([x_cell[:, :, None], y_cell[:, :, None]], axis=-1)
cell = np.tile(cell[:, :, None, :], [1, 1, 10, 1])
print(cell.shape)

#%%
# time string
# time_str = [datetime.datetime(2018, 1, 1) + datetime.timedelta(seconds=i*60/5) for i in range(5*24)]
time_list = list(range(5*24))
def time2str(arr, FPS=5):
    res = []
    for i in arr:
        res.append('{}:{:02d}'.format(i//FPS, (i%FPS)*int(60/FPS)))
    return res
time_str = time2str(time_list)
print(time_str)

# %%
class Episode():
    def __init__(self, data, length, info=None, FPS=25):
        self.data = data
        self.length = length
        self.info = info
        self.FPS = FPS
        self.portfolio()

    def portfolio(self):
        self.off_own = []
        self.def_own = []
        self.off_score = []
        self.def_score = []
        for i in range(self.length):
            dist = np.linalg.norm(cell - self.data[i, 1:11], axis=-1)
            weight = 1.0 / (dist + 1.0)
            argmin = np.tile(np.argmin(dist, axis=-1)[:, :, None], [1, 1, 10])
            mask = np.tile(np.arange(0, 10)[None, None, :], [24, 25, 1])
            own = np.where(mask == argmin, weight, 0)
            off_own = np.sum(own[:, :, 0:5], axis=-1)
            def_own = np.sum(own[:, :, 5:10], axis=-1)
            off_score = np.sum(off_own * norm_log_h)
            def_score = np.sum(def_own * norm_log_h)
            self.off_own.append(off_own)
            self.def_own.append(def_own)
            self.off_score.append(off_score)
            self.def_score.append(def_score)
        self.off_own = np.array(self.off_own)
        self.def_own = np.array(self.def_own)
        self.off_score = np.array(self.off_score)
        self.def_score = np.array(self.def_score)
    
    @property
    def ani(self):
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1]) 
        ax = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        def func(i):
            if self.info is not None:
                ax1.clear()
                ax1.plot(time_str[:self.length], self.info[:self.length], c='r')
                ax1.scatter(i, self.info[i], c='r')
                ax1.set_xticks(list(range(0,self.length, self.length//6)))

            ax.clear()
            ax.imshow(img_court, extent=[0,94,0,50])
            ax.scatter(self.data[i, 1:6, 0], self.data[i, 1:6, 1], c='r', s=100)
            ax.scatter(self.data[i, 6:11, 0], self.data[i, 6:11, 1], c='b', s=100)
            ax.scatter(self.data[i, 0, 0], self.data[i, 0, 1], c='g')
            if self.info is not None:
                epv_str = '{:.2f}'.format(self.info[i, 0])
                ax.text(1, 1, epv_str, fontsize=12)
            ax.set_xlim(-10, 104)
            ax.set_ylim(-10, 60)
            return _,

        return animation.FuncAnimation(fig, func, frames=self.length, interval=1000/self.FPS)

    @property
    def ani_voronoi(self):
        
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1, height_ratios=[8, 1, 1]) 
        ax = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        
        def func(i):
            ax1.clear()
            ax1.plot(time_str[:self.length], self.off_score, c='r')
            ax1.scatter(i, self.off_score[i], c='r')
            ax1.set_xticks(list(range(0,self.length, self.length//6)))
            
            ax2.clear()
            ax2.plot(time_str[:self.length], self.def_score, c='b')
            ax2.scatter(i, self.def_score[i], c='b')
            ax2.set_xticks(list(range(0,self.length, self.length//6)))

            ax.clear()
            ax.imshow(self.off_own[i].T, interpolation='nearest', cmap='Reds', alpha=0.5, origin='lower', extent=[46,94,0,50])
            ax.imshow(self.def_own[i].T, interpolation='nearest', cmap='Blues', alpha=0.5, origin='lower', extent=[46,94,0,50])
            ax.imshow(img_court, extent=[0,94,0,50])
            vor = Voronoi(self.data[i, 1:])
            boundary = Polygon([[47, 0], [47, 50], [94, 50], [94, 0]])
            voronoi_plot_2d_inside_convex_polygon(vor, boundary, ax=ax, show_points=False, show_vertices=False)
            ax.scatter(self.data[i, 1:6, 0], self.data[i, 1:6, 1], c='r')
            ax.scatter(self.data[i, 6:11, 0], self.data[i, 6:11, 1], c='b')
            ax.scatter(self.data[i, 0, 0], self.data[i, 0, 1], c='g')
            ax.set_xlim(40, 104)
            ax.set_ylim(-10, 60)
            return _,

        return animation.FuncAnimation(fig, func, frames=self.length, interval=1000/self.FPS)

    def show_voronoi(self, display_type='js'):
        if display_type == 'js':
            return HTML(self.ani_voronoi.to_jshtml())
        if display_type == 'html5':
            return HTML(self.ani_voronoi.to_html5_video())
        else:
            raise ValueError("display_type should be 'js' or 'html5'.")

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

    def output_voronoi(self, filename):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.FPS, metadata=dict(artist='Me'), bitrate=1800)
        self.ani_voronoi.save(filename, writer=writer)

#%%
# i = 0
# e = Episode(pos_[i], len_[i], info=epv_[i], FPS=5)
# e.show_ani()

# %%
# dirname = 'shallow_0127_mse'
# if not os.path.exists(dirname):
#     os.makedirs(dirname)
# for i in range(pos_.shape[0]):
#     try:
#         e = Episode(pos_[i], len_[i], info=epv_[i], FPS=5)
#         e.output_ani('{}/{}_{}.mp4'.format(dirname, i, val_[i]))
#     except Exception:
#         print(traceback.format_exc())
