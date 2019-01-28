# coding: utf-8

# %%
import os
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skimage.io as ski_io
from scipy.spatial import Voronoi, voronoi_plot_2d
from IPython.display import HTML

plt.style.use('default')
img_court = ski_io.imread('fullcourt.png')

# %%
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

# %%

plt.scatter(shots_df.LOC_X, shots_df.LOC_Y, marker='.', alpha='0.1', edgecolors='#FFFFFFFF')
plt.show()

# %%
plt.gca().set_aspect('equal', adjustable='box')
xedges = list(range(46, 96, 2))
yedges = list(range(0, 52, 2))
bins = [xedges, yedges]
h, xedges, yedges, image = plt.hist2d(shots_df.LOC_X, shots_df.LOC_Y, bins=bins)
print(h)
print(h.shape)
print(xedges)
print(yedges)
plt.show()

# %%
print(h[:, 12])
plt.plot(list(range(24)), h[:, 12])
plt.show()

#%%
outlier = shots_df.loc[shots_df.LOC_Y >= 500]
outlier.drop_duplicates(['ACTION_TYPE', 'SHOT_ZONE_RANGE', 'SHOT_DISTANCE'])
outlier = outlier.loc[:, ['ACTION_TYPE', 'SHOT_ZONE_RANGE', 'SHOT_DISTANCE']]
#%%
display((HTML(outlier.to_html())))

#%%
