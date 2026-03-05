# -*- coding: utf-8 -*-
"""
Actin Flow Quantification Pipeline (Flow Only)

This script performs:
1. Actin flow quantification using PIV (Particle Image Velocimetry)
2. Flow visualization overlaid on paxillin channel
3. Animation of flow visualization
"""

# %% Imports
import sys
sys.path.insert(0, '../utilities')

from skimage.io import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import animation

import util_animation as util

# %% Configuration

# File paths
FOLDER = '/media/rico/9d9eec64-93bc-4c31-9305-8348e35db5a4/Research/Splitting_FA_project/Sergey_Data/ActinFlow_J24'
IMAGE_NAME = 'Cell_03'

# Flow quantification settings
FLOW_CONFIG = {
    'winsize': 30,
    'ensemble_winsize': 11,      # Odd windows are ideal for proper centering
    'overlap': 15,              # 3 * winsize // 5
    'time_btwn_frames': 10,     # seconds between frames
    'nm_per_pix': 73,           # Width of a pixel in nm
    'error_thresh': 1.1,        # Error threshold for noisy correlation peaks
}

# %% Load and Prepare Data

# Load multichannel movie
movie = imread(f'{FOLDER}/{IMAGE_NAME}.tif')

# Split channels
imstack = movie[:, :, :, 0]         # Channel 0: Actin speckles
pax_imstack = movie[:, :, :, 1]     # Channel 1: Paxillin
actin_imstack = movie[:, :, :, 2]   # Channel 2: Actin

del movie  # Free memory

# %% Initialize Data Structures

# Create ensemble windows for PIV
a, b = util.create_ensemble_windows(
    FLOW_CONFIG['ensemble_winsize'], 
    imstack.shape[0] #number of frames in the movie
)

# Initialize collectors
speed_list = []
artist_list = []  # Collect artists for animation

# Normalization for paxillin visualization
norm = colors.Normalize(vmin=np.min(pax_imstack), vmax=np.max(pax_imstack))

# Create figure and axes for animation (do this once)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

# Create initial dummy contourf for colorbar
dummy_data = np.zeros((10, 10))
dummy_contour = ax2.contourf(dummy_data, vmin=0, vmax=30, 
                              levels=np.arange(0, 30, 2), cmap='viridis')
colorbar = fig.colorbar(dummy_contour, fraction=0.025, pad=0.01, ax=ax2)

# %% Process Each Ensemble Window

print('Processing frames...')
for i, j in zip(a, b):
    # Compute ensemble-averaged correlation between frames of actin speckles
    mean_corr = util.ensemble_average_corr(
        imstack[i:j].copy(),
        winsize=FLOW_CONFIG['winsize'],
        overlap=FLOW_CONFIG['overlap']
    )
    
    # Compute flow field
    x, y, u, v = util.compute_flow_field(
        mean_corr,
        im=imstack[(i + j) // 2].copy(),
        winsize=FLOW_CONFIG['winsize'],
        overlap=FLOW_CONFIG['overlap'],
        error_thresh=FLOW_CONFIG['error_thresh']
    )
    
    # Get center frame index
    center_frame = (i + j) // 2
    
    # Plot flow field and collect artists
    speed, artist_list = util.plot_flow_on_paxillin(
        x=x, y=y, u=u, v=v,
        im=imstack[center_frame].copy(),
        pax_im=pax_imstack[center_frame].copy(),
        actin=actin_imstack[center_frame].copy(),
        frame=center_frame,
        nm_per_pix=FLOW_CONFIG['nm_per_pix'],
        time_btwn_frames=FLOW_CONFIG['time_btwn_frames'],
        norm=norm,
        artist_list=artist_list,
        fig=fig,
        ax1=ax1,
        ax2=ax2,
        colorbar=colorbar
    )
    
    # Accumulate speed results
    speed_list.append(speed)
    
    print(f'  Processed frame {center_frame}')

# %% Save Results

speed_list = np.array(speed_list)

# Save velocity images
imsave(f'{FOLDER}/{IMAGE_NAME}Flux_Vel.tif', speed_list.astype(np.float32))
print(f"Saved flow velocity map to {FOLDER}/{IMAGE_NAME}Flux_Vel.tif")

# %% Create and Save Animation

print('Creating animation...')
ani = animation.ArtistAnimation(fig, artist_list, interval=100, blit=False,
                                repeat_delay=1000)

print('Saving Animation...')
ani.save(filename=f'{FOLDER}/{IMAGE_NAME}_FlowAnimation.gif', writer='pillow')
print(f'Animation saved to {FOLDER}/{IMAGE_NAME}_FlowAnimation.gif')

plt.close(fig)