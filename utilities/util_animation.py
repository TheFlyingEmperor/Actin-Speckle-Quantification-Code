#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actin Flow Utilities (Flow Only)

Utility functions for:
- Flow field computation and visualization
- Image processing and upsampling
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from skimage.exposure import equalize_adapthist
from openpiv import pyprocess, validation, filters, preprocess
import pyclesperanto as cle
from skimage.filters import threshold_yen, gaussian, threshold_minimum


# =============================================================================
# Ensemble Windows
# =============================================================================

def create_ensemble_windows(e_winsize, movie_length):
    """Create start and end indices for ensemble windows."""
    return np.arange(0, movie_length - e_winsize), np.arange(e_winsize, movie_length)


# =============================================================================
# Image Processing
# =============================================================================

def sigmoid(x, thresh):
    return 1 / (1 + np.exp(-15 * (x - thresh)))


def process_img(img):
    """
    Preprocess image for PIV analysis.
    Applies normalization, CLAHE, and top-hat filtering.
    """
    maximum = np.max(img)
    img = img / maximum
    img = equalize_adapthist(img, clip_limit=0.01, nbins=256)
    thresh = threshold_yen(img)
    img = sigmoid(img, thresh)
    img = gaussian(img, sigma=0.75)
    img = np.asarray(cle.top_hat_box(img, img, 5, 5))
    return img


# =============================================================================
# Flow Field Computation
# =============================================================================

def compute_magnitude(u, v):
    """Compute velocity magnitude from u and v components."""
    return np.sqrt(u**2 + v**2)


def pixels_to_nm(speed, nm_per_pix, time_per_frame):
    """Convert speed from pixels/frame to nm/s."""
    return speed * nm_per_pix / time_per_frame


def cleanup_flow_field(corr, u, v, error_thresh):
    """
    Clean up flow field by replacing outliers.
    Uses signal-to-noise ratio validation and local mean interpolation.
    """
    sig2noise = pyprocess.sig2noise_ratio(corr)
    mask = validation.sig2noise_val(sig2noise, threshold=error_thresh)
    
    u, v = filters.replace_outliers(
        u, v,
        np.reshape(mask, u.shape),
        method='localmean',
        max_iter=10,
        kernel_size=2
    )
    return u, v


def compute_flow_field(mean_correlation, im, winsize, overlap, error_thresh):
    """
    Compute velocity field from correlation data.
    """
    grid = pyprocess.get_field_shape(im.shape, winsize, overlap)
    nrows, ncol = grid[0], grid[1]
    
    u, v = pyprocess.correlation_to_displacement(
        mean_correlation, nrows, ncol, 
        subpixel_method='gaussian'
    )
    
    x, y = pyprocess.get_coordinates(im.shape, winsize, overlap)
    u, v = cleanup_flow_field(mean_correlation, u, v, error_thresh)
    
    return x, y, u, v


# =============================================================================
# Ensemble Correlation
# =============================================================================

def ensemble_average_corr(imstack, winsize, overlap):
    """
    Compute ensemble-averaged correlation from image stack.
    """
    if len(imstack) < 2:
        raise ValueError("imstack must contain at least 2 frames for correlation")
    
    corr = []
    
    for cur_frame, next_frame in zip(imstack[:-1, :, :], imstack[1:, :, :]):
        cur_frame = process_img(cur_frame)
        cur_frame_array = pyprocess.moving_window_array(cur_frame, winsize, overlap)
        cur_frame_array = pyprocess.normalize_intensity(cur_frame_array)
        
        next_frame = process_img(next_frame)
        next_frame_array = pyprocess.moving_window_array(next_frame, winsize, overlap)
        next_frame_array = pyprocess.normalize_intensity(next_frame_array)
        
        corr.append(pyprocess.fft_correlate_images(
            cur_frame_array,
            next_frame_array,
            normalized_correlation=True
        ))
    
    return np.array(corr).mean(axis=0)


# =============================================================================
# Upsampling
# =============================================================================

def upsample_img(img, width, height, x, y):
    """
    Upsample a coarse grid image to full resolution using interpolation.
    Pads edges with zeros to cover the full image dimensions.
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    
    # Extend coordinates to cover full image bounds
    # Add 0 at the start and width/height at the end
    x_extended = np.concatenate([[0], x_unique, [width - 1]])
    y_extended = np.concatenate([[0], y_unique, [height - 1]])
    
    # Pad the image with zeros on all sides to match extended coordinates
    img_padded = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    
    interp = RegularGridInterpolator(
        (y_extended, x_extended),
        img_padded,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    
    new_x = np.arange(0, width)
    new_y = np.arange(0, height)
    xx, yy = np.meshgrid(new_x, new_y)
    
    upsampled = interp((yy, xx))
    
    return upsampled


# =============================================================================
# Visualization
# =============================================================================
def plot_flow_on_paxillin(x, y, u, v, im, pax_im, actin, frame,
                          nm_per_pix, time_btwn_frames, norm,
                          artist_list, fig, ax1, ax2, colorbar, scale=80):
    """
    Plot flow field visualization overlaid on paxillin channel.
    
    Creates a 2-panel visualization showing:
    1. Paxillin channel with flow vectors
    2. Flow speed contour map
    
    Appends artists to artist_list for animation.
    
    Returns
    -------
    speed : ndarray
        Upsampled speed map.
    artist_list : list
        Updated list of artists for animation.
    """
    # Mask flow arrows outside of the cell
    thresh_min = threshold_minimum(actin)
    mask = actin < thresh_min
    grid_mask = preprocess.prepare_mask_on_grid(x, y, mask)
    mask_u = np.ma.masked_array(u, mask=grid_mask)
    mask_v = np.ma.masked_array(v, mask=grid_mask)
    
    # Compute and upsample speed
    speed = pixels_to_nm(compute_magnitude(mask_u, mask_v), nm_per_pix, time_btwn_frames)
    speed = upsample_img(speed, im.shape[1], im.shape[0], x, y)
    
    # Panel 1: Paxillin channel with flow vectors
    pax_fig = ax1.imshow(pax_im, cmap='gray', norm=norm)
    vectors1 = ax1.quiver(x, y, mask_u, -mask_v, scale=scale, width=0.003, color='red')
    title1 = ax1.set_title(f'Paxillin with Actin Flow (Frame {frame})')
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Panel 2: Flow speed contour map
    magnitude = ax2.contourf(
        np.flip(speed, axis=0),
        vmin=0, vmax=30,
        levels=np.arange(0, 30, 2),
        cmap='viridis'
    )
    vectors2 = ax2.quiver(x, im.shape[0] - y, mask_u, -mask_v, scale=scale, width=0.003)
    colorbar.update_normal(magnitude)
    title2 = ax2.set_title('Flow Speed (nm/s)')
    ax2.set_aspect('equal')
    
    # Append all artists for this frame
    artist_list.append([pax_fig, vectors1, title1, magnitude, vectors2, title2])
    
    return speed, artist_list