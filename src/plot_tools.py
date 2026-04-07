#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:     Manuel Ferreira
Created:    April 2026
Updated:     
Version:    v1.0

Description:
    Utility functions for generating consistent, publication-quality figures 
    with standardized layout, fonts, margins, and axis scaling. This module 
    ensures visual uniformity and reproducibility across plots.

Includes:
    make_axes   Create a single figure/axes pair with controlled dimensions,
                margins, and LaTeX rendering.
    c2e_linear  Compute cell edges from linearly spaced center coordinates.
    c2e_log     Compute cell edges from log-spaced (or approximately log-spaced)
                center coordinates.

Notes:
    All functions return NumPy arrays or matplotlib objects. Units for figure 
    dimensions are in millimeters. Designed to produce plots suitable for 
    publication-quality figures with consistent style.
"""

import numpy as np
from matplotlib import pyplot as plt

def make_axes(wf, margins, xlim, ylim, xlabel, ylabel, *, xscale='linear', 
              yscale='linear', title=None, grid=False, **kwargs):
    """
    Create a matplotlib figure and axes with consistent publication-quality layout.

    PARAMETERS

    wf : float
        Total figure width in millimeters.
    margins : list or tuple of float
        Margins [left, right, bottom, top] in millimeters.
    xlim : tuple of float
        Limits of the x-axis as (xmin, xmax).
    ylim : tuple of float
        Limits of the y-axis as (ymin, ymax).
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    xscale : str, optional
        Scale for the x-axis ('linear', 'log', etc.). Default is 'linear'.
    yscale : str, optional
        Scale for the y-axis ('linear', 'log', etc.). Default is 'linear'.
    title : str, optional
        Title of the plot. Default is None.
    grid : bool, optional
        Whether to show grid lines. Default is False.
    **kwargs
        Additional keyword arguments passed to `ax.set()`.

    RETURNS
    
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Configured axes object.
    """
    alpha = 1
    l, r, b, t = margins
    AR = 0.618  # golden ratio aspect ratio
    # Axes width and figure height
    wa = wf - l - r
    ha = AR * wa
    hf = ha + b + t
    # Set LaTeX font rendering
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.usetex': True,
        'figure.dpi': 300
    })
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(alpha * wf / 25.4, hf / 25.4))
    # Adjust margins
    fig.subplots_adjust(
        left=l / (wf * alpha),
        bottom=b / hf,
        right=1 - r / (wf * alpha) + (1 / alpha - 1),
        top=1 - t / hf,
    )
    # Set axes properties
    ax.set(
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        xscale=xscale,
        yscale=yscale,
        axisbelow=True,
        **kwargs
    )
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    return fig, ax


def c2e_linear(centers):
    """
    Compute cell edges from linearly spaced cell centers.

    PARAMETERS

    centers : array_like
        1D array of cell center coordinates.

    RETURNS
    
    edges : ndarray of shape (len(centers)+1,)
        Array of cell edges corresponding to the input centers.
    """
    # Compute midpoints between centers
    mids = (centers[:-1] + centers[1:]) / 2.0
    # Extend the first and last edges
    left = centers[0] - (mids[0] - centers[0])
    right = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate(([left], mids, [right]))


def c2e_log(centers):
    """
    Compute cell edges from log-spaced (or approximately log-spaced) centers.

    PARAMETERS

    centers : array_like
        1D array of log-spaced cell center coordinates.

    RETURNS
    
    edges : ndarray of shape (len(centers)+1,)
        Array of cell edges maintaining proportional log spacing.

    NOTES
    
    Interior edges are computed as geometric mean of neighboring centers.
    The first and last edges are extrapolated to maintain consistent ratios.
    """
    # Interior edges = geometric mean of neighboring centers
    mids = np.sqrt(centers[:-1] * centers[1:])
    # Extend first and last edges to maintain consistent ratio
    left_edge = centers[0] ** 2 / mids[0]
    right_edge = centers[-1] ** 2 / mids[-1]
    return np.concatenate(([left_edge], mids, [right_edge]))