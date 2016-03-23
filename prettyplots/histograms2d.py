"""
Code to generate 2D histograms of embedded spectral data.

Authored by OGT 3/15/16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astroML.plotting import setup_text_plots
import matplotlib as mpl
plt.ion()

setup_text_plots(fontsize=18)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('font', size=18, family='serif', style='normal', variant='normal', stretch='normal', weight='bold')
mpl.rc('legend', labelspacing=0.1, handlelength=2, fontsize=12)

def get_contours(x, y, bins=(50, 50), ranges=None):
    if ranges:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=ranges)
    else:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xmid = (xedges[1:] + xedges[:-1]) / 2.0
    ymid = (yedges[1:] + yedges[:-1]) / 2.0
    return xmid, ymid, H.T

def plot_2Dhist(x, y, xlabel=None, ylabel=None, cblabel=None, ranges=[[-0.007, 0.002],[-0.014, 0.005]], vmin=1, vmax=10**5, normed=False,
                filename=None, annotate_string=None, annotate_loc=None):
    xmid, ymid, H = get_contours(x, y, bins=(50, 50), ranges=ranges)
    
    if normed == True:
        norm = np.sum(H, axis=0)  # get total number of galaxies in each mass bin
        # hacky way to make sure division is done correctly
        H_norm = np.zeros(np.shape(H))
        for col in range(np.shape(H)[1]):
            H_norm[:, col] = H[:, col] / norm[col]
    else:
        H_norm = H

    fig, ax = plt.subplots(figsize=(6.5, 5))
    plt.gcf().subplots_adjust(bottom=0.15)
    # LogNorm was 0.001, 0.5
    plt.imshow(H_norm, origin='lower', aspect='auto',
               interpolation='nearest', cmap=plt.cm.viridis, norm=LogNorm(vmin=vmin, vmax=vmax),
               extent=(ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]), vmin=vmin,
               vmax=vmax)  # cmap=plt.cm.binary, cmap=plt.cm.cubehelix_r

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    cb = plt.colorbar()
    if cblabel:
        cb.set_label(cblabel)
    
    if annotate_string:
        plt.annotate(annotate_string, annotate_loc, fontsize=16)

    plt.draw()
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        
def plot_2Dhist_medians(x, y, z, xlabel=None, ylabel=None, cblabel=None, ranges=[[-0.007, 0.002],[-0.014, 0.005]], vmin=0.0, vmax=10.0,
                filename=None):
    
    xedges = np.linspace(ranges[0][0], ranges[0][1], 51) # these numbers chosen to get 50 bins in final plot
    yedges = np.linspace(ranges[1][0], ranges[1][1], 51)

    xbins = np.digitize(x, xedges) # values falling below min(xedges) assigned 0; values above max(xedges) assigned 51
    ybins = np.digitize(y, yedges)
    
    medians = np.zeros((50,50))
    for i in range(50):
        for j in range(50):
            medians[i,j] = np.nanmedian(z[(xbins == i+1) * (ybins == j+1)])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.imshow(medians.T, origin='lower', aspect='auto',
               interpolation='nearest', cmap=plt.cm.viridis, vmin=vmin, vmax=vmax,
               extent=(ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]))

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    cb = plt.colorbar()
    if cblabel:
        cb.set_label(cblabel)

    plt.draw()
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
