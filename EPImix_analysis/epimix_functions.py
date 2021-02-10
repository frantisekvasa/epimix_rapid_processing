#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:38:35 2020

@author: Frantisek Vasa (fdv247@gmail.com)

Additional functions for manuscript "Rapid processing and quantitative evaluation of multicontrast EPImix scans for adaptive multimodal imaging"

"""

# Additional functions to run EPImix comparison script

# for wbplot
from wbplot import pscalar
import numpy as np 
from matplotlib import cm, lines
#from matplotlib import colors 
#from matplotlib.colors import ListedColormap

# for nilearn masked plot
import nibabel as nb
import nilearn as nl

# colorbar
import matplotlib as mpl
import matplotlib.pyplot as plt

# for spin p-value
import scipy as sp

# formatting of p-values as powers of 10, modified from:
# https://stackoverflow.com/questions/25750170/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot/49330649#49330649
def pow_10_fmt(p):
    if p < 1e-10:
        return 'P < $10^{-10}$'
    elif p > 0.001:
        return 'P = '+str(round(p,3))#'%s' % float('%.3f' % p)
    else:
        s = "%1.2e" % p
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "P = ${}$".format(s)

# plot of high-res (360-ROI) parcellation, excluding "dropped" regions
def pscalar_mmp_hk(file_out, pscalars_hk, mmp_hk, orientation='landscape',
            hemi=None, vrange=None, cmap='magma', transp=False):
    
    # set vrange if it wasn't set before
    if vrange is None:
        vrange = (min(pscalars_hk),max(pscalars_hk))
    
    # replace "dropped" regions values by value smaller than range (for mmp_h, there are 360 ROIs -> hardcoded)
    pscalars = np.ones(360)*(vrange[0]-1); pscalars[mmp_hk] = pscalars_hk    
    
    # # edit colorbar to add grey as minimum value
    # cmap_nan = cm.get_cmap(cmap, 256).colors
    # cmap_nan[0,0:3] = colors.to_rgb('grey')
    # cmap_nan_mpl=ListedColormap(cmap_nan)
    
    # edit colorbar to add grey as minimum value
    cmap_under = cm.get_cmap(cmap, 256)
    cmap_under.set_under('grey')
    
    # call pscalar function with new values
    pscalar(file_out, pscalars, orientation='landscape',
        hemisphere=hemi, vrange=vrange, cmap=cmap_under, transparent=transp) # cmap_nan_mpl
    
# plot of low-res (44-ROI) parcellation, excluding "dropped" regions
def pscalar_mmp_lk(file_out, pscalars_lk, mmp_lk, mmp_ds_ids, orientation='landscape',
            hemi=None, vrange=None, cmap='magma', transp=False):
    
        # set vrange if it wasn't set before
    if vrange is None:
        vrange = (min(pscalars_lk),max(pscalars_lk))
    
    # replace "dropped" regions values by value smaller than range (for mmp_h, there are 44 ROIs -> hardcoded)
    pscalars = np.ones(44)*(vrange[0]-1); pscalars[mmp_lk] = pscalars_lk    
    
    # # edit colorbar to add grey as minimum value
    # cmap_nan = cm.get_cmap(cmap, 256).colors
    # cmap_nan[0,0:3] = colors.to_rgb('grey')
    # cmap_nan_mpl=ListedColormap(cmap_nan)
    
    # edit colorbar to add grey as minimum value
    cmap_under = cm.get_cmap(cmap, 256)
    cmap_under.set_under('grey')
    
    # set vrange if it wasn't set before
    if vrange is None:
        vrange = (min(pscalars_lk),max(pscalars_lk))
    
    # call pscalar function with new values
    pscalar(file_out, pscalars[mmp_ds_ids], orientation='landscape',
        hemisphere=hemi, vrange=vrange, cmap=cmap_under, transparent=transp) # cmap_nan_mpl

# plot colorbar
def plot_cbar(c_lim, cmap_nm, c_label, lbs, save_path):
    
    f, ax = plt.subplots(figsize=(6, 0.75)); f.subplots_adjust(bottom=0.65)
    cmap = cm.get_cmap(cmap_nm, 256) #mpl.cm.plasma_r
    norm = mpl.colors.Normalize(vmin=c_lim[0], vmax=c_lim[1])
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label(c_label, size=lbs)
    if save_path[-3:]=='png':
        plt.savefig(save_path, dpi=500)
    elif save_path[-3:]=='svg':
        plt.savefig(save_path)
    
# Median Absolute Deviation
def mad(a, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    """
    # Median along given axis, but *keeping* the reduced axis so that result can still broadcast against a.
    med = np.nanmedian(a, axis=axis, keepdims=True)
    mad = np.nanmedian(np.absolute(a - med), axis=axis)  # MAD along given axis
    return mad

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def plot_nl_image_masked(img_vec,mask_vec,img_shape,img_affine,cmap,clim=None,*line_args,**line_kwargs):
    if clim is None:
        #clim = (min(img_vec[mask_vec==1]),max(img_vec[mask_vec==1]))
        clim = (min(img_vec[mask_vec==1]),np.percentile(img_vec[mask_vec==1],95))
    # i) edit image and colorbar to map background to black
    img_masked = np.ones(img_vec.size)*(clim[0]-1); img_masked[mask_vec==1] = img_vec[mask_vec==1]
    cmap_under = cm.get_cmap(cmap, 256); cmap_under.set_under('white')
    # ii) convert image to nii and plot
    img_masked_nii = nb.Nifti1Image(np.reshape(img_masked,img_shape),affine=img_affine)
    nl.plotting.plot_img(img_masked_nii,colorbar=True,cmap=cmap_under, vmin=clim[0], vmax=clim[1],*line_args,**line_kwargs)

def add_subnetwork_lines(hm,roi_nums,*line_args,**line_kwargs):
    hm.hlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_xlim(),*line_args,**line_kwargs); hm.vlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_ylim(),*line_args,**line_kwargs)

def add_subnetwork_colours(hm,bb,roi_nums,roi_cols,*line_args,**line_kwargs):
    # add  network colour lines
    ax2 = plt.axes([0,0,1,1], facecolor=(1,1,1,0)); ax2.axis("off"); #ax2.get_xaxis().set_visible(False), ax2.get_yaxis().set_visible(False)
    temp_nroi_cum = [0]+[i-0.25 for i in np.cumsum(roi_nums)]
    for i in range(len(roi_nums)):
        ax2.add_line(lines.Line2D([bb[0,0]-0.02*(bb[1,0]-bb[0,0]) ,bb[0,0]-0.02*(bb[1,0]-bb[0,0])], [bb[1,1]-(bb[1,1]-bb[0,1])*(temp_nroi_cum[i]/sum(roi_nums)) ,bb[1,1]-(bb[1,1]-bb[0,1])*(temp_nroi_cum[i+1]/sum(roi_nums))], color=roi_cols[i], *line_args,**line_kwargs))
        ax2.add_line(lines.Line2D([bb[0,0]+(bb[1,0]-bb[0,0])*(temp_nroi_cum[i]/sum(roi_nums)) ,bb[0,0]+(bb[1,0]-bb[0,0])*(temp_nroi_cum[i+1]/sum(roi_nums))], [bb[1,1]+0.02*(bb[1,1]-bb[0,1]) ,bb[1,1]+0.02*(bb[1,1]-bb[0,1])], color=roi_cols[i], *line_args,**line_kwargs))

def adjust_lightness(color, amount=0.5):
    # from: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def perm_sphere_p(x,y,perm_id,corr_type='spearman'):
    
    # Function to generate a p-value for the spatial correlation between two parcellated cortical surface maps, 
    # using a set of spherical permutations of regions of interest.
    # The function performs the permutation in both directions; i.e.: by permute both measures, 
    # before correlating each permuted measure to the unpermuted version of the other measure
    #
    # Inputs:
    # x                 one of two maps to be correlated                                                                    vector
    # y                 second of two maps to be correlated                                                                 vector
    # perm_id           array of permutations, from set of regions to itself (as generated by "rotate_parcellation")        array of size [n(total regions) x nrot]
    # corr_type         type of correlation                                                                                 "spearman" (default) or "pearson"
    #
    # Output:
    # p_perm            permutation p-value
  
    nroi = perm_id.shape[0]  # number of regions
    nperm = perm_id.shape[1] # number of permutations
    
    if corr_type=='spearman':
        rho_emp = sp.stats.spearmanr(x,y)[0]
    elif corr_type=='pearson':
        rho_emp = sp.stats.pearsonr(x,y)[0]
    
    # permutation of measures
    x_perm = y_perm = np.zeros((nroi,nperm))
    for r in range(nperm):
        for i in range(nroi):
            x_perm[i,r] = x[perm_id[i,r]]
            y_perm[i,r] = y[perm_id[i,r]]
    
    # correlation to unpermuted measures
    rho_null_xy = np.zeros(nperm)
    rho_null_yx = np.zeros(nperm)
    if corr_type=='spearman':
        for r in range(nperm):
            rho_null_xy[r] = sp.stats.spearmanr(x_perm[:,r],y)[0]
            rho_null_yx[r] = sp.stats.spearmanr(y_perm[:,r],x)[0]
    elif corr_type=='pearson':
        for r in range(nperm):
            rho_null_xy[r] = sp.stats.pearsonr(x_perm[:,r],y)[0]
            rho_null_yx[r] = sp.stats.pearsonr(y_perm[:,r],x)[0]
    
    # p-value definition depends on the sign of the empirical correlation
    if (rho_emp>0):
        p_perm_xy = sum(rho_null_xy>rho_emp)/nperm
        p_perm_yx = sum(rho_null_yx>rho_emp)/nperm
    else:
        p_perm_xy = sum(rho_null_xy<rho_emp)/nperm
        p_perm_yx = sum(rho_null_yx<rho_emp)/nperm
    
    # return average p-value
    return((p_perm_xy+p_perm_yx)/2)