#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:17:09 2020

@author: Frantisek Vasa (fdv247@gmail.com)

Evaluation of processing pipelines using Mindboggle-101 dataset, as described in the manuscript 
"Rapid processing and quantitative evaluation of multicontrast EPImix scans for adaptive multimodal imaging"

"""

# %% import libraries

import numpy as np
import scipy as sp
import nibabel as nb

# plotting
import matplotlib.pyplot as plt; plt.rcParams.update({'font.size': 13})
import ptitprince as pt
import nilearn as nl
import nilearn.plotting
import matplotlib as mpl
#from matplotlib.lines import Line2D # custom legend

# home directory
from pathlib import Path
home_dir = str(Path.home()) # home directory

# custom functions
import os

os.chdir(home_dir+'/Desktop/scripts')
import epimix_functions as ef #from epimix_functions import pscalar_mmp_hk, pscalar_mmp_lk, plot_cbar, mad, kth_diag_indices
#from netneurotools_stats import gen_spinsamples

# change plot font
plt.rcParams["font.family"] = "arial"

# %% set up

# directories
home_dir = str(Path.home())                                         # home directory
mindboggle_dir = home_dir+'/Data/Mindboggle/Mindboggle101_volumes'  # Mindboggle data directory
plot_dir =  home_dir+'/Desktop/Mindboggle_compare_pipelines_plots'  # plot directory
if not os.path.isdir(plot_dir):                                     # if plot directory doesn't exist, create it
    os.mkdir(plot_dir)
    os.mkdir(plot_dir+'/png')
    os.mkdir(plot_dir+'/svg')

# pipeline parameters - names (for plot labels), keywords (for filenames) and colors (for plots)
pip_nm = ['1mm SyN','2mm SyN','3mm SyN','2mm N4 SyN','2mm N4 BET SyN','2mm N4 spl-SyN'] # '2mm N4 mask SyN',
pip_kw = ['1mm','2mm','3mm','n4','bet','bsyn'] # 'mask',
pip_col = ['gold','darkorange','red','deepskyblue','darkorchid','darkturquoise'] # 'yellowgreen',
npip = len(pip_nm) # number of pipelines

# DKT atlas ROI IDs (1002,1003,1005-1035,2002,2003,2005-2035)
temp_id = np.concatenate((np.arange(2,4),np.arange(5,32),np.arange(34,36)))
dkt_roi = np.concatenate((1000+temp_id,2000+temp_id)); del temp_id #sorted(np.array(list(set(dkt_1mm_vec)),dtype='int'))[1:len(set(dkt_1mm_vec))] #np.concatenate((np.arange(2,32),np.arange(34,36)))
nroi = len(dkt_roi) # number of ROIs

# DKT atlas MNI centroids
dkt_coords = np.genfromtxt(home_dir+'/Desktop/data/DKT_MNI_coords.txt',delimiter=' ')

# plotting parameters
lbs = 15    # axis label size
pvs = 13.5  # p-value size

# %% load files

# initialise files to store parameters for each pipeline
ns = 101                                # number of subjects
subj_id = np.empty([ns],dtype=object)   # store subject IDs
runtime = np.empty([ns,npip])           # run times
dice_g = np.empty([ns,npip])            # global Dice coefficient values
dice_roi = np.empty([ns,nroi,npip])     # regional Dice coefficient values

# Mindboggle data folders
mindboggle_datasets = ['Extra-18_volumes', 'MMRR-21_volumes', 'NKI-RS-22_volumes', 'NKI-TRT-20_volumes', 'OASIS-TRT-20_volumes']

s = -1 # initialise index to store outputs (start at -1 so that index = 0 on first iteration)

# loop over Mindboggle dataset folders
for dataset in mindboggle_datasets:
    (_, dirnames, _) = next(os.walk(mindboggle_dir+'/'+dataset))

    # loop over subjects within each Mindboggle folder
    for subj_dir in dirnames:
        
        s += 1                                              # increment index
        subj_id[s] = subj_dir                               # store subject ID
        print(subj_dir+'; subj. '+str(s+1)+' of '+str(ns))  # track progress
    
        # processing time
        for t in range(npip):
            time_file = np.genfromtxt(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/'+pip_kw[t]+'_runtime_sec.txt',delimiter=' ')
            if time_file.ndim == 1:         # if a single step was run
                runtime[s,t] = time_file[1]
            else:                           # if multiple steps were run
                runtime[s,t] = np.sum(time_file[:,1])
            
        ### load atlas files
        # "manual" files (mn)   
        dkt_mn_1mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/labels.DKT31.manual.MNI152.nii.gz').dataobj).flatten()
        dkt_mn_2mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31.manual.MNI152_2mm.nii.gz').dataobj).flatten()
        dkt_mn_3mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31.manual.MNI152_3mm.nii.gz').dataobj).flatten()
        # processed files (syn, n4, etc)
        dkt_syn_1mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31_1mm.nii.gz').dataobj).flatten()
        dkt_syn_2mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31_2mm.nii.gz').dataobj).flatten()
        dkt_syn_3mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31_3mm.nii.gz').dataobj).flatten()
        dkt_n4_2mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31_2mm_n4.nii.gz').dataobj).flatten()
        dkt_bet_2mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31_2mm_n4_bet.nii.gz').dataobj).flatten()
        #dkt_mask_2mm = np.array(nb.load(mindboggle_dir+'/'+subj_dir+'/compare_pipelines/labels.DKT31_2mm_n4_mask.nii.gz').dataobj).flatten()
        dkt_bsyn_2mm = np.array(nb.load(mindboggle_dir+'/'+dataset+'/'+subj_dir+'/compare_pipelines/labels.DKT31_2mm_n4_bsyn.nii.gz').dataobj).flatten()
    
        ### Dice coefficient
        # global
        dice_g[s,0] = 1-sp.spatial.distance.dice(dkt_mn_1mm!=0,dkt_syn_1mm!=0)
        dice_g[s,1] = 1-sp.spatial.distance.dice(dkt_mn_2mm!=0,dkt_syn_2mm!=0)
        dice_g[s,2] = 1-sp.spatial.distance.dice(dkt_mn_3mm!=0,dkt_syn_3mm!=0)
        dice_g[s,3] = 1-sp.spatial.distance.dice(dkt_mn_2mm!=0,dkt_n4_2mm!=0)
        dice_g[s,4] = 1-sp.spatial.distance.dice(dkt_mn_2mm!=0,dkt_bet_2mm!=0)
        #dice_g[s,5] = 1-sp.spatial.distance.dice(dkt_mn_2mm!=0,dkt_mask_2mm!=0)
        dice_g[s,5] = 1-sp.spatial.distance.dice(dkt_mn_2mm!=0,dkt_bsyn_2mm!=0)
        # NOTE: identical results obtained from custom numpy function (dice dissimilarity) and built-in scipy function (1-similarity); for two vectors, v1 and v2:
        # dice = np.sum(v1[v2==roi_id]==roi_id)*2.0 / (np.sum(v1[v1==roi_id]==roi_id) + np.sum(v2[v2==roi_id]==roi_id))
        # dice = 1-sp.spatial.distance.dice(v1==roi_id,v2==roi_id)
    
        # regional Dice values
        for n in range(nroi):
            roi_id = dkt_roi[n]
            dice_roi[s,n,0] = 1-sp.spatial.distance.dice(dkt_mn_1mm==roi_id,dkt_syn_1mm==roi_id)
            dice_roi[s,n,1] = 1-sp.spatial.distance.dice(dkt_mn_2mm==roi_id,dkt_syn_2mm==roi_id)
            dice_roi[s,n,2] = 1-sp.spatial.distance.dice(dkt_mn_3mm==roi_id,dkt_syn_3mm==roi_id)
            dice_roi[s,n,3] = 1-sp.spatial.distance.dice(dkt_mn_2mm==roi_id,dkt_n4_2mm==roi_id)
            dice_roi[s,n,4] = 1-sp.spatial.distance.dice(dkt_mn_2mm==roi_id,dkt_bet_2mm==roi_id)
            #dice_roi[s,n,5] = 1-sp.spatial.distance.dice(dkt_mn_2mm==roi_id,dkt_mask_2mm==roi_id)
            dice_roi[s,n,5] = 1-sp.spatial.distance.dice(dkt_mn_2mm==roi_id,dkt_bsyn_2mm==roi_id)

# delete unnecessary variables
del time_file
del dkt_mn_1mm
del dkt_mn_2mm
del dkt_mn_3mm
del dkt_syn_1mm
del dkt_syn_2mm
del dkt_syn_3mm
del dkt_n4_2mm
del dkt_bet_2mm
#del dkt_mask_2mm
del dkt_bsyn_2mm

# median regional Dice coefficients
dice_roi_mr = np.median(dice_roi,0) # median regional Dice coefficient for each region
dice_roi_ms = np.median(dice_roi,1) # median regional Dice coefficient for each subject

# set axis limits, excluding runtime values for 1mm/3mm pipelines (as considerably slower/faster than rest)
runtime_xlim = [13,47]          # runtime x-axis limits (alternatively: [float('%.0g' % min(runtime[:,[1,3,4,5]].flatten())),float('%.0g' % max(runtime[:,[1,3,4,5]].flatten()))])
glob_dice_xlim = [0.25,0.75]    # global dice coeff. x-axis limits (altenartively: [float('%.1g' % min(dice_g.flatten())),float('%.1g' % max(dice_g.flatten()))])
loc_dice_clim = [0.3,0.7]       # local dice coeff. colorbar limits
loc_d_dice_clim = [-0.15,0.15]  # local Delta dice coeff. colorbar limits

# %% Stats and Plots - downsampling

# pipeline parameters for current comparison
id_comp = np.array([0,1,2]).astype(int)  # IDs of current comparison
col_comp = [pip_col[i] for i in id_comp]
nm_comp = [pip_nm[i] for i in id_comp]
ncomp = id_comp.size

# Wilcoxon signed-rank test of runtime differences
p_runtime_12 = sp.stats.wilcoxon(runtime[:,0],runtime[:,1])[1] # 1mm vs 2mm
p_runtime_23 = sp.stats.wilcoxon(runtime[:,1],runtime[:,2])[1] # 2mm vs 3mm

# run time
dx = list(np.repeat(range(ncomp),ns))
dy = list(np.transpose(runtime[:,id_comp]).flatten())
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .4, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs);
ax.set_xticks(np.arange(0,max(runtime[:,id_comp].flatten()),30).astype(int))
for ml in runtime_xlim: plt.axvline(x=ml, color='lightgrey', linestyle='--')
plt.xlabel('processing time (s)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.33,0.66])                                       # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_runtime_12),ef.pow_10_fmt(p_runtime_23)], size=pvs)  # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.2, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_runtime_ds.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_runtime_ds.svg')# , bbox_inches='tight')

# Wilcoxon signed-rank test of global dice coefficient differences
p_dice_12 = sp.stats.wilcoxon(dice_g[:,0],dice_g[:,1])[1] # 1mm vs 2mm
p_dice_23 = sp.stats.wilcoxon(dice_g[:,1],dice_g[:,2])[1] # 2mm vs 3mm

# overlap
dx = list(np.repeat(range(ncomp),ns))
dy = list(np.transpose(dice_g[:,id_comp]).flatten())
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .2, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs); ax.set_xlim(glob_dice_xlim)
plt.xlabel('overlap (Dice coeff.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.33,0.66])                                   # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_dice_12),ef.pow_10_fmt(p_dice_23)], size=pvs)    # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.2, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_glob_dice_ds.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_glob_dice_ds.svg')# , bbox_inches='tight')

## regional plots using nilearn
# 1 mm
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,0],node_size=np.power(10*abs(dice_roi_mr[:,0]),2),node_kwargs={'cmap': 'plasma_r','vmin':loc_dice_clim[0],'vmax':loc_dice_clim[1]}) # np.power(5*abs(dice_roi_mr[:,0]),3)
plt.savefig(plot_dir+'/png/mindboggle_loc_dice_ds_1mm.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_dice_ds_1mm.svg')# , bbox_inches='tight')
# 2 mm
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,1],node_size=np.power(10*abs(dice_roi_mr[:,1]),2),node_kwargs={'cmap': 'plasma_r','vmin':loc_dice_clim[0],'vmax':loc_dice_clim[1]}) # edge_cmap=...
plt.savefig(plot_dir+'/png/mindboggle_loc_dice_ds_2mm.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_dice_ds_2mm.svg')# , bbox_inches='tight')
# 3 mm
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,2],node_size=np.power(10*abs(dice_roi_mr[:,2]),2),node_kwargs={'cmap': 'plasma_r','vmin':loc_dice_clim[0],'vmax':loc_dice_clim[1]}) # edge_cmap=...
plt.savefig(plot_dir+'/png/mindboggle_loc_dice_ds_3mm.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_dice_ds_3mm.svg')# , bbox_inches='tight')

# colorbar for all local plots
f, ax = plt.subplots(figsize=(6, 0.75)); f.subplots_adjust(bottom=0.65)
cmap = mpl.cm.plasma_r
norm = mpl.colors.Normalize(vmin=loc_dice_clim[0], vmax=loc_dice_clim[1])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='horizontal')
cb1.set_label('overlap (Dice coeff.)', size=lbs)
plt.savefig(plot_dir+'/png/mindboggle_loc_dice_colorbar.png', dpi=500) #bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_dice_colorbar.svg') #,bbox_inches='tight')

## regional difference plots using nilearn
# 1 mm vs 2mm
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,0]-dice_roi_mr[:,1],node_size=400*abs(dice_roi_mr[:,0]-dice_roi_mr[:,1]),node_kwargs={'cmap': 'coolwarm','vmin':loc_d_dice_clim[0],'vmax':loc_d_dice_clim[1]}) # np.power(5*abs(dice_roi_mr[:,0]),3)
plt.savefig(plot_dir+'/png/mindboggle_loc_d_dice_1mm_vs_2mm.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_d_dice_1mm_vs_2mm.svg')# , bbox_inches='tight')
# 2 mm vs 3mm
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,1]-dice_roi_mr[:,2],node_size=400*abs(dice_roi_mr[:,1]-dice_roi_mr[:,2]),node_kwargs={'cmap': 'coolwarm','vmin':loc_d_dice_clim[0],'vmax':loc_d_dice_clim[1]}) # edge_cmap=...
plt.savefig(plot_dir+'/png/mindboggle_loc_d_dice_2mm_vs_3mm.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_d_dice_2mm_vs_3mm.svg')# , bbox_inches='tight')

# colorbar for all local difference plots
f, ax = plt.subplots(figsize=(6, 0.75)); f.subplots_adjust(bottom=0.65)
cmap = mpl.cm.coolwarm
norm = mpl.colors.Normalize(vmin=loc_d_dice_clim[0], vmax=loc_d_dice_clim[1])
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='horizontal')
cb1.set_label(r'$\Delta$ overlap (Dice coeff.)', size=lbs)
plt.savefig(plot_dir+'/png/mindboggle_loc_d_dice_colorbar.png', dpi=500) #bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_d_dice_colorbar.svg') #,bbox_inches='tight')

# %% Stats and Plots - bias field correction (N4)

# pipeline parameters for current comparison
id_comp = np.array([1,3]).astype(int)
col_comp = [pip_col[i] for i in id_comp]
nm_comp = [pip_nm[i] for i in id_comp]
ncomp = id_comp.size

# Wilcoxon signed-rank test of runtime differences
p_runtime = sp.stats.wilcoxon(runtime[:,1],runtime[:,3])[1]

# time
dx = list(np.repeat(range(ncomp),ns))  
dy = list(np.transpose(runtime[:,id_comp]).flatten()) 
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .4, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs); ax.set_xlim(runtime_xlim)
plt.xlabel('processing time (s)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.5])                     # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_runtime)], size=pvs)      # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.3, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_runtime_n4.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_runtime_n4.svg')# , bbox_inches='tight')

# dice - global
p_dice = sp.stats.wilcoxon(dice_g[:,1],dice_g[:,3])[1]

dx = list(np.repeat(range(ncomp),ns))
dy = list(np.transpose(dice_g[:,id_comp]).flatten())
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .2, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs); ax.set_xlim(glob_dice_xlim)
plt.xlabel('overlap (Dice coeff.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.5])                     # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_dice)], size=pvs)         # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.3, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_glob_dice_n4.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_glob_dice_n4.svg')# , bbox_inches='tight')

# regional plot using nilearn - N4
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,3],node_size=np.power(10*abs(dice_roi_mr[:,3]),2),node_kwargs={'cmap': 'plasma_r','vmin':loc_dice_clim[0],'vmax':loc_dice_clim[1]}) # edge_cmap=...
plt.savefig(plot_dir+'/png/mindboggle_loc_dice_n4.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_dice_n4.svg')# , bbox_inches='tight')

## regional difference plots using nilearn - 2mm vs 2mm N4
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,1]-dice_roi_mr[:,3],node_size=400*abs(dice_roi_mr[:,1]-dice_roi_mr[:,3]),node_kwargs={'cmap': 'coolwarm','vmin':loc_d_dice_clim[0],'vmax':loc_d_dice_clim[1]}) # np.power(5*abs(dice_roi_mr[:,0]),3)
plt.savefig(plot_dir+'/png/mindboggle_loc_d_dice_2mm_vs_n4.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_d_dice_2mm_vs_n4.svg')# , bbox_inches='tight')

# %% Stats and Plots - brain extraction (BET)

# pipeline parameters for current comparison
id_comp = np.array([3,4]).astype(int) # np.array([3,4,5]).astype(int)
col_comp = [pip_col[i] for i in id_comp]
nm_comp = [pip_nm[i] for i in id_comp]
ncomp = id_comp.size

# Wilcoxon signed-rank test of runtime differences
p_runtime = sp.stats.wilcoxon(runtime[:,3],runtime[:,4])[1]

# time
dx = list(np.repeat(range(ncomp),ns))  
dy = list(np.transpose(runtime[:,id_comp]).flatten()) 
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .4, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs); ax.set_xlim(runtime_xlim)
plt.xlabel('processing time (s)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.5])                     # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_runtime)], size=pvs)      # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.3, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_runtime_extr.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_runtime_extr.svg')# , bbox_inches='tight')

# dice - global
p_dice = sp.stats.wilcoxon(dice_g[:,3],dice_g[:,4])[1]

dx = list(np.repeat(range(ncomp),ns))
dy = list(np.transpose(dice_g[:,id_comp]).flatten())
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .2, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs); ax.set_xlim(glob_dice_xlim)
plt.xlabel('overlap (Dice coeff.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.5])                     # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_dice)], size=pvs)         # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.3, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_glob_dice_extr.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_glob_dice_extr.svg')# , bbox_inches='tight')

# regional plot using nilearn - bet brain extraction
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,4],node_size=np.power(10*abs(dice_roi_mr[:,4]),2),node_kwargs={'cmap': 'plasma_r','vmin':loc_dice_clim[0],'vmax':loc_dice_clim[1]}) # edge_cmap=...
plt.savefig(plot_dir+'/png/mindboggle_loc_dice_extr.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_dice_extr.svg')# , bbox_inches='tight')

## regional difference plots using nilearn - 2mm N4 vs 2mm N4 BET
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,3]-dice_roi_mr[:,4],node_size=400*abs(dice_roi_mr[:,3]-dice_roi_mr[:,4]),node_kwargs={'cmap': 'coolwarm','vmin':loc_d_dice_clim[0],'vmax':loc_d_dice_clim[1]}) # np.power(5*abs(dice_roi_mr[:,0]),3)
plt.savefig(plot_dir+'/png/mindboggle_loc_d_dice_n4_vs_extr.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_d_dice_n4_vs_extr.svg')# , bbox_inches='tight')

# %% Stats and Plots - spline registration (b-spline SyN)

# pipeline parameters for current comparison
id_comp = np.array([3,5]).astype(int) # np.array([3,6]).astype(int)
col_comp = [pip_col[i] for i in id_comp]
nm_comp = [pip_nm[i] for i in id_comp]
ncomp = id_comp.size

# Wilcoxon signed-rank test of runtime differences
p_runtime = sp.stats.wilcoxon(runtime[:,3],runtime[:,5])[1]

# time
dx = list(np.repeat(range(ncomp),ns))  
dy = list(np.transpose(runtime[:,id_comp]).flatten()) 
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .4, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs); ax.set_xlim(runtime_xlim)
plt.xlabel('processing time (s)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.5])                     # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_runtime)], size=pvs)      # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.3, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_runtime_spl.png', dpi=500) #, bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_runtime_spl.svg') #, bbox_inches='tight')

# dice - global
p_dice = sp.stats.wilcoxon(dice_g[:,3],dice_g[:,5])[1]

dx = list(np.repeat(range(ncomp),ns))
dy = list(np.transpose(dice_g[:,id_comp]).flatten())
f, ax = plt.subplots(figsize=(8, ncomp))
ax=pt.RainCloud(x = dx, y = dy, palette = col_comp, bw = .2, width_viol = .6, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(nm_comp, size=lbs); ax.set_xlim(glob_dice_xlim)
plt.xlabel('overlap (Dice coeff.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.5])                     # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_dice)], size=pvs)         # format p-values
f.subplots_adjust(left=0.26, right=0.85, bottom = 0.3, top = 0.95) #f.tight_layout()
plt.savefig(plot_dir+'/png/mindboggle_glob_dice_spl.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_glob_dice_spl.svg')# , bbox_inches='tight')

# regional plot using nilearn - spline registration
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,5],node_size=np.power(10*abs(dice_roi_mr[:,5]),2),node_kwargs={'cmap': 'plasma_r','vmin':loc_dice_clim[0],'vmax':loc_dice_clim[1]}) # edge_cmap=...
plt.savefig(plot_dir+'/png/mindboggle_loc_dice_spl.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_dice_spl.svg')# , bbox_inches='tight')

## regional difference plots using nilearn - 2mm N4 vs 2mm N4 spline SyN
nl.plotting.plot_connectome(np.zeros([nroi,nroi]),node_coords=dkt_coords,node_color=dice_roi_mr[:,3]-dice_roi_mr[:,5],node_size=400*abs(dice_roi_mr[:,3]-dice_roi_mr[:,5]),node_kwargs={'cmap': 'coolwarm','vmin':loc_d_dice_clim[0],'vmax':loc_d_dice_clim[1]}) # np.power(5*abs(dice_roi_mr[:,0]),3)
plt.savefig(plot_dir+'/png/mindboggle_loc_d_dice_n4_vs_spl.png', dpi=500)# , bbox_inches='tight')
plt.savefig(plot_dir+'/svg/mindboggle_loc_d_dice_n4_vs_spl.svg')# , bbox_inches='tight')
