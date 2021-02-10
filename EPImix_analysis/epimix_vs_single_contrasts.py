#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on Thu Oct 24 16:13:20 2019

@author: Frantisek Vasa (fdv247@gmail.com)

Primary analysis script for manuscript "Rapid processing and quantitative evaluation of multicontrast EPImix scans for adaptive multimodal imaging"

Sections:
- Main analyses, on T1 intensity and Jacobians
    - correlation across subjects           
    - identifiability 
- structural covariance
- MSNs
- test-retest reliability

"""

# %% import libraries

# general 
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt; # plt.rcParams.update({'font.size': 15})
from matplotlib import cm        # colormap editing
import seaborn as sb
from statsmodels.stats.multitest import fdrcorrection
import pingouin as pg            # ICC
import ptitprince as pt          # raincloud plots

# neuroimaging
import nibabel as nb
import nilearn as nl
import nilearn.plotting
from wbplot import pscalar
import bct as bct
#from nilearn.masking import apply_mask

# home directory
from pathlib import Path
home_dir = str(Path.home()) # home directory

# custom functions
import os

os.chdir(home_dir+'/Python/KCL')
from netneurotools import freesurfer as nnsurf, stats as nnstats

os.chdir(home_dir+'/Desktop/scripts')
import epimix_functions as ef #from epimix_functions import pscalar_mmp_hk, pscalar_mmp_lk, plot_cbar, mad, kth_diag_indices
#from netneurotools_stats import gen_spinsamples

# change plot font
plt.rcParams["font.family"] = "arial"

# %% set-up & general variables

### directories
# data to be read in
epimix_dir = home_dir+'/Desktop/EPImix'
mmp_dir = home_dir+'/Desktop/MMP'
mni_dir = home_dir+'/Desktop/MNI'
demog_dir = home_dir+'/Desktop/demog'
# locations for storing temporary data (numpy etc) and plots
data_out_dir = home_dir+'/Desktop/python_data'
plot_dir = home_dir+'/Desktop/epimix_plots'

# if data and plot directories don't exist, create them
if not os.path.isdir(data_out_dir): os.mkdir(data_out_dir)  
if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
    
# general variables
epimix_contr = ['T2*','FLAIR','T2','T1','DWI','ADC']    # contrasts
nc = len(epimix_contr)                                  # number of epimix contrasts

## variables relating to the HCP multi-modal parcellation (MMP)
# h = high-resolution (360 regions)
# l = low-resolution (44 regions)
# coordinates
coords_h = np.genfromtxt(mmp_dir+'/MMP_360_2mm_MNI_2009c_asym_coords.txt',delimiter=' ')
coords_l = np.genfromtxt(mmp_dir+'/MMP_44_2mm_MNI_2009c_asym_coords.txt',delimiter=' ')
# region names
nm_h = np.genfromtxt(mmp_dir+'/MMP_360_names.txt',dtype='str',delimiter=' ')
nm_l = np.genfromtxt(mmp_dir+'/MMP/MMP_44_names.txt',dtype='str',delimiter=' ')
# number of regions
nr_h = coords_h.shape[0]
nr_l = coords_l.shape[0]
# upper triangular indices
triu_h = np.triu_indices(nr_h,1)
triu_l = np.triu_indices(nr_l,1)
# ids to downsample ("ds") MMP from 
mmp_ds_ids = np.genfromtxt(mmp_dir+'/MMP_360_to_44.csv',dtype='int',delimiter=',')[:,1]   # 180 IDs
mmp_ds_ids = np.append(mmp_ds_ids,mmp_ds_ids)                                                           # 360
# colors
col_h = np.genfromtxt(mmp_dir+'/MMP_360_col_names.txt',dtype='str',delimiter=' ')     # read in 360 ROI colors
col_l = np.genfromtxt(mmp_dir+'/MMP_44_col_names.txt',dtype='str',delimiter=' ')      # read in 44 ROI colors

# atlas files
# high-res (360)
mmp_h_nii = nb.load(mmp_dir+'/MMP_360_2mm_MNI_2009c_asym.nii.gz')
mmp_h_vec = np.array(mmp_h_nii.dataobj).flatten()
# low-res (44)
mmp_l_nii = nb.load(mmp_dir+'/MMP_44_2mm_MNI_2009c_asym.nii.gz')
mmp_l_vec = np.array(mmp_l_nii.dataobj).flatten()
# cortical GM (using MMP atlas)
gm_vec = (mmp_l_vec!=0)*1

# MNI brain and tissue class masks
bn_vec = np.array(nb.load(mni_dir+'/mni_t1_asym_09c_mask_2mm_res_dil.nii.gz').dataobj).flatten()

# region IDs for high-res atlas (left hemisphere 1->180, righ hemisphere 201-380)
mmp_h_id = [int(i) for i in set(mmp_h_vec)][1:]

nii_shape = np.array(mmp_h_nii.dataobj).shape   # nii image shape
nvox = mmp_h_vec.size                           # total number of voxels
nvox_bn = sum(bn_vec==1)                        # number of voxels in brain mask

# number of voxels per ROI
# high-res (360)
nvox_mmp_h = np.empty([nr_h])
for r in range(nr_h): 
    nvox_mmp_h[r] = np.where(mmp_h_vec==mmp_h_id[r])[0].size
# low-res (44)
nvox_mmp_l = np.empty([nr_l])
for r in range(nr_l): 
    nvox_mmp_l[r] = np.where(mmp_l_vec==(r+1))[0].size

# Yeo networks
yeo_mmp_h = np.genfromtxt(mmp_dir+'/MMP_360_Yeo.txt',delimiter=' ',dtype=int)
nr_yeo_h = [sum(yeo_mmp_h==i) for i in np.unique(yeo_mmp_h)]
nyeo = len(np.unique(yeo_mmp_h))
yeo_col = tuple([[120/256,18/256,134/256],
                 [70/256,130/256,180/256],
                 [0/256,118/256,14/256],
                 [196/256,58/256,250/256],
                 [220/256,248/256,164/256],
                 [230/256,148/256,34/256],
                 [205/256,62/256,78/256]]) # from http://ftp.nmr.mgh.harvard.edu/fswiki/CerebellumParcellation_Buckner2011

# plotting parameters
# general
lbs = 18 # label size
lgs = 16 # legend size
axs = 15 # axis size
# heatmaps
hm_lbs = 15 # label size
hm_lgs = 13 # legend size
hm_axs = 11 # axis size

# %%

"""

Main analyses on T1
 
"""

# plot directory for t1 analyses
plot_dir_t1 = plot_dir+'/t1'
if not os.path.isdir(plot_dir_t1):
    os.mkdir(plot_dir_t1)
    
# figure saving condition
save_fig_t1 = False

# %% read in demographics

# all participants' demographics
demog = pd.read_excel(home_dir+'/Desktop/data/ADAPTA_TINKER_COGNISCAN_demog.xlsx').to_numpy()
ns = demog.shape[0]             # number of subjects
sub = demog[0:ns,0].astype(str) # ID
age = demog[0:ns,1].astype(int) # age
sex = demog[0:ns,2].astype(str) # sex

# subjects with t1 scans
sub_t1 = np.genfromtxt(home_dir+'/Desktop/data/t1_subj_list.txt',dtype='str',delimiter=' ')
ns_t1 = len(sub_t1)
sub_t1_id = []
for s in range(ns_t1):
    sub_t1_id.append(np.where(sub==sub_t1[s])[0][0])
    
# demographics for subset of participants with t1 scans
age_t1 = age[sub_t1_id]
sex_t1 = sex[sub_t1_id]

# # chi-squared test of age differences by sex and scan type
# sub_count = [[sum(sex_t1=='F'),sum(sex_t1=='M')], [sum(sex=='F')-sum(sex_t1=='F'),sum(sex=='M')-sum(sex_t1=='M')]]
# stat, p, dof, expected = sp.stats.chi2_contingency(sub_count) 

# %% demographic summary plots

#bins = np.linspace(min(age), max(age), int(ns/5))
bin_step = 2.5
bins = np.arange(17.5, 62.5, step=bin_step)

# ### all participants       
# plt.figure()
# plt.hist([age[sex=='M'],age[sex=='F']], bins, label=['M (N = '+str(sum(sex=='M'))+')','F (N = '+str(sum(sex=='F'))+')'],color=['orange','purple'], edgecolor='black')#,histtype='barstacked')
# plt.legend(loc='upper right',prop={'size': lgs})
# plt.ylim([0,15])
# plt.xlabel('age (years)',size=lbs); plt.ylabel('# participants',size=lbs)
# plt.xticks(size=axs); plt.yticks(size=axs);
# if save_fig_t1: plt.savefig(plot_dir_t1+'/age_dist_epimix.svg',bbox_inches='tight')

# ### participants with t1 scans
# plt.figure()
# ax = plt.hist([age_t1[sex_t1=='M'],age_t1[sex_t1=='F']], bins, label=['M (N = '+str(sum(sex_t1=='M'))+')','F (N = '+str(sum(sex_t1=='F'))+')'],color=['orange','purple'], edgecolor='black')
# plt.legend(loc='upper right',prop={'size': lgs})
# plt.ylim([0,15])
# plt.xlabel('age (years)',size=lbs); plt.ylabel('# participants',size=lbs)
# plt.xticks(size=axs); plt.yticks(size=axs);
# if save_fig_t1: plt.savefig(plot_dir_t1+'/age_dist_epimix_and_t1.svg',bbox_inches='tight')

### epimix and t1 scans combined
# epimix+t1 bin counts
hist_age_epiandt1_m,_ = np.histogram(age_t1[sex_t1=='M'],bins=bins)
hist_age_epiandt1_f,_ = np.histogram(age_t1[sex_t1=='F'],bins=bins)
# epimix only bin counts
hist_age_epionly_m,_ = np.histogram(age[np.setdiff1d(np.arange(0,ns),sub_t1_id)][sex[np.setdiff1d(np.arange(0,ns),sub_t1_id)]=='M'],bins=bins)
hist_age_epionly_f,_ = np.histogram(age[np.setdiff1d(np.arange(0,ns),sub_t1_id)][sex[np.setdiff1d(np.arange(0,ns),sub_t1_id)]=='F'],bins=bins)
# bar chart parameters
fig, ax = plt.subplots()
bin_w = 0.85
shift_m = (bin_step-2*bin_w)/2
shift_f = shift_m+bin_w
# bar chart - m
ax.bar(bins[:-1]+shift_m, hist_age_epiandt1_m, width=bin_w, align='edge', color='orange', edgecolor='black') 
ax.bar(bins[:-1]+shift_m, hist_age_epionly_m, width=bin_w, align='edge', bottom=hist_age_epiandt1_m, color='gold', edgecolor='black') 
# bar chart - f
ax.bar(bins[:-1]+shift_f, hist_age_epiandt1_f, width=bin_w, align='edge', color='purple', edgecolor='black') 
ax.bar(bins[:-1]+shift_f, hist_age_epionly_f, width=bin_w, align='edge', bottom=hist_age_epiandt1_f, color='mediumorchid', edgecolor='black') 
# labels etc
plt.xlabel('age (years)',size=lbs); plt.ylabel('# participants',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs);
if save_fig_t1: plt.savefig(plot_dir_t1+'/age_dist.svg',bbox_inches='tight')

# %% load epimix data

# recreate "epi" array from numpy (values not within the brain mask == 0, but array dimensions are compatible with analysis code below)
if os.path.isfile(epimix_dir+'/epi_bn.npy'):  # if file exists, load it    
    epi_bn = np.load(epimix_dir+'/epi_bn.npy')
    epi = np.empty([ns,nc,nvox])
    epi[:,:,bn_vec!=0] = epi_bn
    del epi_bn

# %% load t1 data

# recreate "t1" array from numpy (values not within the brain mask == 0, but array dimensions are compatible with analysis code below)
if os.path.isfile(epimix_dir+'/t1_bn.npy'):  # if file exists, load it    
    t1_bn = np.load(epimix_dir+'/t1_bn.npy')
    t1 = np.empty([ns_t1,nvox])
    t1[:,bn_vec!=0] = t1_bn
    del t1_bn

# individual epimix contrasts
epi_t2_id = 2   # epi_t2 = epi[:,2,:]
epi_t1_id = 3   # epi_t1 = epi[:,3,:]
epi_dwi_id = 4  # epi_dwi = epi[:,4,:]
epi_md_id = 5   # epi_md = epi[:,5,:]

# %% EPImix mask, to retain only voxels present in the FoV of most subjects

# count of non-zero voxels
epi_overlap = np.sum((epi[:,epi_t1_id,:]>0)*1,0)/ns

# voxels where "fov_thr" (proportion) of scans are covered
fov_thr = 0.8                       # proportion overlap threshold
fov = (epi_overlap>fov_thr)*1       # field of view = voxels with non-zero values in at least "fov_thr" participants
# FoV + brain
fov_bn_vec = fov*((bn_vec!=0)*1)     # as above, but MNI brain voxels only
nvox_fov_bn = sum(fov_bn_vec==1)        # number of voxels in MNI brain mask
# FoV + GM
fov_gm_vec = fov*((mmp_l_vec!=0)*1)     # as above, but GM (= ROI) voxels only
nvox_fov_gm = sum(fov_gm_vec==1)        # number of voxels in FoV/GM mask

### plot overlap - parameters
c_lim = (0,1)
cut_crd = (30, 0, 5)
# # all voxels
# nl.plotting.plot_img(nb.Nifti1Image(np.reshape(epi_overlap,nii_shape),affine=mmp_h_nii.affine),colorbar=True,cmap='plasma_r',cut_coords=cut_crd[0:2],black_bg=False, vmin=c_lim[0], vmax=c_lim[1],display_mode='yx')
# if save_fig_t1: plt.savefig(plot_dir_t1+'/epi_overlap.png',dpi=500, bbox_inches='tight')
# MNI only
ef.plot_nl_image_masked(epi_overlap, bn_vec, nii_shape, mmp_h_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd[0:2], draw_cross=True,black_bg=False,display_mode='yx')
if save_fig_t1: plt.savefig(plot_dir_t1+'/epi_overlap_bn.png',dpi=500, bbox_inches='tight')
# MNI && FoV only
ef.plot_nl_image_masked(epi_overlap, fov_bn_vec, nii_shape, mmp_h_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd[0:2], draw_cross=True,black_bg=False,display_mode='yx')
if save_fig_t1: plt.savefig(plot_dir_t1+'/epi_overlap_bn_fov.png',dpi=500, bbox_inches='tight')
# GM only
ef.plot_nl_image_masked(epi_overlap, gm_vec, nii_shape, mmp_h_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd[0:2], draw_cross=True,black_bg=False,display_mode='yx')
if save_fig_t1: plt.savefig(plot_dir_t1+'/epi_overlap_gm.png',dpi=500, bbox_inches='tight')
# GM && FoV only
ef.plot_nl_image_masked(epi_overlap, fov_gm_vec, nii_shape, mmp_h_nii.affine, cmap='plasma_r', clim=c_lim, cut_coords=cut_crd[0:2], draw_cross=True,black_bg=False,display_mode='yx')
if save_fig_t1: plt.savefig(plot_dir_t1+'/epi_overlap_gm_fov.png',dpi=500, bbox_inches='tight')

# proportion of voxels in the (above-defined) fov in each region
# high-res
fov_mmp_h = np.empty([nr_h])
for r in range(nr_h): 
    fov_mmp_h[r] = np.intersect1d(np.where(mmp_h_vec==mmp_h_id[r])[0],np.where(fov==1)[0]).size/nvox_mmp_h[r]
# low-res
fov_mmp_l = np.empty([nr_l])
for r in range(nr_l): 
    fov_mmp_l[r] = np.intersect1d(np.where(mmp_l_vec==(r+1))[0],np.where(fov==1)[0]).size/nvox_mmp_l[r]
        
# regional plots of fov for all regions
if save_fig_t1: 
    # mmp h
    c_lim = (0,1) #(min(fov_mmp_h),max(fov_mmp_h))
    pscalar(file_out=plot_dir_t1+'/fov_mmp_h.png', pscalars=fov_mmp_h, cmap='plasma_r',vrange=c_lim)
    ef.plot_cbar(c_lim=c_lim, cmap_nm='plasma_r', c_label='% participants', lbs=14, save_path=plot_dir_t1+'/fov_mmp_h_cbar.png')
    # mmp l
    c_lim = (0,1) #(0.2,max(fov_mmp_l)) # min(fov_mmp_l) = 0.208
    pscalar(file_out=plot_dir_t1+'/fov_mmp_l.png', pscalars=fov_mmp_l[mmp_ds_ids], cmap='plasma_r',vrange=c_lim) 
    ef.plot_cbar(c_lim=c_lim, cmap_nm='plasma_r', c_label='% participants', lbs=14, save_path=plot_dir_t1+'/fov_mmp_l_cbar.png')
    
# ROIs for with at least "roi_thr" voxels with at least "fov_thr" coverage
roi_thr = 0.8
mmp_hk = np.where(fov_mmp_h>roi_thr)[0] # k = "keep"
mmp_lk = np.where(fov_mmp_l>roi_thr)[0] # k = "keep"

# number of retained regions
nr_hk = mmp_hk.size
nr_lk = mmp_lk.size

# plots of retained regions
if save_fig_t1: 
    # mmp h
    c_lim = (0,1) #(roi_thr,max(fov_mmp_h[mmp_hk])) # min(fov_mmp_h[mmp_hk])
    ef.pscalar_mmp_hk(file_out=plot_dir_t1+'/fov_mmp_hk.png', pscalars_hk=fov_mmp_h[mmp_hk], mmp_hk=mmp_hk, cmap='plasma_r',vrange=c_lim)
    ef.plot_cbar(c_lim=c_lim, cmap_nm='plasma_r', c_label='% participants', lbs=14, save_path=plot_dir_t1+'/fov_mmp_hk_cbar.png')
    # mmp l
    c_lim = (0,1) #(roi_thr,max(fov_mmp_l[mmp_lk])) # min(fov_mmp_h[mmp_hk])
    ef.pscalar_mmp_lk(file_out=plot_dir_t1+'/fov_mmp_lk.png', pscalars_lk=fov_mmp_l[mmp_lk], mmp_lk=mmp_lk, mmp_ds_ids=mmp_ds_ids, cmap='plasma_r',vrange=c_lim)
    ef.plot_cbar(c_lim=c_lim, cmap_nm='plasma_r', c_label='% participants', lbs=14, save_path=plot_dir_t1+'/fov_mmp_lk_cbar.png')

# upper triangular indices for retained regions
triu_lk = np.triu_indices(nr_lk,1)
triu_hk = np.triu_indices(nr_hk,1)

# verify number of retained regions in LH (and consequently RH) by inspecting coordinates
nr_lk_lh = sum(coords_l[mmp_lk,0]<0)
nr_hk_lh = sum(coords_h[mmp_hk,0]<0)

# Yeo network ids for retained regions
yeo_mmp_hk = yeo_mmp_h[mmp_hk]                                      # ids for subset of mmp h retained regions
yeo_mmp_hk_ord = np.argsort(yeo_mmp_hk)                             # order of region (for plot sorting)
nr_yeo_hk = [sum(yeo_mmp_hk==i) for i in np.unique(yeo_mmp_hk)]     # number of regions per Yeo network

# Yeo network plots
if save_fig_t1: 
    from matplotlib.colors import ListedColormap
    # all
    pscalar(file_out=plot_dir_t1+'/yeo.png', pscalars=yeo_mmp_h, cmap=ListedColormap(yeo_col),vrange=(0.5,nyeo+0.5)) # shift color range by 0.5 to align with colors
    # only fov regions
    ef.pscalar_mmp_hk(file_out=plot_dir_t1+'/yeo_hk.png', pscalars_hk=yeo_mmp_hk, mmp_hk=mmp_hk, cmap=ListedColormap(yeo_col),vrange=(0.5,nyeo+0.5)) # shift color range by 0.5 to align with colors

# %% median values within ROIs

# remap grays in MMP / Glasser colorscheme, so as not to contrast with excluded regions
col_l_remap = np.copy(col_l)
col_l_remap[col_l=='gray'] = 'orange'
col_l_remap[col_l=='silver'] = 'gold'

# plot regions
if save_fig_t1: 
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    #### MMP h
    col_h_rgb = np.ones([int(nr_h/2),4])
    for i in range(int(nr_l/2)):
        for j in np.where(mmp_ds_ids[0:int(nr_h/2)]==(i+1))[0]:
            if i==13 or i == 20: # black colours - lighten only
                col_h_rgb[j,0:3] = ef.adjust_lightness(mcolors.to_rgb('dimgray'),np.random.uniform(0.1,0.5,1)[0])
                #col_h_rgb[j,:] = np.concatenate((np.array(mcolors.to_rgb(col_l[i])),np.random.uniform(0.5,1,1)))
            else: # other colours - lighten or darken
                col_h_rgb[j,0:3] = ef.adjust_lightness(mcolors.to_rgb(col_l_remap[i]),np.random.uniform(0.8,1.2,1)[0])
    col_h_rgb = np.vstack((col_h_rgb,col_h_rgb))
    # all
    pscalar(file_out=plot_dir_t1+'/mmp_h.png', pscalars=np.arange(0,nr_h,1), cmap=ListedColormap(col_h_rgb),vrange=(-0.5,nr_h-0.5))
    # only fov regions
    ef.pscalar_mmp_hk(file_out=plot_dir_t1+'/mmp_hk.png', pscalars_hk=np.arange(0,nr_h,1)[mmp_hk], mmp_hk=mmp_hk, cmap=ListedColormap(col_h_rgb),vrange=(-0.5,nr_h-0.5))
    #### MMP l
    col_l_rgb = np.ones([nr_l,4])
    for i in range(nr_l):
        col_l_rgb[i,0:3] = mcolors.to_rgb(col_l_remap[i])
    # all
    pscalar(file_out=plot_dir_t1+'/mmp_l.png', pscalars=np.arange(0,nr_l,1)[mmp_ds_ids], cmap=ListedColormap(col_l_rgb),vrange=(0.5,nr_l+0.5)) # shift color range by 0.5 to align with colors
    # only fov regions
    ef.pscalar_mmp_lk(file_out=plot_dir_t1+'/mmp_lk.png', pscalars_lk=np.arange(0,nr_l,1)[mmp_lk], mmp_lk=mmp_lk, mmp_ds_ids = mmp_ds_ids, cmap=ListedColormap(col_l_rgb),vrange=(0.5,nr_l+0.5)) # shift color range by 0.5 to align with colors
    ef.pscalar_mmp_hk(file_out=plot_dir_t1+'/mmp_l_hk.png', pscalars_hk=np.arange(0,nr_l,1)[mmp_ds_ids][mmp_hk], mmp_hk=mmp_hk, cmap=ListedColormap(col_l_rgb),vrange=(0.5,nr_l+0.5)) # shift color range by 0.5 to align with colors
    
# mmp h
epi_hk = np.empty([ns,nc,nr_hk])
t1_hk = np.empty([ns_t1,nr_hk])
for r in range(nr_hk):
    # replace zeros by nans
    temp_epi = epi[:,:,mmp_h_vec==mmp_h_id[mmp_hk[r]]]; temp_epi[temp_epi==0] = np.nan
    temp_t1 = t1[:,mmp_h_vec==mmp_h_id[mmp_hk[r]]]; temp_t1[temp_t1==0] = np.nan
    # calculate median, ignoring nans
    epi_hk[:,:,r] = np.nanmedian(temp_epi,axis=2)
    t1_hk[:,r] = np.nanmedian(temp_t1,axis=1)

# mmp l
epi_lk = np.empty([ns,nc,nr_lk])
t1_lk = np.empty([ns_t1,nr_lk])
for r in range(nr_lk):
    # replace zeros by nans
    temp_epi = epi[:,:,mmp_l_vec==(mmp_lk[r]+1)]; temp_epi[temp_epi==0] = np.nan
    temp_t1 = t1[:,mmp_l_vec==(mmp_lk[r]+1)]; temp_t1[temp_t1==0] = np.nan
    # calculate median, ignoring nans
    epi_lk[:,:,r] = np.nanmedian(temp_epi,axis=2)
    t1_lk[:,r] = np.nanmedian(temp_t1,axis=1)

del temp_epi
del temp_t1

# %% t1 VS epimix t1 at voxels and regions (across participants)
# part_rho = participant rho

# voxels
if os.path.isfile(data_out_dir+'/part_rho_v.npz'):  # if file exists, load it
    npz = np.load(data_out_dir+'/part_rho_v.npz')
    part_rho_v = npz['part_rho_v']
    part_p_v = npz['part_p_v']
    del npz
else:                                           # else recreate values using code below
    part_rho_v = np.empty([nvox])
    part_p_v = np.empty([nvox])
    for v in range(nvox):
        if v % 10000 == 0: print(v)
        part_rho_v[v],part_p_v[v] = sp.stats.spearmanr(t1[:,v],epi[np.array(sub_t1_id),epi_t1_id,v])
    np.savez(data_out_dir+'/part_rho_v.npz',part_rho_v=part_rho_v,part_p_v=part_p_v)

# voxel-wise plots
c_lim = (-0.8,0.8) # np.nanmax(abs(part_rho_v))
cut_crd = (30,0,5) 
# # all voxels
# nl.plotting.plot_img(nb.Nifti1Image(np.reshape(part_rho_v,nii_shape),affine=mmp_h_nii.affine),colorbar=True,cmap='coolwarm', vmin=-np.nanmax(abs(part_rho_v)), vmax=np.nanmax(abs(part_rho_v)), cut_coords=(30, 0, 0), draw_cross=False)
# if save_fig_t1: plt.savefig(plot_dir_t1+'/part_rho_v.png',dpi=500, bbox_inches='tight')
# MNI && FoV only
ef.plot_nl_image_masked(part_rho_v, fov_bn_vec, nii_shape, mmp_h_nii.affine, cmap='coolwarm', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False)
if save_fig_t1: plt.savefig(plot_dir_t1+'/part_rho_v_bn_fov.png',dpi=500, bbox_inches='tight')
# GM && FoV only
ef.plot_nl_image_masked(part_rho_v, fov_gm_vec, nii_shape, mmp_h_nii.affine, cmap='coolwarm', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False)
if save_fig_t1: plt.savefig(plot_dir_t1+'/part_rho_v_gm_fov.png',dpi=500, bbox_inches='tight')
# colorbar for voxel-wise plots
if save_fig_t1: ef.plot_cbar(c_lim=c_lim, cmap_nm='coolwarm', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_t1+'/part_rho_v_cbar.svg')

# mmp hk
part_rho_hk = np.empty([nr_hk])
part_p_hk = np.empty([nr_hk])
for v in range(nr_hk):
    part_rho_hk[v],part_p_hk[v] = sp.stats.spearmanr(t1_hk[:,v],epi_hk[np.array(sub_t1_id),epi_t1_id,v])
    #part_rho_h[v],part_p_h[v] = sp.stats.spearmanr(t1_hk[:,v],epi_hk[np.array(sub_t1_id),epi_t1_id,v])

# mmp hk plots
if save_fig_t1:
    c_lim = (-max(abs(part_rho_hk)),max(abs(part_rho_hk))) # min(fov_mmp_h[mmp_hk])
    ef.plot_cbar(c_lim=c_lim, cmap_nm='coolwarm', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_t1+'/part_rho_mmp_hk_cbar.svg')
    ef.pscalar_mmp_hk(file_out=plot_dir_t1+'/part_rho_mmp_hk.png', pscalars_hk=part_rho_hk, mmp_hk=mmp_hk, cmap='coolwarm',vrange=c_lim)

# mmp lk
part_rho_lk = np.empty([nr_lk])   # nr_lk
part_p_lk = np.empty([nr_lk])     # nr_lk
for v in range(nr_lk):           # range(nr_lk):
    part_rho_lk[v],part_p_lk[v] = sp.stats.spearmanr(t1_lk[:,v],epi_lk[np.array(sub_t1_id),epi_t1_id,v])
    #part_rho_l[v],part_p_l[v] = sp.stats.spearmanr(t1_lk[:,v],epi_lk[np.array(sub_t1_id),epi_t1_id,v])
    
# mmp lk plots
if save_fig_t1:
    c_lim = (-max(abs(part_rho_lk)),max(abs(part_rho_lk))) # min(fov_mmp_h[mmp_hk])
    ef.plot_cbar(c_lim=c_lim, cmap_nm='coolwarm', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_t1+'/part_rho_mmp_lk_cbar.svg')
    ef.pscalar_mmp_lk(file_out=plot_dir_t1+'/part_rho_mmp_lk.png', pscalars_lk=part_rho_lk, mmp_lk=mmp_lk ,mmp_ds_ids=mmp_ds_ids, cmap='coolwarm',vrange=c_lim)

# %% T1 identifiability - self- & other-similarity (correlation) between contrasts

### voxel level - brain
sub_rho_v_bn = sp.stats.spearmanr(np.transpose(np.concatenate((t1,epi[sub_t1_id,epi_t1_id,:]))[:,fov_bn_vec==1]))[0]

# all
plt.figure()
hm = sb.heatmap(sub_rho_v_bn,cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_vox_brain.svg',bbox_inches='tight')
#if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_vox_brain.png',dpi=600,bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(sub_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
#if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_vox_brain_block.svg',bbox_inches='tight') 

### voxel level - GM
sub_rho_v_gm = sp.stats.spearmanr(np.transpose(np.concatenate((t1,epi[sub_t1_id,epi_t1_id,:]))[:,fov_gm_vec==1]))[0]

# all
plt.figure()
hm = sb.heatmap(sub_rho_v_gm,cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_vox_gm.svg',bbox_inches='tight')
#if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_vox_gm.png',dpi=600,bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(sub_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_voxels_block.svg',bbox_inches='tight') 

### mmp hk
sub_rho_hk = sp.stats.spearmanr(np.transpose(np.concatenate((t1_hk,epi_hk[sub_t1_id,epi_t1_id,:]))),nan_policy='omit')[0]

# all
plt.figure()
hm = sb.heatmap(sub_rho_hk,cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_hk.svg',bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(sub_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_hk_block.svg',bbox_inches='tight')   

### mmp lk
sub_rho_lk = sp.stats.spearmanr(np.transpose(np.concatenate((t1_lk,epi_lk[sub_t1_id,epi_t1_id,:]))),nan_policy='omit')[0]

# all
plt.figure()
hm = sb.heatmap(sub_rho_lk,cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_lk.svg',bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(sub_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='plasma_r',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_lk_block.svg',bbox_inches='tight')   

# %% spin test null of correlation between maps (for mmp h and mmp l only)

# MMP annot files downloaded from https://figshare.com/articles/HCP-MMP1_0_projected_on_fsaverage/3498446
if os.path.isfile(data_out_dir+'/coords_sphere_geodesic.npz'):  # if file exists, load it
    npz = np.load(data_out_dir+'/coords_sphere_geodesic.npz')
    coords_h_sphere = npz['coords_h_sphere']
    coords_l_sphere = npz['coords_l_sphere']
    hemi_h = npz['hemi_h']
    hemi_l = npz['hemi_l']
    del npz
else:                                           # else recreate values using code below
    coords_h_sphere, hemi_h = nnsurf.find_parcel_centroids(lhannot=home_dir+'/Desktop/data/lh.HCPMMP1.annot',rhannot=home_dir+'/Desktop/data/rh.HCPMMP1.annot',version='fsaverage',surf='sphere',method='geodesic')#,drop=['unknown', 'corpuscallosum', 'Background+FreeSurfer_Defined_Medial_Wall','???'])
    coords_l_sphere, hemi_l = nnsurf.find_downsampled_parcel_centroids(lhannot=home_dir+'/Desktop/data/lh.HCPMMP1.annot',rhannot=home_dir+'/Desktop/data/rh.HCPMMP1.annot',ds_ind=mmp_ds_ids,ds_ind_hemi=np.append(np.zeros(int(nr_h/2)),np.ones(int(nr_h/2))),version='fsaverage',surf='sphere',method='geodesic')
    np.savez(data_out_dir+'/coords_sphere_geodesic.npz',coords_h_sphere=coords_h_sphere,hemi_h=hemi_h,coords_l_sphere=coords_l_sphere,hemi_l=hemi_l)

if os.path.isfile(data_out_dir+'/rho_spin.npz'):  # if file exists, load it
    npz = np.load(data_out_dir+'/rho_spin.npz')
    spins_hk = npz['spins_hk']
    spins_lk = npz['spins_lk']
    sub_rho_hk_rho = npz['sub_rho_hk_rho']
    sub_rho_lk_rho = npz['sub_rho_lk_rho']
    sub_rho_hk_pspin = npz['sub_rho_hk_pspin']
    sub_rho_lk_pspin = npz['sub_rho_lk_pspin']
    del npz
else:                                           # else recreate values using code below

    spins_hk = nnstats.gen_spinsamples(coords_h_sphere[mmp_hk,:], hemi_h[mmp_hk], method='vasa', n_rotate=10000, verbose=True, check_duplicates=True, seed=1234)
    spins_lk = nnstats.gen_spinsamples(coords_l_sphere[mmp_lk,:], hemi_l[mmp_lk], method='vasa', n_rotate=10000, verbose=True, check_duplicates=True, seed=1234)
    
    ### mmp h
    sub_rho_hk_rho = np.zeros((ns_t1,ns_t1))
    sub_rho_hk_pspin = np.zeros((ns_t1,ns_t1))
    for i in range(ns_t1):
        print(i)
        for j in range(ns_t1):
            #print(j)
            #sub_rho_hk_pspin[i,j] = ef.perm_sphere_p(t1_h[i,mmp_hk],epi_h[sub_t1_id[j],epi_t1_id,mmp_hk],spins_h,'spearman')
            sub_rho_hk_rho[i,j],sub_rho_hk_pspin[i,j] = nnstats.permtest_spearmanr(t1_hk[i,:], epi_hk[sub_t1_id[j],epi_t1_id,:], resamples=spins_hk, nan_pol='omit')
    
    ### mmp l
    sub_rho_lk_rho = np.zeros((ns_t1,ns_t1))
    sub_rho_lk_pspin = np.zeros((ns_t1,ns_t1))
    for i in range(ns_t1):
        print(i)
        for j in range(ns_t1):
            #print(j)
            #sub_rho_lk_pspin[i,j] = ef.perm_sphere_p(t1_l[i,mmp_lk],epi_l[sub_t1_id[j],epi_t1_id,mmp_lk],spins_l,'spearman')
            sub_rho_lk_rho[i,j],sub_rho_lk_pspin[i,j] = nnstats.permtest_spearmanr(t1_lk[i,:], epi_lk[sub_t1_id[j],epi_t1_id,:], resamples=spins_lk, nan_pol='omit')

    np.savez(data_out_dir+'/rho_spin.npz',spins_hk = spins_hk,spins_lk = spins_lk,sub_rho_hk_rho = sub_rho_hk_rho,sub_rho_lk_rho = sub_rho_lk_rho,sub_rho_hk_pspin = sub_rho_hk_pspin,sub_rho_lk_pspin = sub_rho_lk_pspin)
    
# FDR correction
sub_rho_hk_pspin_fdr = np.reshape(fdrcorrection(sub_rho_hk_pspin.flatten(), alpha=0.01, method='indep', is_sorted=False)[1],sub_rho_hk_pspin.shape)

# plots
c_lim = (0,1) #(np.min(sub_rho_hk_rho[sub_rho_hk_pspin_fdr<0.01]),np.max(sub_rho_hk_rho[sub_rho_hk_pspin_fdr<0.01]))
### P FDR 
sub_rho_hk_thr_block = np.ones(sub_rho_hk_rho.shape)*(c_lim[0]-1); sub_rho_hk_thr_block[sub_rho_hk_pspin_fdr<0.01] = sub_rho_hk_rho[sub_rho_hk_pspin_fdr<0.01]
cmap_under = cm.get_cmap('plasma_r', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(sub_rho_hk_thr_block,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_hk_block_pspin_thr.svg',bbox_inches='tight') 
### whole plot with permuted upper triangular
sub_rho_hk_thr = np.copy(sub_rho_hk); sub_rho_hk_thr[0:ns_t1,(ns_t1):(2*ns_t1)][sub_rho_hk_pspin_fdr>0.01] = (c_lim[0]-1)#sub_rho_hk_rho.T[sub_rho_hk_pspin_fdr.T<0.01]
cmap_under = cm.get_cmap('plasma_r', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(sub_rho_hk_thr,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_hk_triu_pspin_thr.svg',bbox_inches='tight') 
#if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_hk_triu_pspin_thr.png',dpi=600,bbox_inches='tight') 

# FDR correction
sub_rho_lk_pspin_fdr = np.reshape(fdrcorrection(sub_rho_lk_pspin.flatten(), alpha=0.01, method='indep', is_sorted=False)[1],sub_rho_lk_pspin.shape)

# plots
c_lim = (0,1) #(np.min(sub_rho_lk_rho[sub_rho_lk_pspin_fdr<0.01]),np.max(sub_rho_lk_rho[sub_rho_lk_pspin_fdr<0.01]))
### P FDR 
sub_rho_lk_thr_block = np.ones(sub_rho_lk_rho.shape)*(c_lim[0]-1); sub_rho_lk_thr_block[sub_rho_lk_pspin_fdr<0.01] = sub_rho_lk_rho[sub_rho_lk_pspin_fdr<0.01]
cmap_under = cm.get_cmap('plasma_r', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(sub_rho_lk_thr_block,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_lk_block_pspin_thr.svg',bbox_inches='tight') 
### whole plot with permuted lower triangular
sub_rho_lk_thr = np.copy(sub_rho_lk); sub_rho_lk_thr[0:ns_t1,(ns_t1):(2*ns_t1)][sub_rho_lk_pspin_fdr>0.01] = (c_lim[0]-1)#sub_rho_lk_rho.T[sub_rho_lk_pspin_fdr.T<0.01]
cmap_under = cm.get_cmap('plasma_r', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(sub_rho_lk_thr,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_lk_triu_pspin_thr.svg',bbox_inches='tight')
#if save_fig_t1: plt.savefig(plot_dir_t1+'/sub_t1_vs_epimix_t1_mmp_lk_triu_pspin_thr.png',dpi=600,bbox_inches='tight') 

### proportion of within-/between-participant correlations that survive permutation test
# within-participant correlations
sum(sub_rho_hk_pspin_fdr[np.diag_indices(ns_t1)]<0.01)
sum(sub_rho_lk_pspin_fdr[np.diag_indices(ns_t1)]<0.01)
# between-participant correlations
sum(sub_rho_hk_pspin_fdr[np.triu_indices(ns_t1,1)]<0.01)+sum(sub_rho_hk_pspin_fdr[np.tril_indices(ns_t1,-1)]<0.01)
sum(sub_rho_lk_pspin_fdr[np.triu_indices(ns_t1,1)]<0.01)+sum(sub_rho_lk_pspin_fdr[np.tril_indices(ns_t1,-1)]<0.01)

# %% "differential identifiability" Idiff

# indices (in "sub_rho_" matrices) of:
# - self-similarity ("sim_self_id" = off-diagonal)
# - other-similarity ("sim_other_id" = rest of off-diagonal block)
sim_self_id = ef.kth_diag_indices(np.zeros([2*ns_t1,2*ns_t1]),ns_t1)
sim_other_u = np.triu_indices(2*ns_t1,ns_t1+1)
temp_l = np.tril_indices(2*ns_t1,ns_t1-1)
sim_other_l = tuple([temp_l[0][np.intersect1d(np.where(temp_l[0]<ns_t1)[0],np.where(temp_l[1]>=ns_t1)[0])],temp_l[1][np.intersect1d(np.where(temp_l[0]<ns_t1)[0],np.where(temp_l[1]>=ns_t1)[0])]]) # keep elements for rows <= 65 and columns >= 66
sim_other_id = tuple([np.concatenate((sim_other_u[0],sim_other_l[0])),np.concatenate((sim_other_u[1],sim_other_l[1]))])
# delete temporary variables
del temp_l
del sim_other_u
del sim_other_l

### test plot
# test matrix
test_mat = np.zeros([2*ns_t1,2*ns_t1])
test_mat[sim_self_id] = 1
test_mat[sim_other_id] = 2
# colormap
cmap_cst = sb.xkcd_palette(['white','red','grey'])
# plot
fig = plt.figure()
fig.add_subplot(111,aspect='equal')
hm = sb.heatmap(test_mat,cmap=cmap_cst,xticklabels=False,yticklabels=False,cbar=False)#,cbar_kws={"ticks":[0,1]})
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
plt.text(ns_t1/2,2.15*ns_t1,'single-contrast scan',horizontalalignment='center',size=hm_lbs-3); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix contrast',horizontalalignment='center',size=hm_lbs-3); 
plt.text(-0.15*ns_t1,ns_t1/2,'single-contrast scan',rotation=90,verticalalignment='center',size=hm_lbs-3); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix contrast',rotation=90,verticalalignment='center',size=hm_lbs-3); 
if save_fig_t1: plt.savefig(plot_dir_t1+'/idiff_example_mat.svg',bbox_inches='tight')  
#if save_fig_t1: plt.savefig(plot_dir_t1+'/idiff_example_mat.png',dpi=600,bbox_inches='tight')  

## calculate Idiff
# median
idiff_v_bn = np.median(sub_rho_v_bn[sim_self_id])-np.median(sub_rho_v_bn[sim_other_id])
idiff_v_gm = np.median(sub_rho_v_gm[sim_self_id])-np.median(sub_rho_v_gm[sim_other_id])
idiff_hk = np.median(sub_rho_hk[sim_self_id])-np.median(sub_rho_hk[sim_other_id])
idiff_lk = np.median(sub_rho_lk[sim_self_id])-np.median(sub_rho_lk[sim_other_id])

## plot histograms of off-diagonal values with diagonal lines
bins = np.linspace(0, 1, 35)

# voxelwise - brain
f_bn = plt.figure()
plt.hist([sub_rho_v_bn[sim_self_id],sub_rho_v_bn[sim_other_id]], bins, label=['Md = '+str(round(np.median(sub_rho_v_bn[sim_self_id]),2)),'Md = '+str(round(np.median(sub_rho_v_bn[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(sub_rho_v_bn[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(sub_rho_v_bn[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper right',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(idiff_v_bn,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); #plt.gca().set_ylim(yl)
ax_bn = plt.gca(); yl_bn = ax_bn.get_ylim()

# voxelwise - GM
f_gm = plt.figure()
plt.hist([sub_rho_v_gm[sim_self_id],sub_rho_v_gm[sim_other_id]], bins, label=['Md = '+str(round(np.median(sub_rho_v_gm[sim_self_id]),2)),'Md = '+str(round(np.median(sub_rho_v_gm[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(sub_rho_v_gm[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(sub_rho_v_gm[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper right',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(idiff_v_gm,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); 
ax_gm = plt.gca(); yl_gm = ax_gm.get_ylim()

# mmp hk
f_hk = plt.figure()
plt.hist([sub_rho_hk[sim_self_id],sub_rho_hk[sim_other_id]], bins, label=['Md = '+str(round(np.median(sub_rho_hk[sim_self_id]),2)),'Md = '+str(round(np.median(sub_rho_hk[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(sub_rho_hk[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(sub_rho_hk[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper right',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(idiff_hk,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); #plt.gca().set_ylim(yl)
ax_hk = plt.gca(); yl_hk = ax_hk.get_ylim()

# mmp lk
f_lk = plt.figure()
plt.hist([sub_rho_lk[sim_self_id],sub_rho_lk[sim_other_id]], bins, label=['Md = '+str(round(np.median(sub_rho_lk[sim_self_id]),2)),'Md = '+str(round(np.median(sub_rho_lk[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(sub_rho_lk[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(sub_rho_lk[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper left',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(idiff_lk,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); #plt.gca().set_ylim(yl)
ax_lk = plt.gca(); yl_lk = ax_lk.get_ylim()

# fix common axis limits
yl = max([yl_gm,yl_bn,yl_hk,yl_lk])
ax_bn.set_ylim(yl); ax_gm.set_ylim(yl); ax_hk.set_ylim(yl); ax_lk.set_ylim(yl)
yl_idiff_gm = yl

# save plots
if save_fig_t1: f_bn.savefig(plot_dir_t1+'/idiff_v_bn.svg',bbox_inches='tight')
if save_fig_t1: f_gm.savefig(plot_dir_t1+'/idiff_v_gm.svg',bbox_inches='tight')
if save_fig_t1: f_hk.savefig(plot_dir_t1+'/idiff_hk.svg',bbox_inches='tight')
if save_fig_t1: f_lk.savefig(plot_dir_t1+'/idiff_lk.svg',bbox_inches='tight')

del f_bn
del f_gm
del f_hk
del f_lk
del ax_bn
del ax_gm
del ax_hk
del ax_lk
del yl_bn
del yl_gm
del yl_hk
del yl_lk
del yl

# %% "individual participant identifiability" - fraction of times across-subject measurement is smaller than within-subject measurement
# (related to discriminability or Discr in Bridgeford et al. 2020)

# epimix as reference - identify single t1 scan relative to epimix scans (= rows of "sub_rho" matrices)
ind_idiff_epiref_v_bn = np.zeros(ns_t1)
ind_idiff_epiref_v_gm = np.zeros(ns_t1)
ind_idiff_epiref_hk = np.zeros(ns_t1)
ind_idiff_epiref_lk = np.zeros(ns_t1)
# t1 as reference - identify single epimix scan relative to t1 scans (= columns of "sub_rho" matrices)
ind_idiff_t1ref_v_bn = np.zeros(ns_t1)
ind_idiff_t1ref_v_gm = np.zeros(ns_t1)
ind_idiff_t1ref_hk = np.zeros(ns_t1)
ind_idiff_t1ref_lk = np.zeros(ns_t1)
for i in range(ns_t1):
    # t1 as reference
    ind_idiff_epiref_v_bn[i] = sum(sub_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    ind_idiff_epiref_v_gm[i] = sum(sub_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    ind_idiff_epiref_hk[i] = sum(sub_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    ind_idiff_epiref_lk[i] = sum(sub_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    # epimix as reference
    ind_idiff_t1ref_v_bn[i] = sum(sub_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)
    ind_idiff_t1ref_v_gm[i] = sum(sub_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)
    ind_idiff_t1ref_hk[i] = sum(sub_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)
    ind_idiff_t1ref_lk[i] = sum(sub_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>sub_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)

### raincloud plots
    
## epimix as reference
# p-values
p_bg = sp.stats.wilcoxon(ind_idiff_epiref_v_bn,ind_idiff_epiref_v_gm)[1]    # brain vs gm
p_gh = sp.stats.wilcoxon(ind_idiff_epiref_v_gm,ind_idiff_epiref_hk)[1]      # gm vs hk
p_hl = sp.stats.wilcoxon(ind_idiff_epiref_hk,ind_idiff_epiref_lk)[1]        # hk vs lk
# plot
dx = list(np.repeat(range(4),ns_t1))
dy = list(np.concatenate((ind_idiff_epiref_v_bn,ind_idiff_epiref_v_gm,ind_idiff_epiref_hk,ind_idiff_epiref_lk))) 
f, ax = plt.subplots(figsize=(7,4))
ax=pt.RainCloud(x = dx, y = dy, palette = ['darkturquoise','deepskyblue','dodgerblue','steelblue'], bw = .2, width_viol = .8, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(['voxels brain','voxels GM','MMP high-res.','MMP low-res.'], size=lbs); ax.set_xlim([-0.05,1.05])
plt.xlabel(r'ind. I$_{diff}$ (EPImix T$_1$-w. ref.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.25,0.5,0.75])                                   # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_hl),ef.pow_10_fmt(p_gh),ef.pow_10_fmt(p_bg)], size=lbs)    # format p-values
xt = ax.get_xticks(); ax.set_xticklabels(np.vectorize(round)(xt,2), size=axs); 
#f.subplots_adjust(left=0.26, right=0.85, bottom = 0.2, top = 0.95) #f.tight_layout()
if save_fig_t1: plt.savefig(plot_dir_t1+'/idiff_ind_epiref.svg', bbox_inches='tight')

## t1 as reference
# p-values
p_bg = sp.stats.wilcoxon(ind_idiff_t1ref_v_bn,ind_idiff_t1ref_v_gm)[1]    # brain vs gm
p_gh = sp.stats.wilcoxon(ind_idiff_t1ref_v_gm,ind_idiff_t1ref_hk)[1]        # gm vs hk
p_hl = sp.stats.wilcoxon(ind_idiff_t1ref_hk,ind_idiff_t1ref_lk)[1]          # hk vs lk
# plot
dx = list(np.repeat(range(4),ns_t1))
dy = list(np.concatenate((ind_idiff_t1ref_v_bn,ind_idiff_t1ref_v_gm,ind_idiff_t1ref_hk,ind_idiff_t1ref_lk))) 
f, ax = plt.subplots(figsize=(7,4))
ax=pt.RainCloud(x = dx, y = dy, palette = ['darkturquoise','deepskyblue','dodgerblue','steelblue'], bw = .2, width_viol = .8, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(['voxels brain','voxels GM','MMP high-res.','MMP low-res.'], size=lbs); ax.set_xlim([-0.05,1.05])
plt.xlabel(r'ind. I$_{diff}$ (T$_1$-w. ref.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.25,0.5,0.75])                                   # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_hl),ef.pow_10_fmt(p_gh),ef.pow_10_fmt(p_bg)], size=lbs)    # format p-values
xt = ax.get_xticks(); ax.set_xticklabels(np.vectorize(round)(xt,2), size=axs); 
#f.subplots_adjust(left=0.26, right=0.85, bottom = 0.2, top = 0.95) #f.tight_layout()
if save_fig_t1: plt.savefig(plot_dir_t1+'/idiff_ind_t1ref.svg', bbox_inches='tight')

### Medians and percentiles of individual identifiability
# EPImix ref
print(str(round(np.median(ind_idiff_epiref_v_bn),2))+' ['+str(round(np.percentile(ind_idiff_epiref_v_bn,25),2))+','+str(round(np.percentile(ind_idiff_epiref_v_bn,75),2))+']')
print(str(round(np.median(ind_idiff_epiref_v_gm),2))+' ['+str(round(np.percentile(ind_idiff_epiref_v_gm,25),2))+','+str(round(np.percentile(ind_idiff_epiref_v_gm,75),2))+']')
print(str(round(np.median(ind_idiff_epiref_hk),2))+' ['+str(round(np.percentile(ind_idiff_epiref_hk,25),2))+','+str(round(np.percentile(ind_idiff_epiref_hk,75),2))+']')
print(str(round(np.median(ind_idiff_epiref_lk),2))+' ['+str(round(np.percentile(ind_idiff_epiref_lk,25),2))+','+str(round(np.percentile(ind_idiff_epiref_lk,75),2))+']')
# T1 ref
print(str(round(np.median(ind_idiff_t1ref_v_bn),2))+' ['+str(round(np.percentile(ind_idiff_t1ref_v_bn,25),2))+','+str(round(np.percentile(ind_idiff_t1ref_v_bn,75),2))+']')
print(str(round(np.median(ind_idiff_t1ref_v_gm),2))+' ['+str(round(np.percentile(ind_idiff_t1ref_v_gm,25),2))+','+str(round(np.percentile(ind_idiff_t1ref_v_gm,75),2))+']')
print(str(round(np.median(ind_idiff_t1ref_hk),2))+' ['+str(round(np.percentile(ind_idiff_t1ref_hk,25),2))+','+str(round(np.percentile(ind_idiff_t1ref_hk,75),2))+']')
print(str(round(np.median(ind_idiff_t1ref_lk),2))+' ['+str(round(np.percentile(ind_idiff_t1ref_lk,25),2))+','+str(round(np.percentile(ind_idiff_t1ref_lk,75),2))+']')

# %% 
    
"""

Jacobians
 
"""

# plot directory for main analyses
plot_dir_jcb = plot_dir+'/jcb'
if not os.path.isdir(plot_dir_jcb):
    os.mkdir(plot_dir_jcb)
    
# figure saving condition
save_fig_jcb = False

# %% load Jacobian data

# recreate "jcb_epi" array from numpy (values not within the brain mask == 0, but array dimensions are compatible with analysis code below)
if os.path.isfile(epimix_dir+'/jcb_epi_bn.npy'):  # if file exists, load it    
    jcb_epi_bn = np.load(epimix_dir+'/jcb_epi_bn.npy')
    jcb_epi = np.empty([ns,nvox])
    jcb_epi[:,bn_vec!=0] = jcb_epi_bn
    del jcb_epi_bn
    
# recreate "jcb_t1" array from numpy (values not within the brain mask == 0, but array dimensions are compatible with analysis code below)
if os.path.isfile(epimix_dir+'/jcb_t1_bn.npy'):  # if file exists, load it    
    jcb_t1_bn = np.load(epimix_dir+'/jcb_t1_bn.npy')
    jcb_t1 = np.empty([ns_t1,nvox])
    jcb_t1[:,bn_vec!=0] = jcb_t1_bn
    del jcb_t1_bn
    
# %% Median values within ROIs
    
# mmp h
jcb_epi_hk = np.empty([ns,nr_hk])
jcb_t1_hk = np.empty([ns_t1,nr_hk])
for r in range(nr_hk):
    # replace zeros by nans
    temp_jcb_epi = jcb_epi[:,mmp_h_vec==mmp_h_id[mmp_hk[r]]]; temp_jcb_epi[temp_jcb_epi==0] = np.nan
    temp_jcb_t1 = jcb_t1[:,mmp_h_vec==mmp_h_id[mmp_hk[r]]]; temp_jcb_t1[temp_jcb_t1==0] = np.nan
    # calculate median, ignoring nans
    jcb_epi_hk[:,r] = np.nanmedian(temp_jcb_epi,axis=1)
    jcb_t1_hk[:,r] = np.nanmedian(temp_jcb_t1,axis=1)

# mmp l
jcb_epi_lk = np.empty([ns,nr_lk])
jcb_t1_lk = np.empty([ns_t1,nr_lk])
for r in range(nr_lk):
    # replace zeros by nans
    temp_jcb_epi = jcb_epi[:,mmp_l_vec==(mmp_lk[r]+1)]; temp_jcb_epi[temp_jcb_epi==0] = np.nan
    temp_jcb_t1 = jcb_t1[:,mmp_l_vec==(mmp_lk[r]+1)]; temp_jcb_t1[temp_jcb_t1==0] = np.nan
    # calculate median, ignoring nans
    jcb_epi_lk[:,r] = np.nanmedian(temp_jcb_epi,axis=1)
    jcb_t1_lk[:,r] = np.nanmedian(temp_jcb_t1,axis=1)

del temp_jcb_epi
del temp_jcb_t1
    
# %% t1 VS epimix t1 at voxels and regions (across participants)
# part_rho = participant rho

# voxels
if os.path.isfile(data_out_dir+'/jcb_part_rho_v.npz'):  # if file exists, load it
    npz = np.load(data_out_dir+'/jcb_part_rho_v.npz')
    jcb_part_rho_v = npz['jcb_part_rho_v']
    jcb_part_p_v = npz['jcb_part_p_v']
    del npz
else:                                           # else recreate values using code below
    jcb_part_rho_v = np.empty([nvox])
    jcb_part_p_v = np.empty([nvox])
    for v in range(nvox):
        if v % 10000 == 0: print(v)
        jcb_part_rho_v[v],jcb_part_p_v[v] = sp.stats.spearmanr(jcb_t1[:,v],jcb_epi[np.array(sub_t1_id),v])
    np.savez(data_out_dir+'/jcb_part_rho_v.npz',jcb_part_rho_v=jcb_part_rho_v,jcb_part_p_v=jcb_part_p_v)

# voxel-wise plots
c_lim = (-1,1) # np.nanmax(abs(jcb_part_rho_v))
cut_crd = (30,0,5) 
# # all voxels
# nl.plotting.plot_img(nb.Nifti1Image(np.reshape(part_rho_v,nii_shape),affine=mmp_h_nii.affine),colorbar=True,cmap='coolwarm', vmin=-np.nanmax(abs(part_rho_v)), vmax=np.nanmax(abs(part_rho_v)), cut_coords=(30, 0, 0), draw_cross=False)
# if save_fig_t1: plt.savefig(plot_dir_t1+'/part_rho_v.png',dpi=500, bbox_inches='tight')
# MNI && FoV only
ef.plot_nl_image_masked(jcb_part_rho_v, fov_bn_vec, nii_shape, mmp_h_nii.affine, cmap='coolwarm', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False)
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_part_rho_v_bn_fov.png',dpi=500, bbox_inches='tight')
# GM && FoV only
ef.plot_nl_image_masked(jcb_part_rho_v, fov_gm_vec, nii_shape, mmp_h_nii.affine, cmap='coolwarm', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False)
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_part_rho_v_gm_fov.png',dpi=500, bbox_inches='tight')
# colorbar for voxel-wise plots
if save_fig_jcb: ef.plot_cbar(c_lim=c_lim, cmap_nm='coolwarm', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_jcb+'/jcb_part_rho_v_cbar.svg')

# mmp h
jcb_part_rho_hk = np.empty([nr_hk])
jcb_part_p_hk = np.empty([nr_hk])
for v in range(nr_hk):
    jcb_part_rho_hk[v],jcb_part_p_hk[v] = sp.stats.spearmanr(jcb_t1_hk[:,v],jcb_epi_hk[np.array(sub_t1_id),v])
    #part_rho_h[v],part_p_h[v] = sp.stats.spearmanr(t1_hk[:,v],epi_hk[np.array(sub_t1_id),epi_t1_id,v])

# mmp h plots
if save_fig_jcb:
    c_lim = (-1,1) #(-max(abs(jcb_part_rho_hk)),max(abs(jcb_part_rho_hk))) # min(fov_mmp_h[mmp_hk])
    ef.plot_cbar(c_lim=c_lim, cmap_nm='coolwarm', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_jcb+'/jcb_part_rho_mmp_hk_cbar.svg')
    ef.pscalar_mmp_hk(file_out=plot_dir_jcb+'/jcb_part_rho_mmp_hk.png', pscalars_hk=jcb_part_rho_hk, mmp_hk=mmp_hk, cmap='coolwarm',vrange=c_lim)

# mmp l
jcb_part_rho_lk = np.empty([nr_lk])   # nr_lk
jcb_part_p_lk = np.empty([nr_lk])     # nr_lk
for v in range(nr_lk):           # range(nr_lk):
    jcb_part_rho_lk[v],jcb_part_p_lk[v] = sp.stats.spearmanr(jcb_t1_lk[:,v],jcb_epi_lk[np.array(sub_t1_id),v])
    #part_rho_l[v],part_p_l[v] = sp.stats.spearmanr(t1_lk[:,v],epi_lk[np.array(sub_t1_id),epi_t1_id,v])
    
# mmp l plots
if save_fig_jcb:
    c_lim = (-1,1) #(-max(abs(jcb_part_rho_lk)),max(abs(jcb_part_rho_lk))) # min(fov_mmp_h[mmp_hk])
    ef.plot_cbar(c_lim=c_lim, cmap_nm='coolwarm', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_jcb+'/jcb_part_rho_mmp_lk_cbar.svg')
    ef.pscalar_mmp_lk(file_out=plot_dir_jcb+'/jcb_part_rho_mmp_lk.png', pscalars_lk=jcb_part_rho_lk, mmp_lk=mmp_lk,mmp_ds_ids=mmp_ds_ids, cmap='coolwarm',vrange=c_lim)

# %% Jacobian identifiability - self- & other-similarity (correlation) between contrasts

### voxel level - brain
jcb_rho_v_bn = sp.stats.spearmanr(np.transpose(np.concatenate((jcb_t1,jcb_epi[sub_t1_id,:]))[:,fov_bn_vec==1]))[0]

# all
plt.figure()
hm = sb.heatmap(jcb_rho_v_bn,cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_vox_brain.png',dpi=600,bbox_inches='tight')
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_vox_brain.svg',bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(jcb_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_vox_brain_block.svg',bbox_inches='tight') 

### voxel level - GM
jcb_rho_v_gm = sp.stats.spearmanr(np.transpose(np.concatenate((jcb_t1,jcb_epi[sub_t1_id,:]))[:,fov_gm_vec==1]))[0]

# all
plt.figure()
hm = sb.heatmap(jcb_rho_v_gm,cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_vox_gm.png',dpi=600,bbox_inches='tight')
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_vox_gm.svg',bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(jcb_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_voxels_block.svg',bbox_inches='tight') 

#####

### mmp hk
jcb_rho_hk = sp.stats.spearmanr(np.transpose(np.concatenate((jcb_t1_hk,jcb_epi_hk[sub_t1_id,:]))),nan_policy='omit')[0]

# all
plt.figure()
hm = sb.heatmap(jcb_rho_hk,cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_hk.png',dpi=500,bbox_inches='tight')
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_hk.svg',bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(jcb_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_hk_block.svg',bbox_inches='tight')   

### mmp lk
jcb_rho_lk = sp.stats.spearmanr(np.transpose(np.concatenate((jcb_t1_lk,jcb_epi_lk[sub_t1_id,:]))),nan_policy='omit')[0]

# all
plt.figure()
hm = sb.heatmap(jcb_rho_lk,cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_lk.png',dpi=500,bbox_inches='tight')
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_lk.svg',bbox_inches='tight')

# subset
plt.figure()
hm = sb.heatmap(jcb_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)],cmap='PuOr',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=-1,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_lk_block.svg',bbox_inches='tight')   

if save_fig_jcb: ef.plot_cbar(c_lim=(-1,1), cmap_nm='PuOr', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_jcb+'/jcb_t1_vs_epimix_t1_cbar.svg')

# %% spin test null of correlation between maps (for mmp h and mmp l only)

if os.path.isfile(data_out_dir+'/jcb_rho_spin.npz'):  # if file exists, load it
    # use spins generated above in t1 analysis
    npz = np.load(data_out_dir+'/rho_spin.npz')
    spins_hk = npz['spins_hk']
    spins_lk = npz['spins_lk']
    del npz
    # load permuted jcb maps
    npz = np.load(data_out_dir+'/jcb_rho_spin.npz')
    jcb_rho_hk_rho = npz['jcb_rho_hk_rho']
    jcb_rho_lk_rho = npz['jcb_rho_lk_rho']
    jcb_rho_hk_pspin = npz['jcb_rho_hk_pspin']
    jcb_rho_lk_pspin = npz['jcb_rho_lk_pspin']
    del npz
else:                                           # else recreate values using code below

    ### mmp h
    jcb_rho_hk_rho = np.zeros((ns_t1,ns_t1))
    jcb_rho_hk_pspin = np.zeros((ns_t1,ns_t1))
    for i in range(ns_t1):
        print(i)
        for j in range(ns_t1):
            #print(j)
            #sub_rho_hk_pspin[i,j] = ef.perm_sphere_p(t1_h[i,mmp_hk],epi_h[sub_t1_id[j],epi_t1_id,mmp_hk],spins_h,'spearman')
            jcb_rho_hk_rho[i,j],jcb_rho_hk_pspin[i,j] = nnstats.permtest_spearmanr(jcb_t1_hk[i,:], jcb_epi_hk[sub_t1_id[j],:], resamples=spins_hk, nan_pol='omit')
    
    ### mmp l
    jcb_rho_lk_rho = np.zeros((ns_t1,ns_t1))
    jcb_rho_lk_pspin = np.zeros((ns_t1,ns_t1))
    for i in range(ns_t1):
        print(i)
        for j in range(ns_t1):
            #print(j)
            #sub_rho_lk_pspin[i,j] = ef.perm_sphere_p(t1_l[i,mmp_lk],epi_l[sub_t1_id[j],epi_t1_id,mmp_lk],spins_l,'spearman')
            jcb_rho_lk_rho[i,j],jcb_rho_lk_pspin[i,j] = nnstats.permtest_spearmanr(jcb_t1_lk[i,:], jcb_epi_lk[sub_t1_id[j],:], resamples=spins_lk, nan_pol='omit')

    np.savez(data_out_dir+'/jcb_rho_spin.npz',jcb_rho_hk_rho = jcb_rho_hk_rho,jcb_rho_lk_rho = jcb_rho_lk_rho,jcb_rho_hk_pspin = jcb_rho_hk_pspin,jcb_rho_lk_pspin = jcb_rho_lk_pspin)
    
# FDR correction
jcb_rho_hk_pspin_fdr = np.reshape(fdrcorrection(jcb_rho_hk_pspin.flatten(), alpha=0.01, method='indep', is_sorted=False)[1],jcb_rho_hk_pspin.shape)

# plots
c_lim = (-1,1) #(np.min(jcb_rho_hk_rho[jcb_rho_hk_pspin_fdr<0.01]),np.max(jcb_rho_hk_rho[jcb_rho_hk_pspin_fdr<0.01]))
### P FDR 
jcb_rho_hk_thr_block = np.ones(jcb_rho_hk_rho.shape)*(c_lim[0]-1); jcb_rho_hk_thr_block[jcb_rho_hk_pspin_fdr<0.01] = jcb_rho_hk_rho[jcb_rho_hk_pspin_fdr<0.01]
cmap_under = cm.get_cmap('PuOr', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(jcb_rho_hk_thr_block,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_hk_block_pspin_thr.svg',bbox_inches='tight') 
### whole plot with permuted upper triangular
jcb_rho_hk_thr = np.copy(jcb_rho_hk); jcb_rho_hk_thr[0:ns_t1,(ns_t1):(2*ns_t1)][jcb_rho_hk_pspin_fdr>0.01] = (c_lim[0]-1)#jcb_rho_hk_rho.T[jcb_rho_hk_pspin_fdr.T<0.01]
cmap_under = cm.get_cmap('PuOr', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(jcb_rho_hk_thr,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_hk_triu_pspin_thr.png',dpi=600,bbox_inches='tight') 
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_hk_triu_pspin_thr.svg',bbox_inches='tight') 

# FDR correction
jcb_rho_lk_pspin_fdr = np.reshape(fdrcorrection(jcb_rho_lk_pspin.flatten(), alpha=0.01, method='indep', is_sorted=False)[1],jcb_rho_lk_pspin.shape)

# plots
c_lim = (-1,1) #(np.min(jcb_rho_lk_rho[jcb_rho_lk_pspin_fdr<0.01]),np.max(jcb_rho_lk_rho[jcb_rho_lk_pspin_fdr<0.01]))
### P FDR 
jcb_rho_lk_thr_block = np.ones(jcb_rho_lk_rho.shape)*(c_lim[0]-1); jcb_rho_lk_thr_block[jcb_rho_lk_pspin_fdr<0.01] = jcb_rho_lk_rho[jcb_rho_lk_pspin_fdr<0.01]
cmap_under = cm.get_cmap('PuOr', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(jcb_rho_lk_thr_block,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1], *hm.get_ylim())
plt.ylabel('T$_1$-w.',size=lbs); plt.xlabel('EPImix T$_1$-w.',size=lbs);
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_lk_block_pspin_thr.svg',bbox_inches='tight') 
### whole plot with permuted lower triangular
jcb_rho_lk_thr = np.copy(jcb_rho_lk); jcb_rho_lk_thr[0:ns_t1,(ns_t1):(2*ns_t1)][jcb_rho_lk_pspin_fdr>0.01] = (c_lim[0]-1)#jcb_rho_lk_rho.T[jcb_rho_lk_pspin_fdr.T<0.01]
cmap_under = cm.get_cmap('PuOr', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(jcb_rho_lk_thr,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$'},vmin=c_lim[0],vmax=c_lim[1])#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,ns_t1,2*ns_t1], *hm.get_xlim()); hm.vlines([0,ns_t1,2*ns_t1], *hm.get_ylim())
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_lk_triu_pspin_thr.png',dpi=600,bbox_inches='tight')
plt.text(ns_t1/2,2.15*ns_t1,'T$_1$-w.',horizontalalignment='center',size=hm_lbs); plt.text((3*ns_t1)/2,2.15*ns_t1,'EPImix T$_1$-w.',horizontalalignment='center',size=hm_lbs); 
plt.text(-0.15*ns_t1,ns_t1/2,'T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); plt.text(-0.15*ns_t1,(3*ns_t1)/2,'EPImix T$_1$-w.',rotation=90,verticalalignment='center',size=hm_lbs); 
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_t1_vs_epimix_t1_mmp_lk_triu_pspin_thr.svg',bbox_inches='tight') 

### proportion of within-/between-participant correlations that survive permutation test
# within-participant correlations
sum(jcb_rho_hk_pspin_fdr[np.diag_indices(ns_t1)]<0.01)
sum(jcb_rho_lk_pspin_fdr[np.diag_indices(ns_t1)]<0.01)
# between-participant correlations
sum(jcb_rho_hk_pspin_fdr[np.triu_indices(ns_t1,1)]<0.01)+sum(jcb_rho_hk_pspin_fdr[np.tril_indices(ns_t1,-1)]<0.01)
sum(jcb_rho_lk_pspin_fdr[np.triu_indices(ns_t1,1)]<0.01)+sum(jcb_rho_lk_pspin_fdr[np.tril_indices(ns_t1,-1)]<0.01)

# %% identifiability histograms / violin plots

## calculate Idiff
# median
jcb_idiff_v_bn = np.median(jcb_rho_v_bn[sim_self_id])-np.median(jcb_rho_v_bn[sim_other_id])
jcb_idiff_v_gm = np.median(jcb_rho_v_gm[sim_self_id])-np.median(jcb_rho_v_gm[sim_other_id])
jcb_idiff_hk = np.median(jcb_rho_hk[sim_self_id])-np.median(jcb_rho_hk[sim_other_id])
jcb_idiff_lk = np.median(jcb_rho_lk[sim_self_id])-np.median(jcb_rho_lk[sim_other_id])

## plot histograms of off-diagonal values with diagonal lines
bins = np.linspace(-0.6, 1, 35)

# voxelwise - brain
f_bn = plt.figure()
plt.hist([jcb_rho_v_bn[sim_self_id],jcb_rho_v_bn[sim_other_id]], bins, label=['Md = '+str(round(np.median(jcb_rho_v_bn[sim_self_id]),2)),'Md = '+str(round(np.median(jcb_rho_v_bn[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(jcb_rho_v_bn[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(jcb_rho_v_bn[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper right',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(jcb_idiff_v_bn,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); #plt.gca().set_ylim(yl)
ax_bn = plt.gca(); yl_bn = ax_bn.get_ylim(); ax_bn.set_xticks(np.arange(-0.5,1.1,0.5)); 

# voxelwise - GM
f_gm = plt.figure()
plt.hist([jcb_rho_v_gm[sim_self_id],jcb_rho_v_gm[sim_other_id]], bins, label=['Md = '+str(round(np.median(jcb_rho_v_gm[sim_self_id]),2)),'Md = '+str(round(np.median(jcb_rho_v_gm[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(jcb_rho_v_gm[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(jcb_rho_v_gm[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper right',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(jcb_idiff_v_gm,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); 
ax_gm = plt.gca(); yl_gm = ax_gm.get_ylim(); ax_gm.set_xticks(np.arange(-0.5,1.1,0.5)); 

# mmp hk
f_hk = plt.figure()
plt.hist([jcb_rho_hk[sim_self_id],jcb_rho_hk[sim_other_id]], bins, label=['Md = '+str(round(np.median(jcb_rho_hk[sim_self_id]),2)),'Md = '+str(round(np.median(jcb_rho_hk[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(jcb_rho_hk[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(jcb_rho_hk[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper right',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(jcb_idiff_hk,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); #plt.gca().set_ylim(yl)
ax_hk = plt.gca(); yl_hk = ax_hk.get_ylim(); ax_hk.set_xticks(np.arange(-0.5,1.1,0.5)); 

# mmp lk
f_lk = plt.figure()
plt.hist([jcb_rho_lk[sim_self_id],jcb_rho_lk[sim_other_id]], bins, label=['Md = '+str(round(np.median(jcb_rho_lk[sim_self_id]),2)),'Md = '+str(round(np.median(jcb_rho_lk[sim_other_id]),2))],color=['indianred','lightgrey'], edgecolor='black')
plt.axvline(np.median(jcb_rho_lk[sim_self_id]), color='darkred', linestyle='dashed', linewidth=2)
plt.axvline(np.median(jcb_rho_lk[sim_other_id]), color='dimgray', linestyle='dashed', linewidth=2)
plt.legend(loc='upper left',prop={'size': lgs})
plt.title(r'I$_{diff} = $'+str(round(jcb_idiff_lk,2)),size=lbs)
plt.xlabel(r'Spearman $\rho$',size=lbs); plt.ylabel('frequency',size=lbs)
plt.xticks(size=axs); plt.yticks(size=axs); #plt.gca().set_ylim(yl)
ax_lk = plt.gca(); yl_lk = ax_lk.get_ylim(); ax_lk.set_xticks(np.arange(-0.5,1.1,0.5)); 

# fix common y-axis limits
yl = max([yl_gm,yl_bn,yl_hk,yl_lk])
ax_bn.set_ylim(yl); ax_gm.set_ylim(yl); ax_hk.set_ylim(yl); ax_lk.set_ylim(yl)

# save plots
if save_fig_jcb: f_bn.savefig(plot_dir_jcb+'/jcb_idiff_v_bn.svg',bbox_inches='tight')
if save_fig_jcb: f_gm.savefig(plot_dir_jcb+'/jcb_idiff_v_gm.svg',bbox_inches='tight')
if save_fig_jcb: f_hk.savefig(plot_dir_jcb+'/jcb_idiff_hk.svg',bbox_inches='tight')
if save_fig_jcb: f_lk.savefig(plot_dir_jcb+'/jcb_idiff_lk.svg',bbox_inches='tight')

del f_bn
del f_gm
del f_hk
del f_lk
del ax_bn
del ax_gm
del ax_hk
del ax_lk
del yl_bn
del yl_gm
del yl_hk
del yl_lk
del yl

# %% ### "individual participant identifiability" - fraction of times across-subject measurement is smaller than within-subject measurement

# epimix as reference - identify single t1 scan relative to epimix scans (= rows of "sub_rho" matrices)
jcb_ind_idiff_epiref_v_bn = np.zeros(ns_t1)
jcb_ind_idiff_epiref_v_gm = np.zeros(ns_t1)
jcb_ind_idiff_epiref_hk = np.zeros(ns_t1)
jcb_ind_idiff_epiref_lk = np.zeros(ns_t1)
# t1 as reference - identify single epimix scan relative to t1 scans (= columns of "sub_rho" matrices)
jcb_ind_idiff_t1ref_v_bn = np.zeros(ns_t1)
jcb_ind_idiff_t1ref_v_gm = np.zeros(ns_t1)
jcb_ind_idiff_t1ref_hk = np.zeros(ns_t1)
jcb_ind_idiff_t1ref_lk = np.zeros(ns_t1)
for i in range(ns_t1):
    # t1 as reference
    jcb_ind_idiff_epiref_v_bn[i] = sum(jcb_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    jcb_ind_idiff_epiref_v_gm[i] = sum(jcb_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    jcb_ind_idiff_epiref_hk[i] = sum(jcb_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    jcb_ind_idiff_epiref_lk[i] = sum(jcb_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][i,np.setdiff1d(range(ns_t1),i)])/(ns_t1-1)
    # epimix as reference
    jcb_ind_idiff_t1ref_v_bn[i] = sum(jcb_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_v_bn[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)
    jcb_ind_idiff_t1ref_v_gm[i] = sum(jcb_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_v_gm[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)
    jcb_ind_idiff_t1ref_hk[i] = sum(jcb_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_hk[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)
    jcb_ind_idiff_t1ref_lk[i] = sum(jcb_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][i,i]>jcb_rho_lk[0:ns_t1,(ns_t1):(2*ns_t1)][np.setdiff1d(range(ns_t1),i),i])/(ns_t1-1)
    
### raincloud plots
    
## epimix as reference
# p-values
p_bg = sp.stats.wilcoxon(jcb_ind_idiff_epiref_v_bn,jcb_ind_idiff_epiref_v_gm)[1]    # brain vs gm
p_gh = sp.stats.wilcoxon(jcb_ind_idiff_epiref_v_gm,jcb_ind_idiff_epiref_hk)[1]      # gm vs hk
p_hl = sp.stats.wilcoxon(jcb_ind_idiff_epiref_hk,jcb_ind_idiff_epiref_lk)[1]        # hk vs lk
# plot
dx = list(np.repeat(range(4),ns_t1))
dy = list(np.concatenate((jcb_ind_idiff_epiref_v_bn,jcb_ind_idiff_epiref_v_gm,jcb_ind_idiff_epiref_hk,jcb_ind_idiff_epiref_lk))) 
f, ax = plt.subplots(figsize=(7,4))
ax=pt.RainCloud(x = dx, y = dy, palette = ['darkturquoise','deepskyblue','dodgerblue','steelblue'], bw = .2, width_viol = .8, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(['voxels brain','voxels GM','MMP high-res.','MMP low-res.'], size=lbs); ax.set_xlim([-0.05,1.05])
plt.xlabel(r'ind. I$_{diff}$ (EPImix T$_1$-w. ref.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.25,0.5,0.75])                                   # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_hl),ef.pow_10_fmt(p_gh),ef.pow_10_fmt(p_bg)], size=lbs)    # format p-values
xt = ax.get_xticks(); ax.set_xticklabels(np.vectorize(round)(xt,2), size=axs); 
#f.subplots_adjust(left=0.26, right=0.85, bottom = 0.2, top = 0.95) #f.tight_layout()
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_idiff_ind_epiref.svg', bbox_inches='tight')

## t1 as reference
# p-values
p_bg = sp.stats.wilcoxon(jcb_ind_idiff_t1ref_v_bn,jcb_ind_idiff_t1ref_v_gm)[1]    # brain vs gm
p_gh = sp.stats.wilcoxon(jcb_ind_idiff_t1ref_v_gm,jcb_ind_idiff_t1ref_hk)[1]        # gm vs hk
p_hl = sp.stats.wilcoxon(jcb_ind_idiff_t1ref_hk,jcb_ind_idiff_t1ref_lk)[1]          # hk vs lk
# plot
dx = list(np.repeat(range(4),ns_t1))
dy = list(np.concatenate((jcb_ind_idiff_t1ref_v_bn,jcb_ind_idiff_t1ref_v_gm,jcb_ind_idiff_t1ref_hk,jcb_ind_idiff_t1ref_lk))) 
f, ax = plt.subplots(figsize=(7,4))
ax=pt.RainCloud(x = dx, y = dy, palette = ['darkturquoise','deepskyblue','dodgerblue','steelblue'], bw = .2, width_viol = .8, ax = ax, orient = "h", box_showfliers=False)
ax.set_yticklabels(['voxels brain','voxels GM','MMP high-res.','MMP low-res.'], size=lbs); ax.set_xlim([-0.05,1.05])
plt.xlabel(r'ind. I$_{diff}$ (T$_1$-w. ref.)', size=lbs)
ax2 = ax.twinx(); ax2.set_yticks([0.25,0.5,0.75])                                   # add second y-axis for p-values
ax2.set_yticklabels([ef.pow_10_fmt(p_hl),ef.pow_10_fmt(p_gh),ef.pow_10_fmt(p_bg)], size=lbs)    # format p-values
xt = ax.get_xticks(); ax.set_xticklabels(np.vectorize(round)(xt,2), size=axs); 
#f.subplots_adjust(left=0.26, right=0.85, bottom = 0.2, top = 0.95) #f.tight_layout()
if save_fig_jcb: plt.savefig(plot_dir_jcb+'/jcb_idiff_ind_t1ref.svg', bbox_inches='tight')

### Medians and percentiles of individual identifiability
# EPImix ref
print(str(round(np.median(jcb_ind_idiff_epiref_v_bn),2))+' ['+str(round(np.percentile(jcb_ind_idiff_epiref_v_bn,25),2))+','+str(round(np.percentile(jcb_ind_idiff_epiref_v_bn,75),2))+']')
print(str(round(np.median(jcb_ind_idiff_epiref_v_gm),2))+' ['+str(round(np.percentile(jcb_ind_idiff_epiref_v_gm,25),2))+','+str(round(np.percentile(jcb_ind_idiff_epiref_v_gm,75),2))+']')
print(str(round(np.median(jcb_ind_idiff_epiref_hk),2))+' ['+str(round(np.percentile(jcb_ind_idiff_epiref_hk,25),2))+','+str(round(np.percentile(jcb_ind_idiff_epiref_hk,75),2))+']')
print(str(round(np.median(jcb_ind_idiff_epiref_lk),2))+' ['+str(round(np.percentile(jcb_ind_idiff_epiref_lk,25),2))+','+str(round(np.percentile(jcb_ind_idiff_epiref_lk,75),2))+']')
# T1 ref
print(str(round(np.median(jcb_ind_idiff_t1ref_v_bn),2))+' ['+str(round(np.percentile(jcb_ind_idiff_t1ref_v_bn,25),2))+','+str(round(np.percentile(jcb_ind_idiff_t1ref_v_bn,75),2))+']')
print(str(round(np.median(jcb_ind_idiff_t1ref_v_gm),2))+' ['+str(round(np.percentile(jcb_ind_idiff_t1ref_v_gm,25),2))+','+str(round(np.percentile(jcb_ind_idiff_t1ref_v_gm,75),2))+']')
print(str(round(np.median(jcb_ind_idiff_t1ref_hk),2))+' ['+str(round(np.percentile(jcb_ind_idiff_t1ref_hk,25),2))+','+str(round(np.percentile(jcb_ind_idiff_t1ref_hk,75),2))+']')
print(str(round(np.median(jcb_ind_idiff_t1ref_lk),2))+' ['+str(round(np.percentile(jcb_ind_idiff_t1ref_lk,25),2))+','+str(round(np.percentile(jcb_ind_idiff_t1ref_lk,75),2))+']')

# %%

"""

Structural covariance and Morphometric Similarity Networks
 
"""

# plot directory for main analyses
plot_dir_net = plot_dir+'/net'
if not os.path.isdir(plot_dir_net):
    os.mkdir(plot_dir_net)
    
# figure saving condition
save_fig_net = True

# %% normalise regional median values using MAD (retained regions only)

# epi
epi_hk_n = np.empty([ns,nc,nr_hk])
epi_lk_n = np.empty([ns,nc,nr_lk])
for s in range(ns):
    for c in range(nc):
        epi_hk_n[s,c,:] = np.divide(epi_hk[s,c,:] - np.nanmedian(epi_hk[s,c,:]),ef.mad(epi_hk[s,c,:]))
        epi_lk_n[s,c,:] = np.divide(epi_lk[s,c,:] - np.nanmedian(epi_lk[s,c,:]),ef.mad(epi_lk[s,c,:]))

# t1
t1_hk_n = np.empty([ns_t1,nr_hk])
t1_lk_n = np.empty([ns_t1,nr_lk])
for s in range(ns_t1):
    t1_hk_n[s,:] = np.divide(t1_hk[s,:] - np.nanmedian(t1_hk[s,:]),ef.mad(t1_hk[s,:]))
    t1_lk_n[s,:] = np.divide(t1_lk[s,:] - np.nanmedian(t1_lk[s,:]),ef.mad(t1_lk[s,:]))
    
# %% normalise Jacobian values
    
# jcb_epi
jcb_epi_hk_n = np.empty([ns,nr_hk])
jcb_epi_lk_n = np.empty([ns,nr_lk])
for s in range(ns):
    jcb_epi_hk_n[s,:] = np.divide(jcb_epi_hk[s,:] - np.nanmedian(jcb_epi_hk[s,:]),ef.mad(jcb_epi_hk[s,:]))
    jcb_epi_lk_n[s,:] = np.divide(jcb_epi_lk[s,:] - np.nanmedian(jcb_epi_lk[s,:]),ef.mad(jcb_epi_lk[s,:]))

# jcb_t1
jcb_t1_hk_n = np.empty([ns_t1,nr_hk])
jcb_t1_lk_n = np.empty([ns_t1,nr_lk])
for s in range(ns_t1):
    jcb_t1_hk_n[s,:] = np.divide(jcb_t1_hk[s,:] - np.nanmedian(jcb_t1_hk[s,:]),ef.mad(jcb_t1_hk[s,:]))
    jcb_t1_lk_n[s,:] = np.divide(jcb_t1_lk[s,:] - np.nanmedian(jcb_t1_lk[s,:]),ef.mad(jcb_t1_lk[s,:])) 

# %% Morphometric Similarity Networks (MSNs)

msn_hk = np.zeros([nr_hk,nr_hk,ns])
msn_lk = np.zeros([nr_lk,nr_lk,ns])

# # using epimix contrasts only
# for s in range(ns):
#     print(sub[s])
#     msn_hk[:,:,s],_ = sp.stats.spearmanr(epi_hk_n[s,:,:],nan_policy='propagate')
#     msn_lk[:,:,s],_ = sp.stats.spearmanr(epi_lk_n[s,:,:],nan_policy='propagate')
    
# using epimix contrasts + Jacobian
for s in range(ns):
    print(sub[s])
    msn_hk[:,:,s],_ = sp.stats.spearmanr(np.vstack((epi_hk_n[s,:,:],jcb_epi_hk_n[s,:])),nan_policy='propagate')
    msn_lk[:,:,s],_ = sp.stats.spearmanr(np.vstack((epi_lk_n[s,:,:],jcb_epi_lk_n[s,:])),nan_policy='propagate')

# average MSNs (for plotting)
msn_hk_m = np.nanmean(msn_hk,axis=2)
msn_lk_m = np.nanmean(msn_lk,axis=2)

### plot average MSNs

# mmp hk
plt.figure()
hm = sb.heatmap(msn_hk_m,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_hk_lh,nr_hk], *hm.get_xlim()); hm.vlines([0,nr_hk_lh,nr_hk], *hm.get_ylim())
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_hk_m.png',dpi=600,bbox_inches='tight')

# mmp hk sorted by Yeo networks
plt.figure()
hm = sb.heatmap(msn_hk_m[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord],cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
bb = np.array(hm.get_position())
ef.add_subnetwork_lines(hm,nr_yeo_hk,lw=0.5)                                            # add thin black lines
ef.add_subnetwork_colours(hm,bb,nr_yeo_hk,yeo_col,lw=5,alpha=1,solid_capstyle="butt")   # add  network colour lines
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_hk_m_yeo.png',dpi=600,bbox_inches='tight')

# mmp lk
plt.figure()
hm = sb.heatmap(msn_lk_m,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_lk_lh,nr_lk], *hm.get_xlim()); hm.vlines([0,nr_lk_lh,nr_lk], *hm.get_ylim())
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_lk_m.png',dpi=600,bbox_inches='tight')

### plot "connectome" visualisation - top positive and negative edges
# mmp hk - 0.003 = 0.3% edges
msn_hk_m_thr_pn = bct.threshold_proportional(msn_hk_m,0.0015) - bct.threshold_proportional(-msn_hk_m,0.0015)
nl.plotting.plot_connectome(msn_hk_m_thr_pn,node_coords=coords_h[mmp_hk,:],node_color=list(col_h[mmp_hk]),node_size=20,edge_cmap='coolwarm',edge_vmin=-.8,edge_vmax=.8) #node_size=10*np.sum(msn_m_thr,0)
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_hk_m_net_posneg.png',dpi=500,bbox_inches='tight')
# mmp lk - 0.1 = 10% edges
msn_lk_m_thr_pn = bct.threshold_proportional(msn_lk_m,0.05) - bct.threshold_proportional(-msn_lk_m,0.05)
nl.plotting.plot_connectome(msn_lk_m_thr_pn,node_coords=coords_l[mmp_lk,:],node_color=list(col_l[mmp_lk]),node_size=20,edge_cmap='coolwarm',edge_vmin=-.8,edge_vmax=.8) #node_size=10*np.sum(msn_m_thr,0)
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_lk_m_net_posneg.png',dpi=500,bbox_inches='tight')

### plot maximum and minimum correlations with labels for each datapoint (mmp l, example subject s = 0)

# find indices for location of (one) maximum
ind_max = np.where(np.triu(msn_lk[:,:,0],k=1)==np.amax(np.triu(msn_lk[:,:,0],k=1)))
# plot maximum correlation
#df = pd.DataFrame({'x': epi_lk_n[0,:,ind_max[0][0]],'y': epi_lk_n[0,:,ind_max[1][0]],'group': epimix_contr})
df = pd.DataFrame({'x': np.vstack((epi_lk_n[0,:,:],jcb_epi_lk_n[0,:]))[:,ind_max[0][0]],'y': np.vstack((epi_lk_n[0,:,:],jcb_epi_lk_n[0,:]))[:,ind_max[1][0]],'group': epimix_contr+['JCB']})
plt.figure()
p1 = sb.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':50}) # basic plot
for line in range(0,df.shape[0]): # add annotations one by one with a loop
     p1.text(df.x[line]+0.1, df.y[line]-0.1, df.group[line], horizontalalignment='left', size='large', color='black')#, weight='semibold')
x = plt.gca().get_xlim(); y = plt.gca().get_ylim() # symmetrise axis limits
plt.gca().set_xlim([min(x[0],y[0]),max(x[1],y[1])]); plt.gca().set_ylim([min(x[0],y[0]),max(x[1],y[1])])
x = np.linspace(*plt.gca().get_xlim()); plt.gca().plot(x,x,'--',c='grey') # add x=y line
plt.xlabel(nm_l[mmp_lk][ind_max[0][0]],size=lbs)
plt.ylabel(nm_l[mmp_lk][ind_max[1][0]],size=lbs)
plt.title('Max. corr.: Spearman ρ = '+str(round(np.amax(np.triu(msn_lk[:,:,0],k=1)),2)),size=lbs)
plt.xticks(fontsize=axs); plt.yticks(fontsize=axs)
plt.gca().set_aspect('equal')
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_lk_maxcorr.svg',bbox_inches='tight')

# find indices for location of (one) minimum
temp_msn = np.copy(abs(msn_lk[:,:,0])); temp_msn[np.tril_indices(nr_lk,k=-1)] = np.nan
ind_min = np.where(temp_msn==np.nanmin(temp_msn))
# plot minimum correlation
#df = pd.DataFrame({'x': epi_lk_n[0,:,ind_min[0][0]],'y': epi_lk_n[0,:,ind_min[1][0]],'group': epimix_contr})
df = pd.DataFrame({'x': np.vstack((epi_lk_n[0,:,:],jcb_epi_lk_n[0,:]))[:,ind_min[0][2]],'y': np.vstack((epi_lk_n[0,:,:],jcb_epi_lk_n[0,:]))[:,ind_min[1][2]],'group': epimix_contr+['JCB']})
plt.figure()
p1 = sb.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':50}) # basic plot
for line in range(0,df.shape[0]): # add annotations one by one with a loop
     p1.text(df.x[line]+0.1, df.y[line]-0.1, df.group[line], horizontalalignment='left', size='large', color='black')#, weight='semibold')
x = plt.gca().get_xlim(); y = plt.gca().get_ylim() # symmetrise axis limits
plt.gca().set_xlim([min(x[0],y[0]),max(x[1],y[1])]); plt.gca().set_ylim([min(x[0],y[0]),max(x[1],y[1])])
x = np.linspace(*plt.gca().get_xlim()); plt.gca().plot(x,x,'--',c='grey') # add x=y line
plt.xlabel(nm_l[mmp_lk][ind_min[0][2]],size=lbs)
plt.ylabel(nm_l[mmp_lk][ind_min[1][2]],size=lbs)
plt.title('Min. corr.: Spearman ρ = '+str(round(msn_lk[ind_min[0][0],ind_min[1][0],0],2)),size=lbs)
plt.xticks(fontsize=axs); plt.yticks(fontsize=axs)
plt.gca().set_aspect('equal')
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_lk_mincorr.svg',bbox_inches='tight')

# %% MSN regression / ML 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import explained_variance_score #, median_absolute_error#, r2_score
#from sklearn.model_selection import permutation_test_score

# stratified k-fold cross-validation on EPImix data
nfold = 5

# convert sex to numeric
male = np.zeros([ns])
male[sex=='M'] = 1

# bin age (for stratification)
age_bin = np.zeros([ns])
for i in range(nfold):
    age_bin[np.argsort(age)[int(i*(ns/nfold)):int((i+1)*(ns/nfold))]] = i+1

# set-up pipeline
pipe = make_pipeline(RobustScaler(),linear_model.LinearRegression())
skf = StratifiedKFold(n_splits=nfold)

# split only
skf = StratifiedKFold(n_splits=nfold)
train_ind = np.zeros([int(ns*((nfold-1)/nfold)),nfold],dtype=int)
test_ind = np.zeros([int(ns*(1/nfold)),nfold],dtype=int)
i = 0
for train_i, test_i in skf.split(np.zeros(ns),age_bin): # using fake X as "Stratification is done based on the y labels." (for split(X, y))
    #print("train:", train_i, "test:", test_i)
    # X_train, X_test = X[train_i], X[test_i]
    # y_train, y_test = y[train_i], y[test_i]
    train_ind[:,i] = train_i
    test_ind[:,i] = test_i
    i += 1

### MMP lk
msn_lk_linreg_evs = np.zeros([nr_lk,nr_lk,nfold])
#msn_linreg_mae = np.zeros([nr_lk,nr_lk,nfold])
for f in range(nfold): #f = 0 # loop over folds
    print('fold '+str(f+1)+' out of '+str(nfold))
    for i in np.arange(0,nr_lk-1):
        for j in np.arange(i+1,nr_lk):   
            
            # MSN edges as a function of age and sex
            pipe.fit(np.vstack((age,male)).T[train_ind[:,f]],msn_lk[i,j,train_ind[:,f]])
            pred_y = pipe.predict(np.vstack((age,male)).T[test_ind[:,f]])
            
            # metrics
            msn_lk_linreg_evs[i,j,f] = explained_variance_score(msn_lk[i,j,test_ind[:,f]],pred_y)
            #msn_linreg_mae[i,j,f] = median_absolute_error(age[test_ind[:,f]],pred_y)

    # fill lower triangulars with transpose
    msn_lk_linreg_evs[:,:,f] = msn_lk_linreg_evs[:,:,f]+msn_lk_linreg_evs[:,:,f].T
    #msn_linreg_mae[:,:,f] = msn_linreg_mae[:,:,f]+msn_linreg_mae[:,:,f].T

msn_lk_linreg_evs_m = np.median(msn_lk_linreg_evs,axis=2)

## plot (explained variance score > 0.001)
cmap_under = cm.get_cmap('plasma_r', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(msn_lk_linreg_evs_m,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Md expl. var. score'},vmin=0.001)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,nr_lk_lh,nr_lk], *hm.get_xlim()); hm.vlines([0,nr_lk_lh,nr_lk], *hm.get_ylim())
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_lk_linreg_evs_m.png',dpi=600,bbox_inches='tight')
# network plot
nl.plotting.plot_connectome(bct.threshold_proportional(msn_lk_linreg_evs_m,0.1),node_coords=coords_l[mmp_lk,:],node_color=list(col_l[mmp_lk]),node_size=20,edge_cmap='plasma_r',edge_vmin=0.001,edge_vmax=max(msn_lk_linreg_evs_m.flatten())) #node_size=10*np.sum(msn_lk_m_thr,0)
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_lk_linreg_evs_m_net.png',dpi=600,bbox_inches='tight')

### MMP hk
msn_hk_linreg_evs = np.zeros([nr_hk,nr_hk,nfold])
#msn_linreg_mae = np.zeros([nr_lk,nr_lk,nfold])
for f in range(nfold): #f = 0 # loop over folds
    print('fold '+str(f+1)+' out of '+str(nfold))
    for i in np.arange(0,nr_hk-1):
        if i % 30 == 0: print(i)
        for j in np.arange(i+1,nr_hk):   
            
            # exclude nan indices
            notna_train = np.where(np.isnan(msn_hk[i,j,train_ind[:,f]])==False)[0]
            notna_test = np.where(np.isnan(msn_hk[i,j,test_ind[:,f]])==False)[0]
            
            # MSN edges as a function of age and sex
            pipe.fit(np.vstack((age,male)).T[train_ind[:,f]][notna_train,:],msn_hk[i,j,train_ind[:,f]][notna_train])
            pred_y = pipe.predict(np.vstack((age,male)).T[test_ind[:,f]][notna_test,:])
            
            # metrics
            msn_hk_linreg_evs[i,j,f] = explained_variance_score(msn_hk[i,j,test_ind[:,f]][notna_test],pred_y)
            #msn_linreg_mae[i,j,f] = median_absolute_error(age[test_ind[:,f]],pred_y)

    # fill lower triangulars with transpose
    msn_hk_linreg_evs[:,:,f] = msn_hk_linreg_evs[:,:,f]+msn_hk_linreg_evs[:,:,f].T
    #msn_linreg_mae[:,:,f] = msn_linreg_mae[:,:,f]+msn_linreg_mae[:,:,f].T

msn_hk_linreg_evs_m = np.median(msn_hk_linreg_evs,axis=2)

## plot (explained variance score > 0.001)
cmap_under = cm.get_cmap('plasma_r', 256); cmap_under.set_under('white')
plt.figure()
hm = sb.heatmap(msn_hk_linreg_evs_m,cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': 'Md expl. var. score'},vmin=0.001)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
hm.hlines([0,nr_hk_lh,nr_hk], *hm.get_xlim()); hm.vlines([0,nr_hk_lh,nr_hk], *hm.get_ylim())
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_hk_linreg_evs_m.png',dpi=600,bbox_inches='tight')
# network plot - 0.1% edges
nl.plotting.plot_connectome(bct.threshold_proportional(msn_hk_linreg_evs_m,0.001),node_coords=coords_h[mmp_hk,:],node_color=list(col_h[mmp_hk]),node_size=20,edge_cmap='plasma_r',edge_vmin=0.001,edge_vmax=max(msn_hk_linreg_evs_m.flatten())) #node_size=10*np.sum(msn_lk_m_thr,0)
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_hk_linreg_evs_m_yeo.png',dpi=600,bbox_inches='tight')

# mmp hk sorted by Yeo networks
plt.figure()
hm = sb.heatmap(msn_hk_linreg_evs_m[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord],cmap=cmap_under,xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=0.001)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
bb = np.array(hm.get_position())
ef.add_subnetwork_lines(hm,nr_yeo_hk,lw=0.5)                                            # add thin black lines
ef.add_subnetwork_colours(hm,bb,nr_yeo_hk,yeo_col,lw=5,alpha=1,solid_capstyle="butt")   # add  network colour lines
if save_fig_net: plt.savefig(plot_dir_net+'/msn_mmp_hk_linreg_evs_m_yeo.png',dpi=600,bbox_inches='tight')

# %% structural covariance - T1-w.

# high-res
scn_t1_hk = sp.stats.spearmanr(t1_hk_n,nan_policy='omit')[0]
scn_epi_t1_hk = sp.stats.spearmanr(epi_hk_n[:,epi_t1_id,:],nan_policy='omit')[0]
# low-res
scn_t1_lk = sp.stats.spearmanr(t1_lk_n,nan_policy='omit')[0]
scn_epi_t1_lk = sp.stats.spearmanr(epi_lk_n[:,epi_t1_id,:],nan_policy='omit')[0]

### diamond plot
scn_diam_hk = np.copy(scn_t1_hk)
scn_diam_hk[triu_hk] = scn_epi_t1_hk[triu_hk]
plt.figure();
hm = sb.heatmap(scn_diam_hk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'both'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_hk_lh,nr_hk], *hm.get_xlim()); hm.vlines([0,nr_hk_lh,nr_hk], *hm.get_ylim())
plt.title('EPImix T$_1$-w.',size=hm_lbs); plt.ylabel('T$_1$-w.',size=hm_lbs); 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_diam_mmp_hk.png',dpi=600,bbox_inches='tight')

### diamond plot sorted by Yeo networks
# create (sorted) diamond
scn_diam_hk = np.copy(scn_t1_hk[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord])
scn_diam_hk[triu_hk] = scn_epi_t1_hk[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord][triu_hk]
# plot
plt.figure()
hm = sb.heatmap(scn_diam_hk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
plt.title('EPImix T$_1$-w.',size=hm_lbs, pad=20); plt.ylabel('T$_1$-w.',size=hm_lbs, labelpad=20); 
bb = np.array(hm.get_position())
ef.add_subnetwork_lines(hm,nr_yeo_hk,lw=0.5)                                            # add thin black lines
ef.add_subnetwork_colours(hm,bb,nr_yeo_hk,yeo_col,lw=5,alpha=1,solid_capstyle="butt")   # add  network colour lines
if save_fig_net: plt.savefig(plot_dir_net+'/scn_diam_mmp_hk_yeo.png',dpi=600,bbox_inches='tight')

### diamond plot
scn_diam_lk = np.copy(scn_t1_lk)
scn_diam_lk[triu_lk] = scn_epi_t1_lk[triu_lk]
plt.figure()
hm = sb.heatmap(scn_diam_lk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_lk_lh,nr_lk], *hm.get_xlim(),lw=1); hm.vlines([0,nr_lk_lh,nr_lk], *hm.get_ylim(),lw=1)
plt.title('EPImix T$_1$-w.',size=hm_lbs, pad=20); plt.ylabel('T$_1$-w.',size=hm_lbs, labelpad=20); 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_diam_mmp_lk.png',dpi=600,bbox_inches='tight')
#if save_fig_net: plt.savefig(plot_dir_net+'/scn_diam_mmp_lk.svg',bbox_inches='tight')

### plot thresholded networks

# low-res
scn_t1_lk_thr = bct.threshold_proportional(scn_t1_lk,0.10)          # threshold network (to 0.X % density)
scn_epi_t1_lk_thr = bct.threshold_proportional(scn_epi_t1_lk,0.10)  # threshold network (to 0.X % density)

nl.plotting.plot_connectome(scn_t1_lk_thr,node_coords=coords_l[mmp_lk,:],node_color=[col_l[i] for i in mmp_lk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.8,edge_vmax=.8) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_t1_lk_net.png',dpi=500)

nl.plotting.plot_connectome(scn_epi_t1_lk_thr,node_coords=coords_l[mmp_lk,:],node_color=[col_l[i] for i in mmp_lk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.8,edge_vmax=.8) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_epi_lk_net.png',dpi=500)

# high-res
scn_t1_hk_thr = bct.threshold_proportional(scn_t1_hk,0.005)         # threshold network (to 0.X % density)
scn_epi_t1_hk_thr = bct.threshold_proportional(scn_epi_t1_hk,0.005) # threshold network (to 0.X % density)

nl.plotting.plot_connectome(scn_t1_hk_thr,node_coords=coords_h[mmp_hk,:],node_color=[col_h[i] for i in mmp_hk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.8,edge_vmax=.8) 
#nl.plotting.plot_connectome(scn_t1_hk_thr,node_coords=coords_h[mmp_hk,:],node_color=[yeo_col[i-1] for i in yeo_mmp_hk],node_size=20,edge_cmap='coolwarm',edge_vmin=-1,edge_vmax=1) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_t1_h_net.png',dpi=500)

nl.plotting.plot_connectome(scn_epi_t1_hk_thr,node_coords=coords_h[mmp_hk,:],node_color=[col_h[i] for i in mmp_hk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.8,edge_vmax=.8) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_epi_h_net.png',dpi=500)

## correlate structural covariance matrices

### mmp h with forced square aspect ratio
# current min/max
hb_min = min(np.concatenate((scn_t1_hk[triu_hk],scn_epi_t1_hk[triu_hk])))-0.1
hb_max = max(np.concatenate((scn_t1_hk[triu_hk],scn_epi_t1_hk[triu_hk])))+0.1
# plot with added min/max data
plt.figure()
image = plt.hexbin(np.hstack((scn_epi_t1_hk[triu_hk],hb_min,hb_max)),np.hstack((scn_t1_hk[triu_hk],hb_min,hb_max)),cmap=plt.cm.Spectral_r,mincnt=1,gridsize=50) # ,extent=extent,bins='log' #plt.grid(True)
cb = plt.colorbar(image,spacing='uniform',extend='max') #cb.set_label('count', rotation=270)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
plt.xlabel('EPImix T$_1$-w.',size=hm_lbs); plt.xticks(fontsize=hm_axs)
plt.ylabel('T$_1$-w.',size=hm_lbs); plt.yticks(fontsize=hm_axs)
plt.title(r'MMP high-res. $\rho$ = '+str(round(sp.stats.spearmanr(scn_t1_hk[triu_hk],scn_epi_t1_hk[triu_hk])[0],2)),size=hm_lbs) #+'; p = '+str(format(t1_p,'.0e'))) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_t1_vs_epi_hk_square.svg',bbox_inches='tight')

### mmp l with forced square aspect ratio
# current min/max
hb_min = min(np.concatenate((scn_t1_lk[triu_lk],scn_epi_t1_lk[triu_lk])))-0.1
hb_max = max(np.concatenate((scn_t1_lk[triu_lk],scn_epi_t1_lk[triu_lk])))+0.1
# plot with added min/max data
plt.figure()
image = plt.hexbin(np.hstack((scn_epi_t1_lk[triu_lk],hb_min,hb_max)),np.hstack((scn_t1_lk[triu_lk],hb_min,hb_max)),cmap=plt.cm.Spectral_r,mincnt=1,gridsize=20) # ,extent=extent,bins='log' #plt.grid(True)
cb = plt.colorbar(image,spacing='uniform',extend='max') #cb.set_label('count', rotation=270)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
plt.xlabel('EPImix T$_1$-w.',size=hm_lbs); plt.xticks(fontsize=hm_axs)
plt.ylabel('T$_1$-w.',size=hm_lbs); plt.yticks(fontsize=hm_axs)
plt.title(r'MMP low-res. $\rho$ = '+str(round(sp.stats.spearmanr(scn_t1_lk[triu_lk],scn_epi_t1_lk[triu_lk])[0],2)),size=hm_lbs) #+'; p = '+str(format(t1_p,'.0e'))) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_t1_vs_epi_lk_square.svg',bbox_inches='tight')

# %% structural covariance - Jacobian

# high-res
scn_jcb_t1_hk = sp.stats.spearmanr(jcb_t1_hk_n,nan_policy='omit')[0]
scn_jcb_epi_hk = sp.stats.spearmanr(jcb_epi_hk_n,nan_policy='omit')[0]
# low-res
scn_jcb_t1_lk = sp.stats.spearmanr(jcb_t1_lk_n,nan_policy='omit')[0]
scn_jcb_epi_lk = sp.stats.spearmanr(jcb_epi_lk_n,nan_policy='omit')[0]

### diamond plot
scn_jcb_diam_hk = np.copy(scn_jcb_t1_hk)
scn_jcb_diam_hk[triu_hk] = scn_jcb_epi_hk[triu_hk]
plt.figure();
hm = sb.heatmap(scn_jcb_diam_hk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.95,vmax=.95)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_hk_lh,nr_hk], *hm.get_xlim()); hm.vlines([0,nr_hk_lh,nr_hk], *hm.get_ylim())
plt.title('EPImix T$_1$-w.',size=hm_lbs); plt.ylabel('T$_1$-w.',size=hm_lbs); 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_diam_mmp_hk.png',dpi=600,bbox_inches='tight')

### diamond plot sorted by Yeo networks
# create (sorted) diamond
scn_jcb_diam_hk = np.copy(scn_jcb_t1_hk[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord])
scn_jcb_diam_hk[triu_hk] = scn_jcb_epi_hk[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord][triu_hk]
# plot
plt.figure()
hm = sb.heatmap(scn_jcb_diam_hk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.95,vmax=.95)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
plt.title('EPImix T$_1$-w.',size=hm_lbs, pad=20); plt.ylabel('T$_1$-w.',size=hm_lbs, labelpad=20); 
bb = np.array(hm.get_position())
ef.add_subnetwork_lines(hm,nr_yeo_hk,lw=0.5)                                            # add thin black lines
ef.add_subnetwork_colours(hm,bb,nr_yeo_hk,yeo_col,lw=5,alpha=1,solid_capstyle="butt")   # add  network colour lines
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_diam_mmp_hk_yeo.png',dpi=600,bbox_inches='tight')

### diamond plot
scn_diam_lk = np.copy(scn_jcb_t1_lk)
scn_diam_lk[triu_lk] = scn_jcb_epi_lk[triu_lk]
plt.figure()
hm = sb.heatmap(scn_diam_lk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.95,vmax=.95)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_lk_lh,nr_lk], *hm.get_xlim(),lw=1); hm.vlines([0,nr_lk_lh,nr_lk], *hm.get_ylim(),lw=1)
plt.title('EPImix T$_1$-w.',size=hm_lbs, pad=20); plt.ylabel('T$_1$-w.',size=hm_lbs, labelpad=20); 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_diam_mmp_lk.png',dpi=600,bbox_inches='tight')
#if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_diam_mmp_lk.svg',bbox_inches='tight')

### plot thresholded networks

# low-res
scn_jcb_t1_lk_thr = bct.threshold_proportional(scn_jcb_t1_lk,0.10)          # threshold network (to 0.X % density)
scn_jcb_epi_lk_thr = bct.threshold_proportional(scn_jcb_epi_lk,0.10)  # threshold network (to 0.X % density)

nl.plotting.plot_connectome(scn_jcb_t1_lk_thr,node_coords=coords_l[mmp_lk,:],node_color=[col_l[i] for i in mmp_lk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.95,edge_vmax=.95) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_t1_lk_net.png',dpi=500)

nl.plotting.plot_connectome(scn_jcb_epi_lk_thr,node_coords=coords_l[mmp_lk,:],node_color=[col_l[i] for i in mmp_lk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.95,edge_vmax=.95) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_epi_lk_net.png',dpi=500)

# high-res
scn_jcb_t1_hk_thr = bct.threshold_proportional(scn_jcb_t1_hk,0.005)         # threshold network (to 0.X % density)
scn_jcb_epi_hk_thr = bct.threshold_proportional(scn_jcb_epi_hk,0.005) # threshold network (to 0.X % density)

nl.plotting.plot_connectome(scn_jcb_t1_hk_thr,node_coords=coords_h[mmp_hk,:],node_color=[col_h[i] for i in mmp_hk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.95,edge_vmax=.95) 
#nl.plotting.plot_connectome(scn_jcb_t1_hk_thr,node_coords=coords_h[mmp_hk,:],node_color=[yeo_col[i-1] for i in yeo_mmp_hk],node_size=20,edge_cmap='coolwarm',edge_vmin=-1,edge_vmax=1) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_t1_hk_net.png',dpi=500)

nl.plotting.plot_connectome(scn_jcb_epi_hk_thr,node_coords=coords_h[mmp_hk,:],node_color=[col_h[i] for i in mmp_hk],node_size=20,edge_cmap='coolwarm',edge_vmin=-.95,edge_vmax=.95) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_epi_hk_net.png',dpi=500)

## correlate structural covariance matrices

### mmp h with forced square aspect ratio
# current min/max
hb_min = min(np.concatenate((scn_jcb_t1_hk[triu_hk],scn_jcb_epi_hk[triu_hk])))-0.1
hb_max = max(np.concatenate((scn_jcb_t1_hk[triu_hk],scn_jcb_epi_hk[triu_hk])))+0.1
# plot with added min/max data
plt.figure()
image = plt.hexbin(np.hstack((scn_jcb_epi_hk[triu_hk],hb_min,hb_max)),np.hstack((scn_jcb_t1_hk[triu_hk],hb_min,hb_max)),cmap=plt.cm.Spectral_r,mincnt=1,gridsize=50) # ,extent=extent,bins='log' #plt.grid(True)
cb = plt.colorbar(image,spacing='uniform',extend='max') #cb.set_label('count', rotation=270)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
plt.xlabel('EPImix T$_1$-w.',size=hm_lbs); plt.xticks(fontsize=hm_axs)
plt.ylabel('T$_1$-w.',size=hm_lbs); plt.yticks(fontsize=hm_axs)
plt.title(r'MMP high-res. $\rho$ = '+str(round(sp.stats.spearmanr(scn_jcb_t1_hk[triu_hk],scn_jcb_epi_hk[triu_hk])[0],2)),size=hm_lbs) #+'; p = '+str(format(jcb_p,'.0e'))) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_t1_vs_epi_hk_square.svg',bbox_inches='tight')

### mmp l with forced square aspect ratio
# current min/max
hb_min = min(np.concatenate((scn_jcb_t1_lk[triu_lk],scn_jcb_epi_lk[triu_lk])))-0.1
hb_max = max(np.concatenate((scn_jcb_t1_lk[triu_lk],scn_jcb_epi_lk[triu_lk])))+0.1
# plot with added min/max data
plt.figure()
image = plt.hexbin(np.hstack((scn_jcb_epi_lk[triu_lk],hb_min,hb_max)),np.hstack((scn_jcb_t1_lk[triu_lk],hb_min,hb_max)),cmap=plt.cm.Spectral_r,mincnt=1,gridsize=20) # ,extent=extent,bins='log' #plt.grid(True)
cb = plt.colorbar(image,spacing='uniform',extend='max') #cb.set_label('count', rotation=270)
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs)
plt.xlabel('EPImix T$_1$-w.',size=hm_lbs); plt.xticks(fontsize=hm_axs)
plt.ylabel('T$_1$-w.',size=hm_lbs); plt.yticks(fontsize=hm_axs)
plt.title(r'MMP low-res. $\rho$ = '+str(round(sp.stats.spearmanr(scn_jcb_t1_lk[triu_lk],scn_jcb_epi_lk[triu_lk])[0],2)),size=hm_lbs) #+'; p = '+str(format(jcb_p,'.0e'))) 
if save_fig_net: plt.savefig(plot_dir_net+'/scn_jcb_t1_vs_epi_lk_square.svg',bbox_inches='tight')

# %%

"""

Test-retest reliability
 
"""

# plot directory for main analyses
plot_dir_rt = plot_dir+'/rt'
if not os.path.isdir(plot_dir_rt):
    os.mkdir(plot_dir_rt)
    
# figure saving condition
save_fig_rt = False

# %% test-retest data

# ADAPTA - retest

# subjects with retest data - ADAPTA30 - ADAPTA39
sub_rt = np.array(['ADAPTA'+str(i) for i in np.arange(30,40)])
ns_rt = len(sub_rt)
sub_rt_id = []
for s in range(ns_rt):
    sub_rt_id.append(np.where(sub==sub_rt[s])[0][0])
    
# demographics for subset of participants with retest data
age_rt = age[sub_rt_id]
sex_rt = sex[sub_rt_id]

# recreate arrays from numpy (values not within the brain mask == 0, but array dimensions are compatible with analysis code below)
# epi rt
if os.path.isfile(epimix_dir+'/epi_rt_bn.npy'):  # if file exists, load it    
    epi_rt_bn = np.load(epimix_dir+'/epi_rt_bn.npy')
    epi_rt = np.empty([ns_rt,nc,nvox])
    epi_rt[:,:,bn_vec!=0] = epi_rt_bn
    del epi_rt_bn
# jcb_epi_rt
if os.path.isfile(epimix_dir+'/jcb_epi_rt_bn.npy'):  # if file exists, load it    
    jcb_epi_rt_bn = np.load(epimix_dir+'/jcb_epi_rt_bn.npy')
    jcb_epi_rt = np.empty([ns_rt,nvox])
    jcb_epi_rt[:,bn_vec!=0] = jcb_epi_rt_bn
    del jcb_epi_rt_bn

# %% average re-test data within ROIs

# mmp h
epi_rt_hk = np.empty([ns_rt,nc,nr_hk])
jcb_epi_rt_hk = np.empty([ns_rt,nr_hk])
for r in range(nr_hk):
    # EPImix
    temp_epi = epi_rt[:,:,mmp_h_vec==mmp_h_id[mmp_hk[r]]]   # replace zeros by nans
    epi_rt_hk[:,:,r] = np.median(temp_epi,axis=2)           # calculate median, ignoring nans
    # Jacobian
    temp_jcb = jcb_epi_rt[:,mmp_h_vec==mmp_h_id[mmp_hk[r]]] # replace zeros by nans
    jcb_epi_rt_hk[:,r] = np.median(temp_jcb,axis=1)         # calculate median, ignoring nans

# mmp l
epi_rt_lk = np.empty([ns_rt,nc,nr_lk])
jcb_epi_rt_lk = np.empty([ns_rt,nr_lk])
for r in range(nr_lk):
    # EPImix
    temp_epi = epi_rt[:,:,mmp_l_vec==(mmp_lk[r]+1)]         # replace zeros by nans
    epi_rt_lk[:,:,r] = np.median(temp_epi,axis=2)           # calculate median, ignoring nans
    # Jacobian
    temp_jcb = jcb_epi_rt[:,mmp_l_vec==(mmp_lk[r]+1)]         # replace zeros by nans
    jcb_epi_rt_lk[:,r] = np.median(temp_jcb,axis=1)           # calculate median, ignoring nans

del temp_epi
del temp_jcb

# %% retest MSNs
            
# normalise ROI data
epi_rt_hk_n = np.empty([ns_rt,nc,nr_hk])
epi_rt_lk_n = np.empty([ns_rt,nc,nr_lk])
jcb_epi_rt_hk_n = np.empty([ns_rt,nr_hk])
jcb_epi_rt_lk_n = np.empty([ns_rt,nr_lk])
for s in range(ns_rt):
    # EPImix
    for c in range(nc):
        epi_rt_hk_n[s,c,:] = np.divide(epi_rt_hk[s,c,:] - np.nanmedian(epi_rt_hk[s,c,:]),ef.mad(epi_rt_hk[s,c,:]))
        epi_rt_lk_n[s,c,:] = np.divide(epi_rt_lk[s,c,:] - np.nanmedian(epi_rt_lk[s,c,:]),ef.mad(epi_rt_lk[s,c,:]))
    # Jacobian
    jcb_epi_rt_hk_n[s,:] = np.divide(jcb_epi_rt_hk[s,:] - np.nanmedian(jcb_epi_rt_hk[s,:]),ef.mad(jcb_epi_rt_hk[s,:]))
    jcb_epi_rt_lk_n[s,:] = np.divide(jcb_epi_rt_lk[s,:] - np.nanmedian(jcb_epi_rt_lk[s,:]),ef.mad(jcb_epi_rt_lk[s,:]))
        
# construct MSNs
msn_rt_hk = np.zeros([nr_hk,nr_hk,ns_rt])
msn_rt_lk = np.zeros([nr_lk,nr_lk,ns_rt])

# # using EPImix contrasts only
# for s in range(ns_rt):
#     print(sub_rt[s])
#     msn_rt_hk[:,:,s],_ = sp.stats.spearmanr(epi_rt_hk_n[s,:,:],nan_policy='propagate')
#     msn_rt_lk[:,:,s],_ = sp.stats.spearmanr(epi_rt_lk_n[s,:,:],nan_policy='propagate')

# using epimix contrasts + Jacobian
for s in range(ns_rt):
    print(sub_rt[s])
    msn_rt_hk[:,:,s],_ = sp.stats.spearmanr(np.vstack((epi_rt_hk_n[s,:,:],jcb_epi_rt_hk_n[s,:])),nan_policy='propagate')
    msn_rt_lk[:,:,s],_ = sp.stats.spearmanr(np.vstack((epi_rt_lk_n[s,:,:],jcb_epi_rt_lk_n[s,:])),nan_policy='propagate')

# average MSNs (for plotting)
# test
msn_t_hk_m = np.nanmean(msn_hk[:,:,sub_rt_id],axis=2)
msn_t_lk_m = np.nanmean(msn_lk[:,:,sub_rt_id],axis=2)  
# retest
msn_rt_hk_m = np.nanmean(msn_rt_hk,axis=2)
msn_rt_lk_m = np.nanmean(msn_rt_lk,axis=2)

### diamond plots

# mmp hk
msn_rt_diam_hk = np.copy(msn_t_hk_m)
msn_rt_diam_hk[triu_hk] = msn_rt_hk_m[triu_hk]
plt.figure()
hm = sb.heatmap(msn_rt_diam_hk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_hk_lh,nr_hk], *hm.get_xlim()); hm.vlines([0,nr_hk_lh,nr_hk], *hm.get_ylim())
plt.title('re-test',size=hm_lbs, pad=20); plt.ylabel('test',size=hm_lbs, labelpad=20); 
if save_fig_rt: plt.savefig(plot_dir_net+'/msn_rt_diam_mmp_hk.png',dpi=600,bbox_inches='tight')

# mmp hk sorted by Yeo networks
# create sorted diamond
msn_rt_diam_hk = np.copy(msn_t_hk_m[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord])
msn_rt_diam_hk[triu_hk] = msn_rt_hk_m[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord][triu_hk]
# plot
plt.figure()
hm = sb.heatmap(msn_rt_diam_hk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
plt.title('re-test',size=hm_lbs, pad=20); plt.ylabel('test',size=hm_lbs, labelpad=20); 
bb = np.array(hm.get_position())
ef.add_subnetwork_lines(hm,nr_yeo_hk,lw=0.5)                                            # add thin black lines
ef.add_subnetwork_colours(hm,bb,nr_yeo_hk,yeo_col,lw=5,alpha=1,solid_capstyle="butt")   # add  network colour lines
if save_fig_rt: plt.savefig(plot_dir_net+'/msn_rt_diam_mmp_hk_yeo.png',dpi=600,bbox_inches='tight')

# mmp lk
msn_rt_diam_lk = np.copy(msn_t_lk_m)
msn_rt_diam_lk[triu_lk] = msn_rt_lk_m[triu_lk]
plt.figure()
hm = sb.heatmap(msn_rt_diam_lk,cmap='coolwarm',xticklabels=False,yticklabels=False,cbar_kws={'label': r'Spearman $\rho$','extend': 'max'},vmin=-.8,vmax=.8)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_lk_lh,nr_lk], *hm.get_xlim()); hm.vlines([0,nr_lk_lh,nr_lk], *hm.get_ylim())
plt.title('re-test',size=hm_lbs, pad=20); plt.ylabel('test',size=hm_lbs, labelpad=20); 
if save_fig_rt: plt.savefig(plot_dir_net+'/msn_rt_diam_mmp_lk.png',dpi=600,bbox_inches='tight')

# ### plot "connectome" visualisation - top positive and negative edges
# # mmp h - 0.005 = 0.5% edges
# msn_hk_m_thr_pn = bct.threshold_proportional(msn_rt_hk_m,0.0025) - bct.threshold_proportional(-msn_rt_hk_m,0.0025)
# nl.plotting.plot_connectome(msn_hk_m_thr_pn,node_coords=coords_h[mmp_hk,:],node_color=list(col_h[mmp_hk]),node_size=20,edge_cmap='coolwarm') #node_size=10*np.sum(msn_m_thr,0)
# if save_fig_rt: plt.savefig('/Users/Frantisek/Desktop/msn_m_net_posneg.png',dpi=500,bbox_inches='tight')
# # mmp l - 0.1 = 10% edges
# msn_lk_m_thr_pn = bct.threshold_proportional(msn_rt_lk_m,0.05) - bct.threshold_proportional(-msn_rt_lk_m,0.05)
# nl.plotting.plot_connectome(msn_lk_m_thr_pn,node_coords=coords_l[mmp_lk,:],node_color=list(col_l[mmp_lk]),node_size=20,edge_cmap='coolwarm') #node_size=10*np.sum(msn_m_thr,0)
# if save_fig_rt: plt.savefig('/Users/Frantisek/Desktop/msn_m_net_posneg.png',dpi=500,bbox_inches='tight')

# %% retest correlations
            
### "half matrix" indices (for matrix of size ns_rt)
# indices (in "rt_rho_" matrices) of:
# - self-similarity ("sim_self_id" = off-diagonal)
# - other-similarity ("sim_other_id" = rest of off-diagonal block)
sim_self_id_rt = ef.kth_diag_indices(np.zeros([ns_rt,ns_rt]),0)
sim_other_id_rt = tuple([np.concatenate((np.triu_indices(ns_rt,1)[0],np.tril_indices(ns_rt,-1)[0])),np.concatenate((np.triu_indices(ns_rt,1)[1],np.tril_indices(ns_rt,-1)[1]))])
### test plot
# test matrix
test_mat = np.zeros([ns_rt,ns_rt])
test_mat[sim_self_id_rt] = 1
test_mat[sim_other_id_rt] = 2
# colormap
cmap_cst = sb.xkcd_palette(['white','red','grey'])
# plot
fig = plt.figure()
fig.add_subplot(111,aspect='equal')
hm = sb.heatmap(test_mat,cmap=cmap_cst,xticklabels=False,yticklabels=False,cbar=True,vmin=0,vmax=2)#,cbar_kws={"ticks":[0,1]})
hm.hlines([0,ns_rt], *hm.get_xlim()); hm.vlines([0,ns_rt], *hm.get_ylim())
plt.xlabel('test',size=hm_lbs-3); plt.ylabel('retest',size=hm_lbs-3)
if save_fig_rt: plt.savefig(plot_dir_rt+'/idiff_rt_example_mat.svg', bbox_inches='tight')  

# calculate (and combine) metrics
rt_rho = np.zeros([3*ns_rt,1])  # correlations (a la identifiability)
rt_idiff = np.zeros([3,nc])     # identifiability
rt_icc = np.zeros([3,nc])       # ICC
for e in range(nc):
    print(epimix_contr[e])
    # correlations
    rt_rho_v = sp.stats.spearmanr(np.transpose(np.concatenate((epi[sub_rt_id,:,:],epi_rt))[:,e,fov_gm_vec==1]))[0][0:ns_rt,(ns_rt):(2*ns_rt)]
    rt_rho_hk = sp.stats.spearmanr(np.transpose(np.concatenate((epi_hk[sub_rt_id,:,:],epi_rt_hk))[:,e,:]),nan_policy='omit')[0][0:ns_rt,(ns_rt):(2*ns_rt)]
    rt_rho_lk = sp.stats.spearmanr(np.transpose(np.concatenate((epi_lk[sub_rt_id,:,:],epi_rt_lk))[:,e,:]),nan_policy='omit')[0][0:ns_rt,(ns_rt):(2*ns_rt)]
    # combine correlations
    rt_rho = np.hstack((rt_rho,np.vstack((rt_rho_v,rt_rho_hk,rt_rho_lk))))
    # idiff
    rt_idiff[0,e] = np.median(rt_rho_v[sim_self_id_rt])-np.median(rt_rho_v[sim_other_id_rt])
    rt_idiff[1,e] = np.median(rt_rho_hk[sim_self_id_rt])-np.median(rt_rho_hk[sim_other_id_rt])
    rt_idiff[2,e] = np.median(rt_rho_lk[sim_self_id_rt])-np.median(rt_rho_lk[sim_other_id_rt])
    # icc
    rt_icc[0,e] = pg.intraclass_corr(data=pd.DataFrame({
            'subject':np.concatenate((np.repeat(np.arange(0,ns_rt),nvox_fov_gm),np.repeat(np.arange(0,ns_rt),nvox_fov_gm))),
            'scan':np.concatenate((np.repeat('T',nvox_fov_gm*ns_rt),np.repeat('RT',nvox_fov_gm*ns_rt))),
            'score':np.concatenate((epi[sub_rt_id,:,:],epi_rt))[:,e,fov_gm_vec==1].flatten()}),
        targets='subject', raters='scan',ratings='score')['ICC'][2]
    rt_icc[1,e] = pg.intraclass_corr(data=pd.DataFrame({
            'subject':np.concatenate((np.repeat(np.arange(0,ns_rt),nr_hk),np.repeat(np.arange(0,ns_rt),nr_hk))),
            'scan':np.concatenate((np.repeat('T',nr_hk*ns_rt),np.repeat('RT',nr_hk*ns_rt))),
            'score':np.concatenate((epi_hk[sub_rt_id,:,:],epi_rt_hk))[:,e,:].flatten()}),
        targets='subject', raters='scan',ratings='score')['ICC'][2]
    rt_icc[2,e] = pg.intraclass_corr(data=pd.DataFrame({
            'subject':np.concatenate((np.repeat(np.arange(0,ns_rt),nr_lk),np.repeat(np.arange(0,ns_rt),nr_lk))),
            'scan':np.concatenate((np.repeat('T',nr_lk*ns_rt),np.repeat('RT',nr_lk*ns_rt))),
            'score':np.concatenate((epi_lk[sub_rt_id,:,:],epi_rt_lk))[:,e,:].flatten()}),
        targets='subject', raters='scan',ratings='score')['ICC'][2]
# remove first row of correlations
if rt_rho.shape[1] > nc*ns_rt:
    rt_rho = np.delete(rt_rho,(0),axis=1)

# subplots
f, ax = plt.subplots(nrows=3, ncols=nc, figsize=(12, 5),sharex=True,sharey=True)
for e in range(nc):
    # voxels
    ax = plt.subplot(3, nc, e+1, aspect='equal')
    hm = sb.heatmap(rt_rho[range(ns_rt),:][:,range(e*ns_rt,(e+1)*ns_rt)],cmap='plasma_r',xticklabels=False,yticklabels=False,cbar=False,vmin=0,vmax=1) #cbar_kws={'label': r'Spearman $\rho$'},
    hm.hlines([0,ns_rt], *hm.get_xlim()); hm.vlines([0,ns_rt], *hm.get_ylim())   
    if e==0: plt.ylabel('voxels', size=hm_lbs, labelpad=15)
    # mmp h
    ax = plt.subplot(3, nc, nc+e+1, aspect='equal')
    hm = sb.heatmap(rt_rho[range(ns_rt,2*ns_rt),:][:,range(e*ns_rt,(e+1)*ns_rt)],cmap='plasma_r',xticklabels=False,yticklabels=False,cbar=False,vmin=0,vmax=1) #cbar_kws={'label': r'Spearman $\rho$'},
    hm.hlines([0,ns_rt], *hm.get_xlim()); hm.vlines([0,ns_rt], *hm.get_ylim())  
    if e==0: plt.ylabel('MMP h-r.', size=hm_lbs, labelpad=15)
    # mmp l
    ax = plt.subplot(3, nc, 2*nc+e+1, aspect='equal')
    hm = sb.heatmap(rt_rho[range(2*ns_rt,3*ns_rt),:][:,range(e*ns_rt,(e+1)*ns_rt)],cmap='plasma_r',xticklabels=False,yticklabels=False,cbar=False,vmin=0,vmax=1) #cbar_kws={'label': r'Spearman $\rho$'},
    hm.hlines([0,ns_rt], *hm.get_xlim()); hm.vlines([0,ns_rt], *hm.get_ylim()) 
    if e==0: plt.ylabel('MMP l-r.', size=hm_lbs, labelpad=15)
    plt.xlabel(epimix_contr[e], size=hm_lbs, labelpad=15)
# colorbar
f.subplots_adjust(right=0.8)
cax = f.add_axes([0.82, 0.15, 0.015, 0.7])
cb = f.colorbar(cm.ScalarMappable(cmap='plasma_r'), cax=cax)
cax.tick_params(labelsize=hm_axs); cb.set_label(r'Spearman $\rho$',size=hm_lbs)
if save_fig_rt: plt.savefig(plot_dir_rt+'/epimix_rt_rho.svg',bbox_inches='tight')
        
# Idiff
fig = plt.figure(figsize=(12, 5)); # figsize=(12, 3)
fig.add_subplot(111) #fig.add_subplot(1,nc,e+1,aspect='equal')
hm = sb.heatmap(rt_idiff,cmap='plasma_r',cbar_kws={'label': r'identifiability I$_{diff}$'},vmin=0.1,vmax=0.7) #yticklabels=['voxels','360 ROI','44 ROI'],xticklabels=epimix_contr,
hm.figure.axes[-1].yaxis.label.set_size(axs)
hm.figure.axes[-1].tick_params(labelsize=axs-2)
hm.set_yticklabels(['voxels','MMP h-r.','MMP l-r.'],rotation=0,size=axs); hm.set_xticklabels(epimix_contr,size=axs)
hm.hlines([i for i in range(3+2)], *hm.get_xlim())
hm.vlines([i for i in range(nc+2)], *hm.get_ylim())
if save_fig_rt: plt.savefig(plot_dir_rt+'/epimix_rt_idiff.svg',bbox_inches='tight')

# ICC
fig = plt.figure(figsize=(12, 5)); # figsize=(12, 3)
fig.add_subplot(111) #fig.add_subplot(1,nc,e+1,aspect='equal')
hm = sb.heatmap(rt_icc,cmap='Reds',cbar_kws={'label': r'ICC'},vmin=0,vmax=1) #yticklabels=['voxels','360 ROI','44 ROI'],xticklabels=epimix_contr,
hm.figure.axes[-1].yaxis.label.set_size(axs)
hm.figure.axes[-1].tick_params(labelsize=axs-2)
hm.set_yticklabels(['voxels','MMP h-r.','MMP l-r.'],rotation=0,size=axs); hm.set_xticklabels(epimix_contr,size=axs)
hm.hlines([i for i in range(3+2)], *hm.get_xlim())
hm.vlines([i for i in range(nc+2)], *hm.get_ylim())
if save_fig_rt: plt.savefig(plot_dir_rt+'/epimix_rt_icc.svg',bbox_inches='tight')

# %% regional ICC

# calculate
if os.path.isfile(data_out_dir+'/rt_icc.npz'):  # if file exists, load it
    npz = np.load(data_out_dir+'/rt_icc.npz')
    rt_icc_v = npz['rt_icc_v']
    rt_icc_hk = npz['rt_icc_hk']
    rt_icc_lk = npz['rt_icc_lk']
    del npz
else:      
    # voxels
    rt_icc_v = np.zeros([nc,nvox_fov_bn]) # voxelwise - within "brain" mask only
    for e in range(nc):
        print(epimix_contr[e])
        for i in range(nvox_fov_bn):
            if i % 1000 == 0: print(i)
            rt_icc_v[e,i] = pg.intraclass_corr(data=pd.DataFrame({
                    'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
                    'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
                    'score':np.concatenate((epi[sub_rt_id,e,:],epi_rt[:,e,:]))[:,fov_bn_vec==1][:,i]}),
                targets='subject', raters='scan',ratings='score')['ICC'][2]
    # mmp h
    rt_icc_hk = np.zeros([nc,nr_hk])
    for e in range(nc):
        print(epimix_contr[e])
        for i in range(nr_hk):
            if i % 100 == 0: print(i)
            rt_icc_hk[e,i] = pg.intraclass_corr(data=pd.DataFrame({
                    'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
                    'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
                    'score':np.concatenate((epi_hk[sub_rt_id,:,:],epi_rt_hk))[:,e,i]}),
                targets='subject', raters='scan',ratings='score')['ICC'][2]
    # mmp l
    rt_icc_lk = np.zeros([nc,nr_lk])
    for e in range(nc):
        print(epimix_contr[e])
        for i in range(nr_lk):
            rt_icc_lk[e,i] = pg.intraclass_corr(data=pd.DataFrame({
                    'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
                    'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
                    'score':np.concatenate((epi_lk[sub_rt_id,:,:],epi_rt_lk))[:,e,i]}),
                targets='subject', raters='scan',ratings='score')['ICC'][2]
    # save outputs
    np.savez(data_out_dir+'/rt_icc.npz',rt_icc_v = rt_icc_v,rt_icc_hk = rt_icc_hk,rt_icc_lk = rt_icc_lk)

### plots
# convert each contrast back to nii and plot
c_lim = (0,1)
cut_crd = (30, 0, 5)
for e in range(nc):   
    # convert to full (nvox) nii
    temp_icc = np.zeros(nvox); temp_icc[fov_bn_vec==1] = rt_icc_v[e,:]
    # MNI && FoV only
    ef.plot_nl_image_masked(temp_icc, fov_bn_vec, nii_shape, mmp_h_nii.affine, cmap='Reds', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False,annotate=False)
    if save_fig_rt: plt.savefig(plot_dir_rt+'/rt_icc_bn_fov_'+epimix_contr[e]+'.png',dpi=500, bbox_inches='tight')  
    # GM && FoV only
    ef.plot_nl_image_masked(temp_icc, fov_gm_vec, nii_shape, mmp_h_nii.affine, cmap='Reds', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False,annotate=False)
    if save_fig_rt: plt.savefig(plot_dir_rt+'/rt_icc_gm_fov_'+epimix_contr[e]+'.png',dpi=500, bbox_inches='tight')    

# mmp h plots
if save_fig_rt: 
    c_lim = (0,1) #(0,max(rt_icc_hk.flatten()))
    for e in range(nc):
        ef.pscalar_mmp_hk(file_out=plot_dir_rt+'/rt_icc_hk_'+epimix_contr[e]+'.png', pscalars_hk=rt_icc_hk[e,:], mmp_hk=mmp_hk, cmap='Reds',vrange=c_lim)
    ef.plot_cbar(c_lim=c_lim, cmap_nm='Reds', c_label='ICC', lbs=14, save_path=plot_dir_rt+'/rt_icc_hk_cbar.png')
    
# mmp l plots
if save_fig_rt: 
    c_lim = (0,1) #(0,max(rt_icc_lk.flatten()))
    for e in range(nc):
        ef.pscalar_mmp_lk(file_out=plot_dir_rt+'/rt_icc_lk_'+epimix_contr[e]+'.png', pscalars_lk=rt_icc_lk[e,:], mmp_lk=mmp_lk, mmp_ds_ids=mmp_ds_ids, cmap='Reds',vrange=c_lim)
    ef.plot_cbar(c_lim=c_lim, cmap_nm='Reds', c_label='ICC', lbs=14, save_path=plot_dir_rt+'/rt_icc_lk_cbar.png')

# Median and percentile values of ICC
for e in range(nc):
    print(epimix_contr[e])
    temp_icc = np.zeros(nvox); temp_icc[fov_bn_vec==1] = rt_icc_v[e,:]
    print(str(round(np.median(temp_icc[fov_bn_vec==1]),2))+' ['+str(round(np.percentile(temp_icc[fov_bn_vec==1],25),2))+','+str(round(np.percentile(temp_icc[fov_bn_vec==1],75),2))+']')
    print(str(round(np.median(temp_icc[fov_gm_vec==1]),2))+' ['+str(round(np.percentile(temp_icc[fov_gm_vec==1],25),2))+','+str(round(np.percentile(temp_icc[fov_gm_vec==1],75),2))+']')
    print(str(round(np.median(rt_icc_hk[e,:]),2))+' ['+str(round(np.percentile(rt_icc_hk[e,:],25),2))+','+str(round(np.percentile(rt_icc_hk[e,:],75),2))+']')
    print(str(round(np.median(rt_icc_lk[e,:]),2))+' ['+str(round(np.percentile(rt_icc_lk[e,:],25),2))+','+str(round(np.percentile(rt_icc_lk[e,:],75),2))+']')
   
# %% regional ICC for Jacobians
    
# calculate
if os.path.isfile(data_out_dir+'/rt_jcb_icc.npz'):  # if file exists, load it
    npz = np.load(data_out_dir+'/rt_jcb_icc.npz')
    rt_jcb_icc_v = npz['rt_jcb_icc_v']
    rt_jcb_icc_hk = npz['rt_jcb_icc_hk']
    rt_jcb_icc_lk = npz['rt_jcb_icc_lk']
    del npz
else:      
    # voxels
    rt_jcb_icc_v = np.zeros([nvox_fov_bn]) # voxelwise - within "brain" mask only
    for i in range(nvox_fov_bn):
        if i % 1000 == 0: print(i)
        rt_jcb_icc_v[i] = pg.intraclass_corr(data=pd.DataFrame({
                'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
                'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
                'score':np.concatenate((jcb_epi[sub_rt_id,:],jcb_epi_rt))[:,fov_bn_vec==1][:,i]}),
            targets='subject', raters='scan',ratings='score')['ICC'][2]
    # mmp h
    rt_jcb_icc_hk = np.zeros([nr_hk])
    for e in range(nc):
        print(epimix_contr[e])
        for i in range(nr_hk):
            if i % 100 == 0: print(i)
            rt_jcb_icc_hk[i] = pg.intraclass_corr(data=pd.DataFrame({
                    'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
                    'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
                    'score':np.concatenate((jcb_epi_hk[sub_rt_id,:],jcb_epi_rt_hk))[:,i]}),
                targets='subject', raters='scan',ratings='score')['ICC'][2]
    # mmp l
    rt_jcb_icc_lk = np.zeros([nr_lk])
    for e in range(nc):
        print(epimix_contr[e])
        for i in range(nr_lk):
            rt_jcb_icc_lk[i] = pg.intraclass_corr(data=pd.DataFrame({
                    'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
                    'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
                    'score':np.concatenate((jcb_epi_lk[sub_rt_id,:],jcb_epi_rt_lk))[:,i]}),
                targets='subject', raters='scan',ratings='score')['ICC'][2]
    # save outputs
    np.savez(data_out_dir+'/rt_jcb_icc.npz',rt_jcb_icc_v = rt_jcb_icc_v,rt_jcb_icc_hk = rt_jcb_icc_hk,rt_jcb_icc_lk = rt_jcb_icc_lk)

### plots
# convert each contrast back to nii and plot
c_lim = (0,1)
cut_crd = (30, 0, 5)
# convert to full (nvox) nii
temp_icc = np.zeros(nvox); temp_icc[fov_bn_vec==1] = rt_jcb_icc_v
# MNI && FoV only
ef.plot_nl_image_masked(temp_icc, fov_bn_vec, nii_shape, mmp_h_nii.affine, cmap='Reds', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False,annotate=False)
if save_fig_rt: plt.savefig(plot_dir_rt+'/rt_icc_bn_fov_Jcb.png',dpi=500, bbox_inches='tight')  
# GM && FoV only
ef.plot_nl_image_masked(temp_icc, fov_gm_vec, nii_shape, mmp_h_nii.affine, cmap='Reds', clim=c_lim, cut_coords=cut_crd, draw_cross=False,black_bg=False,annotate=False)
if save_fig_rt: plt.savefig(plot_dir_rt+'/rt_icc_gm_fov_Jcb.png',dpi=500, bbox_inches='tight')    

# mmp h plots
if save_fig_rt: 
    c_lim = (0,1) #(0,max(rt_icc_hk.flatten()))
    ef.pscalar_mmp_hk(file_out=plot_dir_rt+'/rt_icc_hk_Jcb.png', pscalars_hk=rt_jcb_icc_hk, mmp_hk=mmp_hk, cmap='Reds',vrange=c_lim)
    #ef.plot_cbar(c_lim=c_lim, cmap_nm='Reds', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_rt+'/rt_icc_hk_cbar.png')
    
# mmp l plots
if save_fig_rt: 
    c_lim = (0,1) #(0,max(rt_icc_lk.flatten()))
    ef.pscalar_mmp_lk(file_out=plot_dir_rt+'/rt_icc_lk_Jcb.png', pscalars_lk=rt_jcb_icc_lk, mmp_lk=mmp_lk, mmp_ds_ids=mmp_ds_ids, cmap='Reds',vrange=c_lim)
    #ef.plot_cbar(c_lim=c_lim, cmap_nm='Reds', c_label=r'Spearman $\rho$', lbs=14, save_path=plot_dir_rt+'/rt_icc_lk_cbar.png')  

# Median and percentile values of ICC
temp_icc = np.zeros(nvox); temp_icc[fov_bn_vec==1] = rt_jcb_icc_v
print(str(round(np.median(temp_icc[fov_bn_vec==1]),2))+' ['+str(round(np.percentile(temp_icc[fov_bn_vec==1],25),2))+','+str(round(np.percentile(temp_icc[fov_bn_vec==1],75),2))+']')
print(str(round(np.median(temp_icc[fov_gm_vec==1]),2))+' ['+str(round(np.percentile(temp_icc[fov_gm_vec==1],25),2))+','+str(round(np.percentile(temp_icc[fov_gm_vec==1],75),2))+']')
print(str(round(np.median(rt_jcb_icc_hk),2))+' ['+str(round(np.percentile(rt_jcb_icc_hk,25),2))+','+str(round(np.percentile(rt_jcb_icc_hk,75),2))+']')
print(str(round(np.median(rt_jcb_icc_lk),2))+' ['+str(round(np.percentile(rt_jcb_icc_lk,25),2))+','+str(round(np.percentile(rt_jcb_icc_lk,75),2))+']')

# %% MSN test-retest

# number of edges in upper triangular
nedge_hk = int(nr_hk*(nr_hk-1)/2)
nedge_lk = int(nr_lk*(nr_lk-1)/2)

# vectorize MSN upper-triangular parts
msn_t_hk_vec = np.zeros([ns_rt,nedge_hk])   # mmp h T
msn_rt_hk_vec = np.zeros([ns_rt,nedge_hk])  # mmp h RT
msn_t_lk_vec = np.zeros([ns_rt,nedge_lk])   # mmp l T
msn_rt_lk_vec = np.zeros([ns_rt,nedge_lk])  # mmp l RT
for s in range(ns_rt):
    msn_t_hk_vec[s,:] = msn_hk[:,:,sub_rt_id[s]][triu_hk]
    msn_rt_hk_vec[s,:] = msn_rt_hk[:,:,s][triu_hk]
    msn_t_lk_vec[s,:] = msn_lk[:,:,sub_rt_id[s]][triu_lk]
    msn_rt_lk_vec[s,:] = msn_rt_lk[:,:,s][triu_lk]

# correlations
msn_rt_hk_rho = sp.stats.spearmanr(np.transpose(np.concatenate((msn_t_hk_vec,msn_rt_hk_vec))))[0][0:ns_rt,(ns_rt):(2*ns_rt)]
msn_rt_lk_rho = sp.stats.spearmanr(np.transpose(np.concatenate((msn_t_lk_vec,msn_rt_lk_vec))))[0][0:ns_rt,(ns_rt):(2*ns_rt)]

# subplots
# mmp h
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(3, 5),sharex=True,sharey=True)
ax = plt.subplot(2,1,1, aspect='equal')
hm = sb.heatmap(msn_rt_hk_rho,cmap='plasma_r',xticklabels=False,yticklabels=False,cbar=False,vmin=0,vmax=1) #cbar_kws={'label': r'Spearman $\rho$'},
hm.hlines([0,ns_rt], *hm.get_xlim()); hm.vlines([0,ns_rt], *hm.get_ylim())   
plt.ylabel('MMP h-r.', size=hm_lbs, labelpad=15)
# mmp l
ax = plt.subplot(2,1,2, aspect='equal')
hm = sb.heatmap(msn_rt_lk_rho,cmap='plasma_r',xticklabels=False,yticklabels=False,cbar=False,vmin=0,vmax=1) #cbar_kws={'label': r'Spearman $\rho$'},
hm.hlines([0,ns_rt], *hm.get_xlim()); hm.vlines([0,ns_rt], *hm.get_ylim()) 
plt.ylabel('MMP l-r.', size=hm_lbs, labelpad=15)
plt.xlabel('MSN', size=hm_lbs, labelpad=15)
# colorbar
f.subplots_adjust(right=0.8)
cax = f.add_axes([0.82, 0.15, 0.05, 0.7])
cb = f.colorbar(cm.ScalarMappable(cmap='plasma_r'), cax=cax)
cax.tick_params(labelsize=hm_axs); cb.set_label(r'Spearman $\rho$',size=hm_lbs)
if save_fig_rt: plt.savefig(plot_dir_rt+'/msn_rt_rho.svg',bbox_inches='tight')

# idiff
msn_rt_hk_idiff = np.median(msn_rt_hk_rho[sim_self_id_rt])-np.median(msn_rt_hk_rho[sim_other_id_rt])
msn_rt_lk_idiff = np.median(msn_rt_lk_rho[sim_self_id_rt])-np.median(msn_rt_lk_rho[sim_other_id_rt])

### icc
## global
msn_rt_h_icc = pg.intraclass_corr(data=pd.DataFrame({
        'subject':np.concatenate((np.repeat(np.arange(0,ns_rt),nedge_hk),np.repeat(np.arange(0,ns_rt),nedge_hk))),
        'scan':np.concatenate((np.repeat('T',nedge_hk*ns_rt),np.repeat('RT',nedge_hk*ns_rt))),
        'score':np.concatenate((msn_t_hk_vec,msn_rt_hk_vec)).flatten()}),
    targets='subject', raters='scan',ratings='score')['ICC'][2]
msn_rt_l_icc = pg.intraclass_corr(data=pd.DataFrame({
        'subject':np.concatenate((np.repeat(np.arange(0,ns_rt),nedge_lk),np.repeat(np.arange(0,ns_rt),nedge_lk))),
        'scan':np.concatenate((np.repeat('T',nedge_lk*ns_rt),np.repeat('RT',nedge_lk*ns_rt))),
        'score':np.concatenate((msn_t_lk_vec,msn_rt_lk_vec)).flatten()}),
    targets='subject', raters='scan',ratings='score')['ICC'][2]

## local
if os.path.isfile(data_out_dir+'/msn_icc.npz'):  # if file exists, load it
    npz = np.load(data_out_dir+'/msn_icc.npz')
    msn_rt_hk_icc_edge = npz['msn_rt_hk_icc_edge']
    msn_rt_lk_icc_edge = npz['msn_rt_lk_icc_edge']
    del npz
else:                                           # else recreate values using code below

    # mmp h
    temp_hk_icc = np.zeros([nedge_hk])
    for i in range(nedge_hk):
        if i % 1000 == 0: print(i)
        temp_hk_icc[i] = pg.intraclass_corr(data=pd.DataFrame({
            'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
            'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
            'score':np.concatenate((msn_t_hk_vec[:,i],msn_rt_hk_vec[:,i]))}),
        targets='subject', raters='scan',ratings='score')['ICC'][2]
    # reshape edge values into matrix 
    msn_rt_hk_icc_edge = np.zeros([nr_hk,nr_hk])
    msn_rt_hk_icc_edge[np.triu_indices(nr_hk,1)] = temp_hk_icc
    if msn_rt_hk_icc_edge[nr_hk-1,0]==0: msn_rt_hk_icc_edge = msn_rt_hk_icc_edge + msn_rt_hk_icc_edge.T
    
    # mmp l
    temp_lk_icc = np.zeros([nedge_lk])
    for i in range(nedge_lk):
        if i % 100 == 0: print(i)
        temp_lk_icc[i] = pg.intraclass_corr(data=pd.DataFrame({
                'subject':np.concatenate((np.arange(0,ns_rt),np.arange(0,ns_rt))),
                'scan':np.concatenate((np.repeat('T',ns_rt),np.repeat('RT',ns_rt))),
                'score':np.concatenate((msn_t_lk_vec[:,i],msn_rt_lk_vec[:,i]))}),
            targets='subject', raters='scan',ratings='score')['ICC'][2]
    # reshape edge values into matrix
    msn_rt_lk_icc_edge = np.zeros([nr_lk,nr_lk])
    msn_rt_lk_icc_edge[np.triu_indices(nr_lk,1)] = temp_lk_icc
    if msn_rt_lk_icc_edge[nr_lk-1,0]==0: msn_rt_lk_icc_edge = msn_rt_lk_icc_edge + msn_rt_lk_icc_edge.T
    
    np.savez(data_out_dir+'/msn_icc.npz',msn_rt_hk_icc_edge = msn_rt_hk_icc_edge,msn_rt_lk_icc_edge = msn_rt_lk_icc_edge)

# plot MMP hk sorted by Yeo networks
plt.figure()
hm = sb.heatmap(msn_rt_hk_icc_edge[yeo_mmp_hk_ord,:][:,yeo_mmp_hk_ord],cmap='Reds',xticklabels=False,yticklabels=False,cbar_kws={'label': 'ICC'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
bb = np.array(hm.get_position())
ef.add_subnetwork_lines(hm,nr_yeo_hk,lw=0.5)                                            # add thin black lines
ef.add_subnetwork_colours(hm,bb,nr_yeo_hk,yeo_col,lw=5,alpha=1,solid_capstyle="butt")   # add  network colour lines
if save_fig_rt: plt.savefig(plot_dir_rt+'/msn_mmp_hk_icc_yeo.png',dpi=600,bbox_inches='tight')

# plot MMP hk sorted by hemispheres
plt.figure()
hm = sb.heatmap(msn_rt_hk_icc_edge,cmap='Reds',xticklabels=False,yticklabels=False,cbar_kws={'label': 'ICC'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_hk_lh,nr_hk], *hm.get_xlim()); hm.vlines([0,nr_hk_lh,nr_hk], *hm.get_ylim())
if save_fig_rt: plt.savefig(plot_dir_rt+'/msn_mmp_hk_icc.png',dpi=600,bbox_inches='tight')

# plot MMP lk
plt.figure()
hm = sb.heatmap(msn_rt_lk_icc_edge,cmap='Reds',xticklabels=False,yticklabels=False,cbar_kws={'label': 'ICC'},vmin=0,vmax=1)#,cbar_kws={"ticks":[0,1]})
cax = plt.gcf().axes[-1]; cax.tick_params(labelsize=hm_axs); hm.figure.axes[-1].yaxis.label.set_size(hm_lbs); hm.set_aspect('equal')
hm.hlines([0,nr_lk_lh,nr_lk], *hm.get_xlim()); hm.vlines([0,nr_lk_lh,nr_lk], *hm.get_ylim())
if save_fig_rt: plt.savefig(plot_dir_rt+'/msn_mmp_lk_icc.png',dpi=600,bbox_inches='tight')

# ### plot "connectome" visualisation - top positive and negative edges
# # mmp h - 0.003 = 0.3% edges
# nl.plotting.plot_connectome(bct.threshold_proportional(msn_rt_hk_icc_edge,0.003),node_coords=coords_h[mmp_hk,:],node_color=list(col_h[mmp_hk]),node_size=20,edge_cmap='Reds')
# if save_fig_rt: plt.savefig(plot_dir_rt+'/msn_mmp_hk_icc_net_hi.png',dpi=500,bbox_inches='tight')
# # mmp l - 0.1 = 10% edges
# nl.plotting.plot_connectome(bct.threshold_proportional(msn_rt_lk_icc_edge,0.1),node_coords=coords_l[mmp_lk,:],node_color=list(col_l[mmp_lk]),node_size=20,edge_cmap='Reds')
# if save_fig_rt: plt.savefig(plot_dir_rt+'/msn_mmp_lk_icc_net_hi.png',dpi=500,bbox_inches='tight')

# Median and percentile values of ICC
print(str(round(np.median(msn_rt_hk_icc_edge[np.triu_indices(nr_hk,1)]),2))+' ['+str(round(np.percentile(msn_rt_hk_icc_edge[np.triu_indices(nr_hk,1)],25),2))+','+str(round(np.percentile(msn_rt_hk_icc_edge[np.triu_indices(nr_hk,1)],75),2))+']')
print(str(round(np.median(msn_rt_lk_icc_edge[np.triu_indices(nr_lk,1)]),2))+' ['+str(round(np.percentile(msn_rt_lk_icc_edge[np.triu_indices(nr_lk,1)],25),2))+','+str(round(np.percentile(msn_rt_lk_icc_edge[np.triu_indices(nr_lk,1)],75),2))+']')
