#!/bin/bash
#
# Script to run the comparison of processing pipelines across subjects (t1_compare_pipelines.sh), using Mindboggle-101 data.
# For the best possible benchmark of processing run-time, close all other user processes before running this script.
#
# Frantisek Vasa (fdv247@gmail.com)

# set directories
mindboggle_dir=${HOME}/Data/Mindboggle/Mindboggle101_volumes 	# Mindboggle dataset directory
mni_dir=${HOME}/Desktop/MNI										# MNI template directory
script_dir=${HOME}/Desktop/scripts 								# script directory

# loop over dataset folders in Mindboggle directory
mindboggle_datasets=(Extra-18_volumes MMRR-21_volumes NKI-RS-22_volumes NKI-TRT-20_volumes OASIS-TRT-20_volumes) # array with folder names; can alternatively be defined using ($(ls -d ${mindboggle_dir}/*/))
for dataset in "${mindboggle_datasets[@]}"; do

	# loop over subjects in each dataset directory
	for subj_dir in ${mindboggle_dir}/${dataset}/*; do

		# echo current subject
		echo '-------------------'
		echo $(basename $subj_dir)
		echo '-------------------'

		# run script on current subject
		bash ${script_dir}/compare_processing_mindboggle.sh $subj_dir $mni_dir $script_dir

	done

done