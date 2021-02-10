#!/bin/bash
#
# Script to run processing scripts on EPImix, T1 and DWI contrasts, on (available) scans of participants from the ADAPTA, TINKER and COGNISCAN studies from King's College London.
#
# Frantisek Vasa (fdv247@gmail.com)

# set directories
main_dir=${HOME}/Data/epimix_data 			# main directory
mni_dir=${HOME}/Desktop/MNI 				# MNI template directory
script_dir=${HOME}/Desktop/scripts 			# script directory

# loop over participant folders in main directory
for subj_dir in ${main_dir}/*; do

	subj_dir=${main_dir}/${subj_nm}

	# run EPImix processing script on current subject
	if [ -d ${subj_dir}/epimix ]; then
		echo '-------------------'
		echo "$(basename $subj_dir) EPImix"
		echo '-------------------'
		bash ${script_dir}/process_epimix.sh $subj_dir $mni_dir $script_dir
	fi

	# run T1 processing script on current subject
	if [ -d ${subj_dir}/t1 ]; then
		echo '-------------------'
		echo "$(basename $subj_dir) T1"
		echo '-------------------'
		bash ${script_dir}/process_t1.sh $subj_dir $mni_dir $script_dir
	fi

done