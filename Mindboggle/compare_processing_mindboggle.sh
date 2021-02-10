#!/bin/bash
#
# Script to compare processing pipelines in terms of speed and registration quality, using Mindboggle-101 data.
# The Mindboggle-101 dataset contains (amongst other resources) 101 T1 brain scans along with manual labels of the Desikan-Killiany-Tourville atlas, in both native and MNI space.
# For futher details on the Mindboggle project, see: https://mindboggle.info/index.html. 
# Data is available from: https://osf.io/9ahyp/ (Mindboggle101_individuals/Mindboggle101_release3)
# The 1mm MNI (152, 6th gen) template is available alongside the dataset above (and/or as part of FSL); the 2mm and 3mm versions were downsampled using the ANTs command ResampleImageBySpacing; for the 2mm version:
# ResampleImageBySpacing 3 [MNI_1mm_file] [MNI_2mm_file] 2 2 2 0 0 0
#
# Script outputs are saved in a folder called "compare_pipelines" in each subjects' (Mindboggle data) directory.
#
# Frantisek Vasa (fdv247@gmail.com)

# Inputs
subj_dir=$1 		# directory containing the current subjects' data
mni_dir=$2			# MNI template directory
script_dir=$3		# script directory

# MNI templates (with skull)
mni_1mm=${mni_dir}/MNI152_T1_1mm.nii.gz
mni_2mm=${mni_dir}/MNI152_T1_2mm.nii.gz
mni_3mm=${mni_dir}/MNI152_T1_3mm.nii.gz

# MNI templates (skull stripped & mask)
mni_2mm_brain=${mni_dir}/MNI152_T1_2mm_brain.nii.gz 	
#mni_2mm_mask=${mni_dir}/MNI152_T1_2mm_mask_dil.nii.gz	# brain mask, created by binarizing and dilating the 2mm brain template above (fslmaths -bin -dilM -bin)

# does T1 file exist?
if [ -f ${subj_dir}/t1weighted.nii.gz ]; then

raw_t1=${subj_dir}/t1weighted.nii.gz 					# current 1mm T1 scan

# create output directory and make it current working directory (cwd)
if [ ! -d ${subj_dir}/compare_pipelines ]; then mkdir ${subj_dir}/compare_pipelines; fi
cwd_subj=${subj_dir}/compare_pipelines

# subject DKT atlas at different resolutions (both in native and MNI spaces)
# 2mm
ResampleImageBySpacing 3 ${subj_dir}/labels.DKT31.manual.nii.gz ${cwd_subj}/labels.DKT31.manual_2mm.nii.gz 2 2 2 0 0 1					# native space
ResampleImageBySpacing 3 ${subj_dir}/labels.DKT31.manual.MNI152.nii.gz ${cwd_subj}/labels.DKT31.manual.MNI152_2mm.nii.gz 2 2 2 0 0 1	# MNI space
# 3mm
ResampleImageBySpacing 3 ${subj_dir}/labels.DKT31.manual.nii.gz ${cwd_subj}/labels.DKT31.manual_3mm.nii.gz 3 3 3 0 0 1					# native space
ResampleImageBySpacing 3 ${subj_dir}/labels.DKT31.manual.MNI152.nii.gz ${cwd_subj}/labels.DKT31.manual.MNI152_3mm.nii.gz 3 3 3 0 0 1	# MNI space

##### 1) 	resolution (downsampling)
### 1.1) 	1mm
if [ ! -f ${cwd_subj}/t1_1mm_mni_Warped.nii.gz ]; then # only run if final output file does not exist

	# output & current working directory
	mkdir ${cwd_subj}/1mm
	cwd=${cwd_subj}/1mm

	# Calculate transform to MNI using T1 scan
	start_time=$SECONDS 															# store current (starting) time
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni_1mm} \
	-m ${raw_t1} \
	-t s \
	-o ${cwd}/t1_1mm_mni_ \
	-n 12
	echo "SyN $(($SECONDS - $start_time))" >> ${cwd_subj}/1mm_runtime_sec.txt 		# store registration run-time

	# generate QC image
	int_range=$(fslstats ${cwd}/t1_1mm_mni_Warped.nii.gz -r)						# store intensity range
	slices ${cwd}/t1_1mm_mni_Warped.nii.gz -i $int_range -o ${cwd_subj}/1mm_QC.png 	# save QC png

	# Apply registration estimated above to register the DKT31 atlas to MNI
	antsApplyTransforms \
	-d 3 \
	-e 3 \
	-n NearestNeighbor \
	-i ${subj_dir}/labels.DKT31.manual.nii.gz \
	-o ${cwd_subj}/labels.DKT31_1mm.nii.gz \
	-r ${mni_1mm} \
	-t ${cwd}/t1_1mm_mni_1Warp.nii.gz \
	-t ${cwd}/t1_1mm_mni_0GenericAffine.mat

	mv ${cwd}/t1_1mm_mni_Warped.nii.gz ${cwd_subj} 	# move Warped file
	rm -r ${cwd} 									# delete extra files

fi

### 1.2) 	2mm
if [ ! -f ${cwd_subj}/t1_2mm_mni_Warped.nii.gz ]; then # only run if final output file does not exist

	# output & current working directory
	mkdir ${cwd_subj}/2mm
	cwd=${cwd_subj}/2mm

	# resample T1 scan to 2mm
	ResampleImageBySpacing 3 ${raw_t1} ${cwd}/t1_2mm.nii.gz 2 2 2 0 				

	# Calculate transform to MNI using T1 scan
	start_time=$SECONDS																# store current (starting) time
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni_2mm} \
	-m ${cwd}/t1_2mm.nii.gz \
	-t s \
	-o ${cwd}/t1_2mm_mni_ \
	-n 12
	echo "SyN $(($SECONDS - $start_time))" >> ${cwd_subj}/2mm_runtime_sec.txt 		# store registration run-time	

	# generate QC image
	int_range=$(fslstats ${cwd}/t1_2mm_mni_Warped.nii.gz -r) 						# store intensity range
	slices ${cwd}/t1_2mm_mni_Warped.nii.gz -i $int_range -o ${cwd_subj}/2mm_QC.png 	# save QC png

	# Apply registration estimated above to register the DKT31 atlas to MNI
	antsApplyTransforms \
	-d 3 \
	-e 3 \
	-n NearestNeighbor \
	-i ${cwd_subj}/labels.DKT31.manual_2mm.nii.gz \
	-o ${cwd_subj}/labels.DKT31_2mm.nii.gz \
	-r ${mni_2mm} \
	-t ${cwd}/t1_2mm_mni_1Warp.nii.gz \
	-t ${cwd}/t1_2mm_mni_0GenericAffine.mat

	mv ${cwd}/t1_2mm_mni_Warped.nii.gz ${cwd_subj} 	# move Warped file
	rm -r ${cwd} 									# delete extra files

fi

### 1.3) 	3mm
if [ ! -f ${cwd_subj}/t1_3mm_mni_Warped.nii.gz ]; then # only run if final output file does not exist

	# output & current working directory
	mkdir ${cwd_subj}/3mm
	cwd=${cwd_subj}/3mm

	# resample T1 scan to 3mm
	ResampleImageBySpacing 3 ${raw_t1} ${cwd}/t1_3mm.nii.gz 3 3 3 0 				

	# Calculate transform to MNI using T1 scan
	start_time=$SECONDS	 															# store current (starting) time
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni_3mm} \
	-m ${cwd}/t1_3mm.nii.gz \
	-t s \
	-o ${cwd}/t1_3mm_mni_ \
	-n 12
	echo "SyN $(($SECONDS - $start_time))" >> ${cwd_subj}/3mm_runtime_sec.txt 		# store registration run-time	

	# generate QC image
	int_range=$(fslstats ${cwd}/t1_3mm_mni_Warped.nii.gz -r) 						# store intensity range
	slices ${cwd}/t1_3mm_mni_Warped.nii.gz -i $int_range -o ${cwd_subj}/3mm_QC.png 	# save QC png

	# Apply registration estimated above to register the DKT31 atlas to MNI
	antsApplyTransforms \
	-d 3 \
	-e 3 \
	-n NearestNeighbor \
	-i ${cwd_subj}/labels.DKT31.manual_3mm.nii.gz \
	-o ${cwd_subj}/labels.DKT31_3mm.nii.gz \
	-r ${mni_3mm} \
	-t ${cwd}/t1_3mm_mni_1Warp.nii.gz \
	-t ${cwd}/t1_3mm_mni_0GenericAffine.mat

	mv ${cwd}/t1_3mm_mni_Warped.nii.gz ${cwd_subj} 	# move Warped file
	rm -r ${cwd} 									# delete extra files

fi

##### 2) N4 bias field correction (at 2mm resolution)
if [ ! -f ${cwd_subj}/t1_2mm_n4_mni_Warped.nii.gz ]; then # only run if final output file does not exist

	# output & current working directory
	mkdir ${cwd_subj}/n4
	cwd=${cwd_subj}/n4

	# resample T1 scan to 2mm
	ResampleImageBySpacing 3 ${raw_t1} ${cwd}/t1_2mm.nii.gz 2 2 2 0

	# N4 bias correction
	start_time=$SECONDS	 																# store current (starting) time
	N4BiasFieldCorrection -d 3 -s 4 -i ${cwd}/t1_2mm.nii.gz -o ${cwd}/t1_2mm_n4.nii.gz	# N4 bias field correction
	echo "N4 $(($SECONDS - $start_time))" >> ${cwd_subj}/n4_runtime_sec.txt 			# store N4 run-time

	# Calculate transform to MNI using T1 scan
	start_time=$SECONDS	 																# store current (starting) time
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni_2mm} \
	-m ${cwd}/t1_2mm_n4.nii.gz \
	-t s \
	-o ${cwd}/t1_2mm_n4_mni_ \
	-n 12
	echo "SyN $(($SECONDS - $start_time))" >> ${cwd_subj}/n4_runtime_sec.txt 			# store registration run-time

	# generate QC image
	int_range=$(fslstats ${cwd}/t1_2mm_n4_mni_Warped.nii.gz -r) 							# store intensity range
	slices ${cwd}/t1_2mm_n4_mni_Warped.nii.gz -i $int_range -o ${cwd_subj}/2mm_n4_QC.png 	# save QC png

	# Apply registration estimated above to register the DKT31 atlas to MNI
	antsApplyTransforms \
	-d 3 \
	-e 3 \
	-n NearestNeighbor \
	-i ${cwd_subj}/labels.DKT31.manual_2mm.nii.gz \
	-o ${cwd_subj}/labels.DKT31_2mm_n4.nii.gz \
	-r ${mni_2mm} \
	-t ${cwd}/t1_2mm_n4_mni_1Warp.nii.gz \
	-t ${cwd}/t1_2mm_n4_mni_0GenericAffine.mat

	mv ${cwd}/t1_2mm_n4_mni_Warped.nii.gz ${cwd_subj} 	# move Warped file
	rm -r ${cwd} 										# delete extra files

fi

##### 3) mask / brain extraction (at 2mm resolution, with N4 bias field correction)
### 3.1) BET brain extraction
if [ ! -f ${cwd_subj}/t1_2mm_n4_bet_mni_Warped.nii.gz ]; then # only run if final output file does not exist

	# output & current working directory
	mkdir ${cwd_subj}/bet
	cwd=${cwd_subj}/bet

	# resample T1 scan to 2mm
	ResampleImageBySpacing 3 ${raw_t1} ${cwd}/t1_2mm.nii.gz 2 2 2 0

	# N4 bias correction
	start_time=$SECONDS 																# store current (starting) time
	N4BiasFieldCorrection -d 3 -s 4 -i ${cwd}/t1_2mm.nii.gz -o ${cwd}/t1_2mm_n4.nii.gz	# N4 bias field correction
	echo "N4 $(($SECONDS - $start_time))" >> ${cwd_subj}/bet_runtime_sec.txt 			# store N4 run-time
	
	# run FSL BET
	start_time=$SECONDS	 														# store current (starting) time
	bet ${cwd}/t1_2mm_n4.nii.gz ${cwd}/t1_2mm_n4_bet.nii.gz -f 0.4				# brain extraction
	echo "BET $(($SECONDS - $start_time))" >> ${cwd_subj}/bet_runtime_sec.txt 	# store BET run-time
	
	# Calculate transform to MNI using T1 scan
	start_time=$SECONDS															# store current (starting) time
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni_2mm_brain} \
	-m ${cwd}/t1_2mm_n4_bet.nii.gz \
	-t s \
	-o ${cwd}/t1_2mm_n4_bet_mni_ \
	-n 12
	echo "SyN $(($SECONDS - $start_time))" >> ${cwd_subj}/bet_runtime_sec.txt	# store registration run-time

	# generate QC image
	int_range=$(fslstats ${cwd}/t1_2mm_n4_bet_mni_Warped.nii.gz -r) 								# store intensity range
	slices ${cwd}/t1_2mm_n4_bet_mni_Warped.nii.gz -i $int_range -o ${cwd_subj}/2mm_n4_bet_QC.png 	# save QC png

	# Apply registration estimated above to register the DKT31 atlas to MNI
	antsApplyTransforms \
	-d 3 \
	-e 3 \
	-n NearestNeighbor \
	-i ${cwd_subj}/labels.DKT31.manual_2mm.nii.gz \
	-o ${cwd_subj}/labels.DKT31_2mm_n4_bet.nii.gz \
	-r ${mni_2mm_brain} \
	-t ${cwd}/t1_2mm_n4_bet_mni_1Warp.nii.gz \
	-t ${cwd}/t1_2mm_n4_bet_mni_0GenericAffine.mat

	mv ${cwd}/t1_2mm_n4_bet_mni_Warped.nii.gz ${cwd_subj} 	# move Warped file
	rm -r ${cwd} 											# delete extra files

fi

##### 4) b-spline SyN registration (at 2mm resolution, with N4 bias field correction)
if [ ! -f ${cwd_subj}/t1_2mm_n4_bsyn_mni_Warped.nii.gz ]; then # only run if final output file does not exist

	# output & current working directory
	mkdir ${cwd_subj}/bsyn
	cwd=${cwd_subj}/bsyn

	# resample T1 scan to 2mm
	ResampleImageBySpacing 3 ${raw_t1} ${cwd}/t1_2mm.nii.gz 2 2 2 0

	# N4 bias correction
	start_time=$SECONDS 																# store current (starting) time
	N4BiasFieldCorrection -d 3 -s 4 -i ${cwd}/t1_2mm.nii.gz -o ${cwd}/t1_2mm_n4.nii.gz 	# N4 bias field correction
	echo "N4 $(($SECONDS - $start_time))" >> ${cwd_subj}/bsyn_runtime_sec.txt 			# store N4 run-time

	# Calculate transform to MNI using T1 scan
	start_time=$SECONDS	 															# store current (starting) time
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni_2mm} \
	-m ${cwd}/t1_2mm_n4.nii.gz \
	-t b \
	-o ${cwd}/t1_2mm_n4_bsyn_mni_ \
	-n 12
	echo "b-SyN $(($SECONDS - $start_time))" >> ${cwd_subj}/bsyn_runtime_sec.txt	# store registration run-time
	
	# generate QC image
	int_range=$(fslstats ${cwd}/t1_2mm_n4_bsyn_mni_Warped.nii.gz -r) 								# store intensity range
	slices ${cwd}/t1_2mm_n4_bsyn_mni_Warped.nii.gz -i $int_range -o ${cwd_subj}/2mm_n4_bsyn_QC.png 	# save QC png

	# Apply registration estimated above to register the DKT31 atlas to MNI
	antsApplyTransforms \
	-d 3 \
	-e 3 \
	-n NearestNeighbor \
	-i ${cwd_subj}/labels.DKT31.manual_2mm.nii.gz \
	-o ${cwd_subj}/labels.DKT31_2mm_n4_bsyn.nii.gz \
	-r ${mni_2mm} \
	-t ${cwd}/t1_2mm_n4_bsyn_mni_1Warp.nii.gz \
	-t ${cwd}/t1_2mm_n4_bsyn_mni_0GenericAffine.mat

	mv ${cwd}/t1_2mm_n4_bsyn_mni_Warped.nii.gz ${cwd_subj} 	# move Warped file
	rm -r ${cwd} 											# delete extra files

fi

fi # does subject's T1 file exist?
