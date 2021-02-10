#!/bin/bash
#
# Script to rapidly process a single T1 scan. (The pipeline is based on the outcome of an evaluation on the Mindboggle dataset, using script "compare_processing_mindboggle.py".)
#
# Frantisek Vasa (fdv247@gmail.com)

subj_dir=$1 		# directory containing the current subjects' data
mni_dir=$2			# MNI template directory
script_dir=$3		# script directory

subj_name=$(basename $subj_dir) 				# subject name
mni=${mni_dir}/mni_t1_asym_09c_2mm_res.nii.gz	# MNI template

if [ -d ${subj_dir}/t1 ]; then

	if [ ! -d ${subj_dir}/t1/processed ]; then mkdir ${subj_dir}/t1/processed; fi

	cwd=${subj_dir}/t1/processed

  	start_time=$SECONDS

	# downsample
	ResampleImageBySpacing 3 ${subj_dir}/t1/nifti/00??_*.nii.gz ${cwd}/t1_2mm.nii.gz 2 2 2 0

	# N4 bias correction
	N4BiasFieldCorrection -d 3 -s 4 -i ${cwd}/t1_2mm.nii.gz -o ${cwd}/t1_2mm_n4.nii.gz

	# runtime 1 - N4
	echo "$subj_name" >> ${cwd}/runtime_sec.txt # subj ID
	n4_time=$(($SECONDS - $start_time))
	echo "n4 $n4_time" >> ${cwd}/runtime_sec.txt

	### registration to MNI
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni} \
	-m ${cwd}/t1_2mm_n4.nii.gz \
	-t s \
	-o ${cwd}/t1_2mm_n4_mni2009casym_ \
	-n 12

	# runtime 2 - reg
	reg_time=$(($SECONDS - $start_time - $n4_time))
	echo "reg $reg_time" >> ${cwd}/runtime_sec.txt

	# generate QC image
	int_range=$(fslstats ${cwd}/t1_2mm_n4_mni2009casym_Warped.nii.gz -r)
	slices ${cwd}/t1_2mm_n4_mni2009casym_Warped.nii.gz -i $int_range -o ${cwd}/t1_2mm_n4_mni2009casym_QC.png

	### Jacobian
	## add affine component to warp
	#ComposeMultiTransform 3 ${cwd}/composite_inv_warp.nii.gz -R ${mni} -i ${cwd}/t1_2mm_n4_mni2009casym_0GenericAffine.mat ${cwd}/t1_2mm_n4_mni2009casym_1InverseWarp.nii.gz
	ComposeMultiTransform 3 ${cwd}/t1_composite_warp.nii.gz -R ${mni} ${cwd}/t1_2mm_n4_mni2009casym_1Warp.nii.gz ${cwd}/t1_2mm_n4_mni2009casym_0GenericAffine.mat

	# create log Jacobian
	#CreateJacobianDeterminantImage 3 ${cwd}/composite_inv_warp.nii.gz ${cwd}/t1_2mm_n4_mni2009casym_inv_logJacobian.nii.gz 1
	CreateJacobianDeterminantImage 3 ${cwd}/t1_composite_warp.nii.gz ${cwd}/t1_2mm_n4_mni2009casym_logJacobian.nii.gz 1

	# remove unnecessary files
	rm ${cwd}/t1_2mm.nii.gz
	rm ${cwd}/t1_2mm_n4.nii.gz
	rm ${cwd}/t1_2mm_n4_mni2009casym_1InverseWarp.nii.gz
	rm ${cwd}/t1_2mm_n4_mni2009casym_InverseWarped.nii.gz
	rm ${cwd}/t1_composite_warp.nii.gz

fi
