#!/bin/bash
#
# Script to rapidly process a single EPImix scan. (The pipeline is based on the outcome of an evaluation of T1 pipelines on the Mindboggle dataset, using script "compare_processing_mindboggle.py".)
#
# Frantisek Vasa (fdv247@gmail.com)

subj_dir=$1 		# directory containing the current subjects' data
mni_dir=$2			# MNI template directory
script_dir=$3		# script directory

subj_name=$(basename $subj_dir) 				# subject name
mni=${mni_dir}/mni_t1_asym_09c_2mm_res.nii.gz	# MNI template

# loop over directories - including main directory ("epimix") and retest scan directory ("epimix_RT")
declare -a epi_dirs=("epimix" "epimix_RT")
for epi_dir in ${epi_dirs[@]}; do

if [ -d ${subj_dir}/${epi_dir} ]; then

	if [ ! -d ${subj_dir}/${epi_dir}/processed ]; then mkdir ${subj_dir}/${epi_dir}/processed; fi

	cwd=${subj_dir}/${epi_dir}/processed

  	start_time=$SECONDS

	# merge epimix files into a single 4D volume
	fslmerge -a ${cwd}/epimix.nii.gz ${subj_dir}/epimix/nifti/*.nii.gz

	# N4 bias correction of T1
	N4BiasFieldCorrection -d 3 -s 4 -i ${subj_dir}/epimix/nifti/*T1_FLAIR*nii.gz -o ${cwd}/epimix_t1_n4.nii.gz

	# runtime 1 - N4
	echo "$subj_name" >> ${cwd}/runtime_sec.txt # subj ID
	n4_time=$(($SECONDS - $start_time))
	echo "n4 $n4_time" >> ${cwd}/runtime_sec.txt

	### registration to MNI
	bash ${script_dir}/antsRegistrationSyNQuick.sh \
	-d 3 \
	-f ${mni} \
	-m ${cwd}/epimix_t1_n4.nii.gz \
	-t s \
	-o ${cwd}/epimix_t1_n4_mni2009casym_ \
	-n 12

	# Apply registration estimated above to register epimix_GM to MNI
	antsApplyTransforms \
	-d 3 \
	-e 3 \
	-i ${cwd}/epimix.nii.gz \
	-o ${cwd}/epimix_mni2009casym.nii.gz \
	-r ${mni} \
	-t ${cwd}/epimix_t1_n4_mni2009casym_1Warp.nii.gz \
	-t ${cwd}/epimix_t1_n4_mni2009casym_0GenericAffine.mat

	# runtime 2 - reg
	reg_time=$(($SECONDS - $start_time - $n4_time))
	echo "reg $reg_time" >> ${cwd}/runtime_sec.txt

	# generate QC image
	int_range=$(fslstats ${cwd}/epimix_t1_n4_mni2009casym_Warped.nii.gz -r)
	slices ${cwd}/epimix_t1_n4_mni2009casym_Warped.nii.gz -i $int_range -o ${cwd}/epimix_t1_n4_mni2009casym_QC.png

	### Jacobian
	## add affine component to warp
	#ComposeMultiTransform 3 ${cwd}/composite_inv_warp.nii.gz -R ${mni} -i ${cwd}/epimix_t1_mni2009casym_0GenericAffine.mat ${cwd}/epimix_t1_mni2009casym_1InverseWarp.nii.gz
	ComposeMultiTransform 3 ${cwd}/epimix_composite_warp.nii.gz -R ${mni} ${cwd}/epimix_t1_n4_mni2009casym_1Warp.nii.gz ${cwd}/epimix_t1_n4_mni2009casym_0GenericAffine.mat

	# create log Jacobian
	#CreateJacobianDeterminantImage 3 ${cwd}/composite_inv_warp.nii.gz ${cwd}/epimix_t1_mni2009casym_inv_logJacobian.nii.gz 1 0
	CreateJacobianDeterminantImage 3 ${cwd}/epimix_composite_warp.nii.gz ${cwd}/epimix_t1_n4_mni2009casym_logJacobian.nii.gz 1 0

	# remove unnecessary files
	rm ${cwd}/epimix_t1_n4.nii.gz
	rm ${cwd}/epimix_t1_n4_mni2009casym_1InverseWarp.nii.gz
	rm ${cwd}/epimix_t1_n4_mni2009casym_InverseWarped.nii.gz
	rm ${cwd}/epimix_composite_warp.nii.gz

fi

done 	# for d in ${epi_dirs[@]}; do
