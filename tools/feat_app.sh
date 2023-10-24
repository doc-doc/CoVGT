#########################################################################
# File Name: feat_app.sh
# Author: Xiao Junbin
# mail: xiaojunbin@u.nus.edu
# Created Time: Sat 19 Sep 2020 09:22:26 PM
#########################################################################
#!/bin/bash
GPUID=$1
CUDA_VISIBLE_DEVICES=$GPUID python preprocess_features.py \
	--dataset 'nextqa' \
	--model 'resnet101' \
	--image_width 224 \
	--image_height 224
