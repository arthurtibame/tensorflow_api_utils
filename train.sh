#!/bin/bash

DATASET_PATH=$1
RATIO=$2
MODEL=$3
PIPELINE_CONFIG_PATH=$(pwd)/export_dataset/pipeline.config
TRAINED_CHECKPOINT_DIR=$(pwd)/export_dataset/ckpt/
OUTPUT_DIRECTORY=${TRAINED_CHECKPOINT_DIR}
TRAINING_ENV=$(pwd)/venv/bin/activate
TFLITE_CONVERT_ENV=$(pwd)/tflite_export/bin/activate


echo "activated training env"
source ${TRAINING_ENV}

python merge_tfrecord.py --path="${DATASET_PATH}" --ratio="${RATIO}" --model="${MODEL}" --batch_size=16
cd ./models/research
echo "Start Training"
python object_detection/model_main_tf2.py --pipeline_config_path=/home/ub/su/export_dataset/pipeline.config --model_dir=/home/ub/su/export_dataset/ckpt --alsologtostderr
#python object_detection/export_tflite_graph_tf2.py --pipeline_config_path=$PIPELINE_CONFIG_PATH --trained_checkpoint_dir=$TRAINED_CHECKPOINT_DIR  --output_directory=$OUTPUT_DIRECTORY

#echo "Switching to tf-nightly environment"
#deactivate
#echo "activated tf-nightly"
#source ${TFLITE_CONVERT_ENV}
#tflite_convert --saved_model_dir=$OUTPUT_DIRECTORY/saved_model --output_file=$OUTPUT_DIRECTORY/detect.tflite

#cd ../../
