#!/bin/bash
NETWORK_DIR=/home/ub/tensorflow_api_export/lpmask_1125_bk
PIPELINE_CONFIG=$NETWORK_DIR/pipeline.config
MODEL_DIR=$NETWORK_DIR/ckpt/saved_model
INFERENCE_GRAPH_DIR=$NETWORK_DIR/ckpt/saved_model
TFOD_API=/home/ub/su/models/research/object_detection
INPUT_TYPE=image_tensor
MODEL_PREFIX=ckpt-26



python $TFOD_API/export_tflite_ssd_graph.py \
    --pipeline_config_path=$PIPELINE_CONFIG \
    --trained_checkpoint_prefix=$MODEL_DIR/$MODEL_PREFIX \
    --output_directory=$INFERENCE_GRAPH_DIR \
    --add_postprocessing_op=true

