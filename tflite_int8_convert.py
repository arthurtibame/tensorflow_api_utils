#!/bin/python3
import argparse
import tensorflow as tf


def convert_int8(PATH):
    converter = tf.lite.TFLiteConverter.from_saved_model(PATH)
    #converter.optimizations = [tf.lite.Optimize.Default]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    tflite_model_quant_file = PATH + "/detect_int8.tflite"
    tflite_model_quant_file.write_bytes(tflite_model_quant)
    #tflite_quant_model = converter.convert()
    #open(PATH + "/detect.tflite").write(tflite_quant_model)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='yolov5s.pt', help='saved model path')
    opt = parser.parse_args()

    convert_int8(opt.path)
