import tensorflow as tf
import os 

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("/home/ub/tensorflow_api_export/test/saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def preprocess(image,
                        height,
                        width,
                        central_fraction=0.875,
                        scope=None,
                        central_crop=True,
                        use_grayscale=False):
  with tf.name_scope("eval_image") as scope:
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if use_grayscale:
      image = tf.image.rgb_to_grayscale(image)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_crop and central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      #image = tf.image.resize_bilinear(image, [height, width],
      #                                 align_corners=False)
      image = tf.image.resize(image, [height, width], method='bilinear', antialias=True, preserve_aspect_ratio=True)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def representative_dataset_gen():
  for i in range(1,2272):
    #dir path where .jpg (image) files are
    image = tf.io.read_file(os.path.join("/home/ub/tensorflow_api_export/lpmask_1125_bk/image/", str(i)+".jpg"))
    image = tf.compat.v1.image.decode_jpeg(image)
 #   image = preprocess(image,640,640)

    yield [image]

converter.representative_dataset = representative_dataset_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_converter = True
#converter.experimental_new_quantizer = True
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]

converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8
tflite_quant_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
