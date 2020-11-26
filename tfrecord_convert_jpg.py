import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

#filenames = ["test.tfrecord"]
#raw_dataset = tf.data.TFRecordDataset(filenames)
#for raw_record in raw_dataset.take(1):
#  example = tf.train.Example()
#  example.ParseFromString(raw_record.numpy())
  
raw_image_dataset = tf.data.TFRecordDataset(['test.tfrecord'])

# Create a dictionary describing the features.
image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

counter = 0
for image_features in tqdm(parsed_image_dataset):
  image_raw = image_features['image/encoded'].numpy()
  decoded = cv2.imdecode(np.frombuffer(image_raw, np.uint8), -1)
  img = Image.fromarray(decoded, 'YCbCr')
  img.save('image/' + str(counter) + '.jpg')
  print(str(counter) + " image saved")
  counter+=1
  

