import tensorflow as tf
import numpy as np

import os

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))

def convert_npy_to_tfrecords(total_motion, camera_motion, relative_camera_angle, relative_camera_translation, 
                             depth_t0, depth_t1, camera_intrinsic, idx, save_path):
    ids = '{:06d}.npy'.format(idx)
    
    writer = tf.io.TFRecordWriter(os.path.join(save_path, '{:06d}.tfrecords'.format(idx)))

    feature = {
        'total_motion': _bytes_feature(serialize_array(total_motion.astype(np.float32))),
        'camera_motion': _bytes_feature(serialize_array(camera_motion.astype(np.float32))),
        'relative_camera_angle': _float_feature(relative_camera_angle.astype(np.float32).flatten()),
        'relative_camera_translation': _float_feature(relative_camera_translation.astype(np.float32).flatten()),
        'depth_t0': _bytes_feature(serialize_array(depth_t0.astype(np.float32))),
        'depth_t1': _bytes_feature(serialize_array(depth_t1.astype(np.float32))),
        'camera_intrinsic': _bytes_feature(serialize_array(camera_intrinsic.astype(np.float32)))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    
    writer.write(serialized)
    writer.close()