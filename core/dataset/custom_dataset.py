import tensorflow as tf

import cv2
import glob
import os

import numpy as np

class CustomDataLoader:
    def __init__(self, args):
        self.args = args
        
        self.train_tfrecord_list = [os.path.join(args.data_path, '{:06d}.tfrecords'.format(i)) 
                                    for i in range(args.training_sample)]
        self.test_tfrecord_list = [os.path.join(args.data_path, '{:06d}.tfrecords'.format(i)) 
                                   for i in range(args.training_sample, args.dataset_size)]
        
    @tf.function
    def parse_tfr_element(self, element):
        data = {
            'total_motion': tf.io.FixedLenFeature([], tf.string),
            'camera_motion': tf.io.FixedLenFeature([], tf.string),
            'relative_camera_angle': tf.io.FixedLenFeature([3], tf.float32),
            'relative_camera_translation': tf.io.FixedLenFeature([3], tf.float32),
            'depth_t0': tf.io.FixedLenFeature([], tf.string),
            'depth_t1': tf.io.FixedLenFeature([], tf.string),
            'camera_intrinsic': tf.io.FixedLenFeature([], tf.string)
        }
        
        content = tf.io.parse_single_example(element, data)
                
        total_motion = tf.io.parse_tensor(content['total_motion'], out_type=tf.float32)
        camera_motion = tf.io.parse_tensor(content['camera_motion'], out_type=tf.float32)
        depth_t0 = tf.io.parse_tensor(content['depth_t0'], out_type=tf.float32)
        depth_t1 = tf.io.parse_tensor(content['depth_t1'], out_type=tf.float32)
        camera_intrinsic = tf.io.parse_tensor(content['camera_intrinsic'], out_type=tf.float32)
        
        return (total_motion, camera_motion, 
                content['relative_camera_angle'], content['relative_camera_translation'],
                depth_t0, depth_t1, camera_intrinsic)
        
    def generate_train_dataset(self):
        flow_dataset = tf.data.TFRecordDataset(self.train_tfrecord_list)
        flow_dataset = flow_dataset.map(lambda x: self.parse_tfr_element(x), num_parallel_calls=100)
        flow_dataset = flow_dataset.shuffle(buffer_size=100).batch(batch_size=self.args.batch_size, drop_remainder=True)
        flow_dataset = flow_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return flow_dataset
    
    def generate_test_dataset(self):
        flow_dataset = tf.data.TFRecordDataset(self.test_tfrecord_list)
        flow_dataset = flow_dataset.map(lambda x: self.parse_tfr_element(x), num_parallel_calls=100)
        flow_dataset = flow_dataset.batch(batch_size=self.args.batch_size, drop_remainder=False)
        flow_dataset = flow_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return flow_dataset
