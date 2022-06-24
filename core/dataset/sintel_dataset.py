import tensorflow as tf

import cv2
import glob
import os

import numpy as np

class SintelDataLoader:
    def __init__(self, args):
        self.args = args
        
        data_root = args.eval_data_path
        
        self.ids_1 = ['{:06d}.npy'.format(i) for i in range(0, 124)]
        self.ids_2 = ['{:06d}.npy'.format(i) for i in range(147, 757)]
        self.ids_3 = ['{:06d}.npy'.format(i) for i in range(783, 1041)]
        
        self.ids = self.ids_1 + self.ids_2 + self.ids_3
        self.ids_number = len(self.ids)
        
        self.total_motion_path = [os.path.join(data_root, 'resize_total_motion', self.ids[i]) 
                                  for i in range(self.ids_number)]
        self.camera_motion_path = [os.path.join(data_root, 'resize_camera_motion', self.ids[i]) 
                                   for i in range(self.ids_number)]
        self.relative_camera_angle_path = [os.path.join(data_root, 'relative_camera_angle', self.ids[i]) 
                                           for i in range(self.ids_number)]
        self.relative_camera_translation_path = [os.path.join(data_root, 'relative_camera_translation', self.ids[i]) 
                                                 for i in range(self.ids_number)]
        self.depth_t0_path = [os.path.join(data_root, 'resize_depth_t0', self.ids[i]) 
                              for i in range(self.ids_number)]
        self.depth_t1_path = [os.path.join(data_root, 'resize_depth_t1', self.ids[i]) 
                              for i in range(self.ids_number)]
        self.camera_intrinsic_t0_path = [os.path.join(data_root, 'resize_camera_intrinsic_t0', self.ids[i]) 
                                         for i in range(self.ids_number)]
        
    def map_fn(self, tm_path, cm_path, rca_path, rct_path, dt0_path, dt1_path, ci_path):
        '''
        WrapFunction -> EagerTensor -> String
        https://blog.csdn.net/qq_27825451/article/details/105247211
        '''
        
        def read_path(path):
            path_name = path.numpy().decode()
            file = np.load(path_name)
            
            return file
        
        def read_file(path_1, path_2, path_3, path_4, path_5, path_6, path_7):
            file_1 = read_path(path_1)
            file_2 = read_path(path_2)
            file_3 = read_path(path_3)
            file_4 = read_path(path_4)
            file_5 = read_path(path_5)
            file_6 = read_path(path_6)
            file_7 = read_path(path_7)

            return file_1, file_2, file_3, file_4, file_5, file_6, file_7
        
        [total_motion, camera_motion, \
         relative_camera_angle, relative_camera_translation, \
         depth_t0, depth_t1, camera_intrinsic_t0] = tf.py_function( 
            read_file, \
            inp=[tm_path, cm_path, rca_path, rct_path, dt0_path, dt1_path, ci_path], \
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        )

        return total_motion, camera_motion, \
               relative_camera_angle, relative_camera_translation, \
               depth_t0, depth_t1, camera_intrinsic_t0
    
    def generate_dataset(self):
        flow_dataset = tf.data.Dataset.from_tensor_slices((self.total_motion_path[:self.ids_number], \
                                                           self.camera_motion_path[:self.ids_number], \
                                                           self.relative_camera_angle_path[:self.ids_number], \
                                                           self.relative_camera_translation_path[:self.ids_number], \
                                                           self.depth_t0_path[:self.ids_number], \
                                                           self.depth_t1_path[:self.ids_number], \
                                                           self.camera_intrinsic_t0_path[:self.ids_number]))
        flow_dataset = flow_dataset.map(lambda t, u, v, w, x, y, z: self.map_fn(t, u, v, w, x, y, z), \
                                        num_parallel_calls=100)
        flow_dataset = flow_dataset.batch(batch_size=self.args.batch_size, drop_remainder=False)
        flow_dataset = flow_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return flow_dataset
