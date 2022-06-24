import cv2
import glob
import random
import os
import argparse
import tqdm
import time

import tensorflow as tf
import numpy as np

from multiprocessing import Pool

from data_utils import save_func, intrinsic_inverse, pixel_coord_generation, create_motion
from data_utils import deg2mat_xyz, deg2mat_zyx
from convert_to_tfrecords import convert_npy_to_tfrecords

parser = argparse.ArgumentParser(description='DataGeneration')

parser.add_argument('--init_height', type=int, default=436, help='Original map height')
parser.add_argument('--init_width', type=int, default=1024, help='Original map width')
parser.add_argument('--height', type=int, default=128, help='Resize map height')
parser.add_argument('--width', type=int, default=256, help='Resize map width')
parser.add_argument('--parallel', action='store_true', default=False, help='Parallelly generate dataset')
parser.add_argument('--npy_save', action='store_true', default=False, help='Save npy file not tfrecords')
parser.add_argument('--dataset_size', type=int, default=100000, help='The dataset samples number you want to generate')
parser.add_argument('--mask_path', type=str, required=True, help='The object mask image save path')
parser.add_argument('--save_path', type=str, required=True, help='The directory you want to save dataset')
args = parser.parse_args()

mask_id = glob.glob(os.path.join(args.mask_path, '*.png'))

pixel_coord_t0 = pixel_coord_generation(args.height, args.width, args.height, args.width)
# flat_pixel_coord_t0 = np.reshape(np.transpose(pixel_coord_t0, (2, 0, 1)), (-1, args.height * args.width))

wh_scale = [(args.width - 1.) / (args.init_width - 1.), \
            (args.height - 1.) / (args.init_height - 1.)]

EPSILON = 1e-12
save_list = ['total_motion', 'camera_motion', 'total_scene_motion', 'camera_scene_motion', \
             'relative_camera_angle', 'relative_camera_translation', \
             'depth_t0', 'depth_t1', 'mask' ,'camera_intrinsic', 'camera_intrinsic_inv']

def random_generation(mean, std, power, prob, lower, upper, noise_scale=0.1): 
    if np.random.rand() > prob:
        random_val = np.random.normal(mean, noise_scale, size=(1))
        
        while random_val < lower or random_val > upper:
            random_val = np.random.normal(mean, noise_scale, size=(1))
        
        return random_val
    else:
        gamma = np.random.normal(mean, std, size=(1))
        xi = np.sign(gamma) * np.power(np.abs(gamma), power)
        
        while xi < lower or xi > upper:
            gamma = np.random.normal(mean, std, size=(1))
            xi = np.sign(gamma) * np.power(np.abs(gamma), power)

        return xi
    
def random_parameter(depth_min):
    random_angle = np.zeros((3))
    random_translation = np.zeros((3))
    
    random_angle[0] = random_generation(0.0, 1.5, 2.5, 0.15, -9.0, 9.0, 0.2)
    random_angle[1] = random_generation(0.0, 2.0, 2.0, 0.15, -9.0, 9.0, 0.2)
    # random_angle[2] = random_generation(0.0, 0.45, 2, 0.3, -1.5, 1.5, 0.1)
    random_angle[2] = random_generation(0.0, 1.5, 2.5, 0.15, -8.0, 8.0, 0.2)
    
    random_translation[0] = random_generation(0.0, 0.4, 2.5, 0.15, -0.7, 0.7, 0.03)
    random_translation[1] = random_generation(0.0, 0.25, 2, 0.2, -0.4, 0.4, 0.025)
    random_translation[2] = random_generation(0.0, 1.8, 2.5, 0.15, \
                                              np.maximum(-5.0, -0.2 * depth_min), 5.0, 0.02)

    return random_angle, random_translation

def random_motion_gen(K, m_id, height, width, focal_length, probability=0.4, mask_number=5):
    expansion_h = height // 4
    expansion_w = width // 4
    pad_h = height + 2 * expansion_h
    pad_w = width + 2 * expansion_w
    
    # Random generate background depth
    if np.random.rand() > 0.1:
        normal_bg_depth = random_generation(0.6, 0.2, 2, 0.5, 0.0, 1.0, 0.1)
        bg_depth = normal_bg_depth * 79.0 + 1.0
    else:
        bg_depth = np.random.uniform(low=80.0, high=3200.0, size=(1))
        
    bg_depth_expand = np.tile(bg_depth, [pad_h, pad_w, 1])
    random_depth_array = np.zeros((mask_number))
    
    # Random generate N object depth
    for i in range(mask_number):
        random_depth_array[i] = random_generation(0.6, 0.6, 1.5, 0.4, 0.0, 1.0, 0.2)
        random_depth_array[i] = random_depth_array[i] * (bg_depth - focal_length / 5200) + focal_length / 5200
    
    # Random generate camera R, T
    relative_camera_angle, relative_camera_translation = random_parameter(np.min(random_depth_array))
    relative_camera_rotation = deg2mat_xyz(relative_camera_angle)
    
    # obj: Global, object: Local
    obj_motion = np.zeros((pad_h, pad_w, 2))
    obj_scene_motion = np.zeros((pad_h, pad_w, 3))
    obj_depth_t0 = np.full((pad_h, pad_w, 1), 9999.)
    obj_pixel_coord_t0 = pixel_coord_generation(pad_h, pad_w, pad_h, pad_w)

    for i in range(mask_number):        
        random_depth_expand = np.tile(random_depth_array[i], [pad_h, pad_w, 1])
            
        # Read random chair mask
        random_chair_id = np.random.choice(len(m_id))
        
        # Random Scale
        random_scale = int(random_generation(0.55, 0.2, 1, 1, 0.25, 0.7, 0.0) * width)
        
        chair_mask = cv2.resize(cv2.imread(m_id[random_chair_id]), (random_scale, random_scale), \
                                interpolation=cv2.INTER_LINEAR)
        chair_mask = cv2.cvtColor(chair_mask, cv2.COLOR_BGR2GRAY) / 255.0
        
        # Horizontal flip or not
        if np.random.rand() > 0.5:
            chair_mask = chair_mask[:, ::-1]
        
        chair_mask = np.where(chair_mask > 0.2, 1.0, 0.0)
        chair_idx = np.where(chair_mask > 0.0)
        
        # Crop minimum chair bbox
        c_top = np.min(chair_idx[0])
        c_bottom = np.max(chair_idx[0])
        c_left = np.min(chair_idx[1])
        c_right = np.max(chair_idx[1])

        c_height = c_bottom - c_top
        c_width = c_right - c_left
        
        # Random top, left of bbox corner
        top = np.random.randint(0, pad_h - c_height)
        left = np.random.randint(0, pad_w - c_width)
        
        chair_bbox_mask = np.expand_dims(chair_mask[c_top: c_bottom, c_left: c_right], axis=2)
        
        random_object_depth_t0 = np.full((pad_h, pad_w, 1), 9999.)
        
        random_object_mask = np.zeros((pad_h, pad_w, 1))
        random_object_mask[top: top + c_height, left: left + c_width, :] += chair_bbox_mask
        
        # Deal with close, far object
        random_object_depth_t0 = random_object_mask * random_depth_expand + \
                                 (1. - random_object_mask) * random_object_depth_t0
        
        object_depth_t0_mask = np.where(random_object_depth_t0 < obj_depth_t0, 1.0, 0.0)
        obj_depth_t0 = object_depth_t0_mask * random_object_depth_t0 + (1. - object_depth_t0_mask) * obj_depth_t0
        
        # Fill number can transfer to instance motion segmentation
        # obj_mask += object_depth_mask
        # obj_mask = np.clip(obj_mask, 0.0, 1.0)
        
        random_object_motion = np.zeros((pad_h, pad_w, 2))
        random_object_scene_motion = np.zeros((pad_h, pad_w, 3))
        
        if np.random.rand() > probability:
            # Create random object motion
            residual_coord_w = np.full((c_height, c_width, 1), expansion_w)
            residual_coord_h = np.full((c_height, c_width, 1), expansion_h)
            residual_coord = np.concatenate([residual_coord_w, residual_coord_h, np.zeros((c_height, c_width, 1))], axis=-1)

            object_pixel_coord_t0 = obj_pixel_coord_t0[top: top + c_height, left: left + c_width, :] - residual_coord
            object_depth_t0 = random_depth_expand[top: top + c_height, left: left + c_width, :]                   
                
            # Random Object movement parameter (Same range with camera)
            random_object_angle, random_object_translation = random_parameter(random_depth_array[i])
            random_object_rotation = deg2mat_xyz(random_object_angle)

            object_scene_motion, object_motion = create_motion(K, K, random_object_rotation, random_object_translation, \
                                                               object_pixel_coord_t0, object_depth_t0)
            object_motion = object_motion * chair_bbox_mask
            object_scene_motion = object_scene_motion * chair_bbox_mask
            
            random_object_motion[top: top + c_height, left: left + c_width, :] = object_motion
            random_object_scene_motion[top: top + c_height, left: left + c_width, :] = object_scene_motion
            
        obj_motion = object_depth_t0_mask * random_object_motion + (1. - object_depth_t0_mask) * obj_motion
        # Just track the pixel appear in t0, do not consider close, far issue
        obj_scene_motion = object_depth_t0_mask * random_object_scene_motion + (1. - object_depth_t0_mask) * obj_scene_motion
    
    h_right = expansion_h + height
    w_right = expansion_w + width
    
    obj_mask_t0 = np.where(obj_depth_t0 < bg_depth_expand, 1.0, 0.0)
    depth_t0 = obj_depth_t0 * obj_mask_t0 + bg_depth_expand * (1. - obj_mask_t0)
    
    crop_depth_t0 = depth_t0[expansion_h: h_right, expansion_w: w_right, :]
    crop_obj_motion = obj_motion[expansion_h: h_right, expansion_w: w_right, :]
    crop_obj_scene_motion = obj_scene_motion[expansion_h: h_right, expansion_w: w_right, :]
    camera_scene_motion, camera_motion = create_motion(K, K, relative_camera_rotation, relative_camera_translation, \
                                                       pixel_coord_t0, crop_depth_t0)
    
    total_motion = camera_motion + crop_obj_motion
    total_scene_motion = camera_scene_motion + crop_obj_scene_motion
    depth_t1 = crop_depth_t0 + total_scene_motion[:, :, 2:]

    return total_motion, camera_motion, total_scene_motion, camera_scene_motion,\
           crop_depth_t0, depth_t1, obj_mask_t0, \
           relative_camera_angle, relative_camera_translation, crop_obj_motion

def motion_gen(i):
    np.random.seed() # multiprocessing require reset numpy random seed
    
    # Random focal length for camera intrinsic
    normal_focal_length = random_generation(0.03, 0.5, 2.0, 0.4, 0.0, 1.0, 0.08)
    focal_length = normal_focal_length[0] * (3200. - 576.) + 576.
    # focal_length_x = focal_length
    # focal_length_y = focal_length / wh_scale[0] * wh_scale[1]
        
    camera_intrinsic = np.array([
        focal_length, 0.0, (args.init_width - 1) / 2.,
        0.0, focal_length, (args.init_height - 1) / 2.,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    camera_intrinsic[0, :] *= wh_scale[0]
    camera_intrinsic[1, :] *= wh_scale[1]
    
    camera_intrinsic_inv = intrinsic_inverse(camera_intrinsic)
        
    # Camera and object motion random generation 
    (total_motion, camera_motion, total_scene_motion, camera_scene_motion, 
     depth_t0, depth_t1, mask, relative_camera_angle, 
     relative_camera_translation, object_motion) = random_motion_gen(camera_intrinsic, mask_id, \
                                                                     args.height, args.width, focal_length)
    
    ids = '{:06d}.npy'.format(i)
    
    if args.npy_save:
        save_func(os.path.join(args.save_path, 'total_motion', ids), total_motion)
        save_func(os.path.join(args.save_path, 'camera_motion', ids), camera_motion)
        save_func(os.path.join(args.save_path, 'total_scene_motion', ids), total_scene_motion)
        save_func(os.path.join(args.save_path, 'camera_scene_motion', ids), camera_scene_motion)
        save_func(os.path.join(args.save_path, 'relative_camera_angle', ids), relative_camera_angle)
        save_func(os.path.join(args.save_path, 'relative_camera_translation', ids), relative_camera_translation) 
        save_func(os.path.join(args.save_path, 'depth_t0', ids), depth_t0)
        save_func(os.path.join(args.save_path, 'depth_t1', ids), depth_t1)
        save_func(os.path.join(args.save_path, 'mask', ids), mask)
        save_func(os.path.join(args.save_path, 'camera_intrinsic', ids), camera_intrinsic)
        save_func(os.path.join(args.save_path, 'camera_intrinsic_inv', ids), camera_intrinsic_inv)
    else:
        convert_npy_to_tfrecords(total_motion, camera_motion, relative_camera_angle, relative_camera_translation, \
                                 depth_t0, depth_t1, camera_intrinsic, i, args.save_path)

if __name__ == '__main__':
    
    if args.npy_save:
        for name in save_list:
            if not os.path.exists(os.path.join(args.save_path, name)):
                os.makedirs(os.path.join(args.save_path, name))

    if args.parallel:
        pool = Pool(processes=64)
        
        result_list_tqdm = []
        
        for result in tqdm.tqdm(pool.imap_unordered(motion_gen, iterable=range(args.dataset_size)), total=args.dataset_size):
            result_list_tqdm.append(result)
    else:
        for idx in tqdm.tqdm(range(args.dataset_size)):
            motion_gen(idx)
