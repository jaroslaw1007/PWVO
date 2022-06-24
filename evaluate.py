import tensorflow as tf
import numpy as np

import sys
import time
import argparse
import os

from core.utils.tf_utils import tf_invert_intrinsic, compute_rigid_flow
from core.utils.tf_utils import z_normalize_data, z_unnormalize_data
from core.utils.viz_utils import visualize_unc, visualize

from core.utils.utils import create_directory, save_func

from core.net.VONet import VONet
from core.net.PWVO import PWVO

from core.metrics import Rotation_Metrics, Translation_Metrics, Depth_Metrics, Flow_Metrics

from core.selection_module import Selection_Module

def evaluate_PWVO(args, dataset, net, epochs_num):
    img_dir = os.path.join(args.eval_img_path, args.detail, str(epochs_num))
    data_dir = os.path.join(args.txt_path, args.detail, str(epochs_num))
    
    create_directory(img_dir)
    create_directory(data_dir)
    
    rotation_list = []
    r_pred_list = []
    r_gt_list = []
    translation_list = []
    t_pred_list = []
    t_gt_list = []
    
    depth_list = []
    motion_list = []
    
    count = 0
    
    for batch_idx, inputs in enumerate(dataset):
        (batch_total_motion, batch_camera_motion, batch_r, batch_t, 
         batch_depth_t0, batch_depth_t1, batch_K) = inputs
                
        if len(batch_depth_t0.get_shape().as_list()) == 3:
            batch_depth_t0 = tf.expand_dims(batch_depth_t0, axis=-1)
            batch_depth_t1 = tf.expand_dims(batch_depth_t1, axis=-1)
        
        batch_shape = batch_total_motion.get_shape().as_list()
        count += batch_shape[0]
                
        if batch_shape[0] != args.batch_size:
            tmp_m = tf.zeros(shape=[args.batch_size - batch_shape[0], args.height, args.width, 2])
            tmp_rt = tf.zeros(shape=[args.batch_size - batch_shape[0], 3])
            tmp_d = tf.zeros(shape=[args.batch_size - batch_shape[0], args.height, args.width, 1])
            tmp_k = tf.zeros(shape=[args.batch_size - batch_shape[0], 3, 3])
            
            batch_total_motion = tf.concat([batch_total_motion, tmp_m], axis=0)
            batch_camera_motion = tf.concat([batch_camera_motion, tmp_m], axis=0)
            batch_depth_t0 = tf.concat([batch_depth_t0, tmp_d], axis=0)
            batch_depth_t1 = tf.concat([batch_depth_t1, tmp_d], axis=0)
            batch_r = tf.concat([batch_r, tmp_rt], axis=0)
            batch_t = tf.concat([batch_t, tmp_rt], axis=0)
            batch_K = tf.concat([batch_K, tmp_k], axis=0)
            
        batch_K_inv = tf_invert_intrinsic(batch_K)
            
        batch_inv_depth_t0 = tf.math.reciprocal_no_nan(batch_depth_t0)
        batch_inv_depth_t1 = tf.math.reciprocal_no_nan(batch_depth_t1)
        
        if args.ego:
            batch_input_motion = z_normalize_data(batch_camera_motion, args.data_mean, args.data_std)
        else:
            batch_input_motion = z_normalize_data(batch_total_motion, args.data_mean, args.data_std)
            
        test_outputs = net(batch_input_motion, batch_inv_depth_t0, batch_inv_depth_t1, \
                           batch_K, training=False)
        
        batch_pixel_rotation = test_outputs[0]
        batch_pixel_translation = test_outputs[1]
        batch_pixel_rt = tf.concat([batch_pixel_rotation, batch_pixel_translation], axis=-1)
        
        batch_rt_unc = test_outputs[2]
        batch_r_unc, batch_t_unc = tf.split(batch_rt_unc, num_or_size_splits=2, axis=-1)
        
        batch_rotation, batch_translation = Selection_Module(batch_pixel_rt, batch_rt_unc, args.top_k, \
                                                             args.patch_size, args.selection)
        
        ego_recon, rt_depth_t1 = compute_rigid_flow(batch_depth_t0, batch_rotation, batch_translation, \
                                                    batch_K, batch_K_inv)

        rotation_metrics = Rotation_Metrics(batch_r, batch_rotation)
        translation_metrics = Translation_Metrics(batch_t, batch_translation)
        depth_metrics = Depth_Metrics(batch_depth_t1, rt_depth_t1)
        motion_metrics = Flow_Metrics(batch_camera_motion, ego_recon)
        
        r_pred_list.append(batch_rotation.numpy())
        t_pred_list.append(batch_translation.numpy())
        r_gt_list.append(batch_r.numpy())
        t_gt_list.append(batch_t.numpy())
        
        rotation_list.append(rotation_metrics.numpy())
        translation_list.append(translation_metrics.numpy())
        depth_list.append(tf.reduce_mean(depth_metrics, axis=[1, 2, 3]).numpy())
        motion_list.append(tf.reduce_mean(motion_metrics, axis=[1, 2, 3]).numpy())
        
        unc_rx, unc_ry, unc_rz = tf.split(tf.exp(batch_r_unc), num_or_size_splits=3, axis=-1)
        
        visualize_unc(batch_total_motion[:4], batch_camera_motion[:4], ego_recon[:4], \
                      unc_rx[:4], (batch_camera_motion - ego_recon)[:4], img_dir, count)
        
    rotation_array = np.asarray(rotation_list).reshape((-1, 3))[:count]
    translation_array = np.asarray(translation_list).reshape((-1, 3))[:count]
    motion_array = np.asarray(motion_list).reshape((-1, 1))[:count]
    depth_array = np.asarray(depth_list).reshape((-1, 1))[:count]
    r_array = np.asarray(r_pred_list).reshape((-1, 3))[:count]
    t_array = np.asarray(t_pred_list).reshape((-1, 3))[:count]
    r_gt_array = np.asarray(r_gt_list).reshape((-1, 3))[:count]
    t_gt_array = np.asarray(t_gt_list).reshape((-1, 3))[:count]
    
    print('\n')
    print('EPE {:.3f}'.format(np.mean(motion_array)))
    print('R Err {:.3f}'.format(np.mean(rotation_array)))
    print('T Err {:.3f}'.format(np.mean(translation_array)))
    
    save_func(os.path.join(data_dir, 'rotation.npy'), rotation_array)
    save_func(os.path.join(data_dir, 'translation.npy'), translation_array)
    save_func(os.path.join(data_dir, 'motion.npy'), motion_array)
    save_func(os.path.join(data_dir, 'depth.npy'), depth_array)
    save_func(os.path.join(data_dir, 'r_pred.npy'), r_array)
    save_func(os.path.join(data_dir, 't_pred.npy'), t_array)
    save_func(os.path.join(data_dir, 'r_gt.npy'), r_gt_array)
    save_func(os.path.join(data_dir, 't_gt.npy'), t_gt_array)

def evaluate_VONet(args, dataset, net, epochs_num):
    img_dir = os.path.join(args.eval_img_path, args.detail, str(epochs_num))
    data_dir = os.path.join(args.txt_path, args.detail, str(epochs_num))
    
    create_directory(img_dir)
    create_directory(data_dir)
    
    rotation_list = []
    r_pred_list = []
    r_gt_list = []
    translation_list = []
    t_pred_list = []
    t_gt_list = []
    
    depth_list = []
    motion_list = []
    
    count = 0
    
    for batch_idx, inputs in enumerate(dataset):
        (batch_total_motion, batch_camera_motion, batch_r, batch_t, 
         batch_depth_t0, batch_depth_t1, batch_K) = inputs
                
        if len(batch_depth_t0.get_shape().as_list()) == 3:
            batch_depth_t0 = tf.expand_dims(batch_depth_t0, axis=-1)
            batch_depth_t1 = tf.expand_dims(batch_depth_t1, axis=-1)
        
        batch_shape = batch_total_motion.get_shape().as_list()
        count += batch_shape[0]
                
        if batch_shape[0] != args.batch_size:
            tmp_m = tf.zeros(shape=[args.batch_size - batch_shape[0], args.height, args.width, 2])
            tmp_rt = tf.zeros(shape=[args.batch_size - batch_shape[0], 3])
            tmp_d = tf.zeros(shape=[args.batch_size - batch_shape[0], args.height, args.width, 1])
            tmp_k = tf.zeros(shape=[args.batch_size - batch_shape[0], 3, 3])
            
            batch_total_motion = tf.concat([batch_total_motion, tmp_m], axis=0)
            batch_camera_motion = tf.concat([batch_camera_motion, tmp_m], axis=0)
            batch_depth_t0 = tf.concat([batch_depth_t0, tmp_d], axis=0)
            batch_depth_t1 = tf.concat([batch_depth_t1, tmp_d], axis=0)
            batch_r = tf.concat([batch_r, tmp_rt], axis=0)
            batch_t = tf.concat([batch_t, tmp_rt], axis=0)
            batch_K = tf.concat([batch_K, tmp_k], axis=0)
            
        batch_K_inv = tf_invert_intrinsic(batch_K)
            
        batch_inv_depth_t0 = tf.math.reciprocal_no_nan(batch_depth_t0)
        batch_inv_depth_t1 = tf.math.reciprocal_no_nan(batch_depth_t1)
        
        if args.ego:
            batch_input_motion = z_normalize_data(batch_camera_motion, args.data_mean, args.data_std)
        else:
            batch_input_motion = z_normalize_data(batch_total_motion, args.data_mean, args.data_std)
            
        test_outputs = net(batch_input_motion, batch_inv_depth_t0, batch_inv_depth_t1, \
                           batch_K, training=False)
        
        batch_rotation = test_outputs[0]
        batch_translation = test_outputs[1]
        
        ego_recon, rt_depth_t1 = compute_rigid_flow(batch_depth_t0, batch_rotation, batch_translation, \
                                                    batch_K, batch_K_inv)

        rotation_metrics = Rotation_Metrics(batch_r, batch_rotation)
        translation_metrics = Translation_Metrics(batch_t, batch_translation)
        depth_metrics = Depth_Metrics(batch_depth_t1, rt_depth_t1)
        motion_metrics = Flow_Metrics(batch_camera_motion, ego_recon)
        
        r_pred_list.append(batch_rotation.numpy())
        t_pred_list.append(batch_translation.numpy())
        r_gt_list.append(batch_r.numpy())
        t_gt_list.append(batch_t.numpy())
        
        rotation_list.append(rotation_metrics.numpy())
        translation_list.append(translation_metrics.numpy())
        depth_list.append(tf.reduce_mean(depth_metrics, axis=[1, 2, 3]).numpy())
        motion_list.append(tf.reduce_mean(motion_metrics, axis=[1, 2, 3]).numpy())
        
        visualize(batch_total_motion[:4], batch_camera_motion[:4], ego_recon[:4], \
                 (batch_camera_motion - ego_recon)[:4], img_dir, count)
        
    rotation_array = np.asarray(rotation_list).reshape((-1, 3))[:count]
    translation_array = np.asarray(translation_list).reshape((-1, 3))[:count]
    motion_array = np.asarray(motion_list).reshape((-1, 1))[:count]
    depth_array = np.asarray(depth_list).reshape((-1, 1))[:count]
    r_array = np.asarray(r_pred_list).reshape((-1, 3))[:count]
    t_array = np.asarray(t_pred_list).reshape((-1, 3))[:count]
    r_gt_array = np.asarray(r_gt_list).reshape((-1, 3))[:count]
    t_gt_array = np.asarray(t_gt_list).reshape((-1, 3))[:count]
    
    print('\n')
    print('EPE {:.3f}'.format(np.mean(motion_array)))
    print('R Err {:.3f}'.format(np.mean(rotation_array)))
    print('T Err {:.3f}'.format(np.mean(translation_array)))
    
    save_func(os.path.join(data_dir, 'rotation.npy'), rotation_array)
    save_func(os.path.join(data_dir, 'translation.npy'), translation_array)
    save_func(os.path.join(data_dir, 'motion.npy'), motion_array)
    save_func(os.path.join(data_dir, 'depth.npy'), depth_array)
    save_func(os.path.join(data_dir, 'r_pred.npy'), r_array)
    save_func(os.path.join(data_dir, 't_pred.npy'), t_array)
    save_func(os.path.join(data_dir, 'r_gt.npy'), r_gt_array)
    save_func(os.path.join(data_dir, 't_gt.npy'), t_gt_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvaluationPhase')
    
    parser.add_argument('--height', type=int, default=128, help='Height of input')
    parser.add_argument('--width', type=int, default=256, help='Width of input')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=100000, help='Total samples of dataset')
    parser.add_argument('--training_sample', type=int, default=99000, help='Training samples of dataset')
    
    parser.add_argument('--detail', type=str, required=True, help='Save directory name')
    parser.add_argument('--weight_path', type=str, required=True, help='Weight checkpoint path')
    parser.add_argument('--txt_path', type=str, default='txt', help='Directory path of saving experiment txt file')
    
    parser.add_argument('--eval_data_path', type=str, required=True, help='Directory path of your input evaluation datas')
    parser.add_argument('--eval_img_path', type=str, default='visual_sintel', \
                        help='Directory path of saving evaluation visualization')
    
    parser.add_argument('--baseline', action='store_true', default=False, help='Model VONet or PWVO')
    parser.add_argument('--coord', action='store_true', default=False, help='Add coordinate layer or not')
    parser.add_argument('--ego', action='store_true', default=False, help='Input ego or total')
    
    parser.add_argument('--selection', type=str, choices=['mean_select', 'top_k_select', \
                                                          'patch_select', 'patch_soft_select', \
                                                          'soft_select'], help='Different types of Selection Module')
    
    parser.add_argument('--eval_dataset', type=str, choices=['sintel', 'tartan', 'custom'], \
                        help='Which dataset to evaluate')
    
    parser.add_argument('--data_mean', type=float, default=0.0)
    parser.add_argument('--data_std', type=float, default=200.0)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=5)
    
    parser.add_argument('--epoch_num', type=int)
    
    args = parser.parse_args()
    
    if args.eval_dataset == 'sintel':
        from core.dataset.sintel_dataset import SintelDataLoader
        
        dataloader = SintelDataLoader(args)
        test_dataset = dataloader.generate_dataset()
    elif args.eval_dataset == 'tartan':
        from core.dataset.tartan_dataset import TartanDataLoader
        
        dataloader = TartanDataLoader(args)
        test_dataset = dataloader.generate_dataset()
    else:
        from core.dataset.custom_dataset import CustomDataLoader
        
        dataloader = CustomDataLoader(args)
        test_dataset = dataloader.generate_test_dataset()
        
    if args.baseline:
        net = VONet(args)
        net.load_weights(args.weight_path)
        evaluate_PWVO(args, test_dataset, net, args.epoch_num)
    else:
        net = PWVO(args)
        net.load_weights(args.weight_path)
        evaluate_PWVO(args, test_dataset, net, args.epoch_num)