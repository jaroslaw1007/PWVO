import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from core.utils.utils import flow_to_image_v3

def visualize_unc(total_motion, camera_motion, camera_motion_pred, unc_map, flow_error, save_path, count):
    
    visual_tot_motion = total_motion.numpy()
    visual_cam_motion = camera_motion.numpy()
    visual_cam_motion_pred = camera_motion_pred.numpy()
    visual_obj_motion = visual_tot_motion - visual_cam_motion
    visual_obj_motion_pred = visual_tot_motion - visual_cam_motion_pred
    visual_unc_map = unc_map.numpy()
    visual_flow_error = tf.sqrt(tf.reduce_sum(flow_error ** 2, axis=-1)).numpy()
    
    col = 7
    
    plt.figure(figsize=(26, 10))
    
    for i in range(visual_tot_motion.shape[0]):
        plt.subplot(4, col, i * col + 1)
        plt.title('Input Motion')
        plt.imshow(flow_to_image_v3(visual_tot_motion[i]))
        
        plt.subplot(4, col, i * col + 2)
        plt.title('GT Ego Motion')
        plt.imshow(flow_to_image_v3(visual_cam_motion[i]))
        
        plt.subplot(4, col, i * col + 3)
        plt.title('GT Obj Motion')
        plt.imshow(flow_to_image_v3(visual_obj_motion[i]))
        
        plt.subplot(4, col, i * col + 4)
        plt.title('Ego Motion')
        plt.imshow(flow_to_image_v3(visual_cam_motion_pred[i]))
        
        plt.subplot(4, col, i * col + 5)
        plt.title('Obj Motion')
        plt.imshow(flow_to_image_v3(visual_obj_motion_pred[i]))
        
        plt.subplot(4, col, i * col + 6)
        plt.title('Uncertainty Map')
        plt.imshow(visual_unc_map[i], cmap="jet")
        
        plt.subplot(4, col, i * col + 7)
        plt.title('Uncertainty RZ')
        plt.imshow(visual_flow_error[i], cmap="jet")

    plt.savefig(save_path + '/{}.png'.format(count))
    plt.close()

def visualize(total_motion, camera_motion, camera_motion_pred, flow_error, save_path, count):
    
    visual_tot_motion = total_motion.numpy()
    visual_cam_motion = camera_motion.numpy()
    visual_cam_motion_pred = camera_motion_pred.numpy()
    visual_obj_motion = visual_tot_motion - visual_cam_motion
    visual_obj_motion_pred = visual_tot_motion - visual_cam_motion_pred
    visual_flow_error = tf.sqrt(tf.reduce_sum(flow_error ** 2, axis=-1)).numpy()
    
    col = 6
    
    plt.figure(figsize=(26, 10))
    
    for i in range(visual_tot_motion.shape[0]):
        plt.subplot(4, col, i * col + 1)
        plt.title('Input Motion')
        plt.imshow(flow_to_image_v3(visual_tot_motion[i]))
        
        plt.subplot(4, col, i * col + 2)
        plt.title('GT Ego Motion')
        plt.imshow(flow_to_image_v3(visual_cam_motion[i]))
        
        plt.subplot(4, col, i * col + 3)
        plt.title('GT Obj Motion')
        plt.imshow(flow_to_image_v3(visual_obj_motion[i]))
        
        plt.subplot(4, col, i * col + 4)
        plt.title('Ego Motion')
        plt.imshow(flow_to_image_v3(visual_cam_motion_pred[i]))
        
        plt.subplot(4, col, i * col + 5)
        plt.title('Obj Motion')
        plt.imshow(flow_to_image_v3(visual_obj_motion_pred[i]))
        
        plt.subplot(4, col, i * col + 6)
        plt.title('Uncertainty RZ')
        plt.imshow(visual_flow_error[i], cmap="jet")

    plt.savefig(save_path + '/{}.png'.format(count))
    plt.close()