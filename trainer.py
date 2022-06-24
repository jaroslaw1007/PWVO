import tensorflow as tf
import numpy as np

from core.utils.tf_utils import z_normalize_data, z_unnormalize_data
from core.utils.tf_utils import tf_invert_intrinsic, compute_rigid_flow

from core.loss import Rotation_Loss, Pixelwise_Rotation_Loss, Pixelwise_Uncertainty_Rotation_Loss
from core.loss import Translation_L1_Loss, Pixelwise_Translation_Loss, Pixelwise_Uncertainty_Translation_Loss
from core.loss import Depth_Loss, Depth_Uncertainty_Loss
from core.loss import Flow_Loss, Flow_Uncertainty_Loss

from core.metrics import Rotation_Metrics, Translation_Metrics, Depth_Metrics, Flow_Metrics

from core.selection_module import Selection_Module

class Trainer:
    def __init__(self, args):
        self.args = args
    
    @tf.function
    def train_PWVO_step(self, net, optimizer, inputs):
        (batch_total_motion, batch_camera_motion, \
        batch_r, batch_t, \
        batch_depth_t0, batch_depth_t1, batch_K) = inputs
        
        batch_shape = batch_total_motion.get_shape().as_list()
    
        batch_inv_depth_t0 = tf.math.reciprocal_no_nan(batch_depth_t0)
        batch_inv_depth_t1 = tf.math.reciprocal_no_nan(batch_depth_t1)

        batch_normal_total_motion = z_normalize_data(batch_total_motion, self.args.data_mean, self.args.data_std)
        batch_normal_camera_motion = z_normalize_data(batch_camera_motion, self.args.data_mean, self.args.data_std)
        batch_K_inv = tf_invert_intrinsic(batch_K)

        with tf.GradientTape() as tape:
            if self.args.ego:
                batch_outputs = net(batch_normal_camera_motion, batch_inv_depth_t0, batch_inv_depth_t1, batch_K, training=True)
            else:
                batch_outputs = net(batch_normal_total_motion, batch_inv_depth_t0, batch_inv_depth_t1, batch_K, training=True)

            batch_pixel_rotation = batch_outputs[0]
            batch_pixel_translation = batch_outputs[1]

            batch_rt_unc = batch_outputs[2]
            batch_r_unc, batch_t_unc = tf.split(batch_rt_unc, num_or_size_splits=2, axis=-1)
            batch_pixel_rt = tf.concat([batch_pixel_rotation, batch_pixel_translation], axis=-1)

            batch_rotation, batch_translation = Selection_Module(batch_pixel_rt, batch_rt_unc, self.args.top_k, \
                                                                 self.args.patch_size, self.args.selection)
            
            ego_recon, rt_depth_t1 = compute_rigid_flow(batch_depth_t0, batch_rotation, batch_translation, \
                                                        batch_K, batch_K_inv)

            if self.args.rt_unc:
                rot_loss = Pixelwise_Uncertainty_Rotation_Loss(batch_r, batch_pixel_rotation, batch_r_unc)
                trans_loss = Pixelwise_Uncertainty_Translation_Loss(batch_t, batch_pixel_translation, batch_t_unc)
            else:
                rot_loss = Pixelwise_Rotation_Loss(batch_r, batch_pixel_rotation)
                trans_loss = Pixelwise_Translation_Loss(batch_t, batch_pixel_translation)

            rt_inv_depth_t1 = tf.math.reciprocal_no_nan(rt_depth_t1)

            if self.args.d_unc:
                depth_loss = Depth_Uncertainty_Loss(batch_inv_depth_t1, rt_inv_depth_t1, 
                                                    batch_r_unc, batch_t_unc, self.args.dunc_scale)
            else:
                depth_loss = Depth_Loss(batch_inv_depth_t1, rt_inv_depth_t1)

            if self.args.f_unc:
                flow_loss = Flow_Uncertainty_Loss(batch_camera_motion, ego_recon, batch_r_unc, batch_t_unc, \
                                                      self.args.flow_delta, self.args.func_scale)
            else:
                flow_loss = Flow_Loss(batch_camera_motion, ego_recon, self.args.flow_delta)

            r_loss = rot_loss * self.args.r_weight
            t_loss = trans_loss * self.args.t_weight
            d_loss = depth_loss * self.args.d_weight
            f_loss = flow_loss * self.args.f_weight

            loss = 0

            if not self.args.no_extrinsic:
                loss += r_loss + t_loss
                
            if self.args.depth:
                loss += d_loss

            if self.args.ego_motion:
                loss += f_loss
                
        rotation_metrics = Rotation_Metrics(batch_r, batch_rotation)
        translation_metrics = Translation_Metrics(batch_t, batch_translation)
        depth_metrics = Depth_Metrics(batch_depth_t1, rt_depth_t1)
        flow_metrics = Flow_Metrics(batch_camera_motion, ego_recon)

        var_list = net.trainable_variables
        gradients = tape.gradient(loss, var_list)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        optimizer.apply_gradients(zip(gradients, var_list))

        return {
            'total_loss': loss,
            'rotation_loss': r_loss,
            'translation_loss': t_loss,
            'depth_loss': d_loss,
            'flow_loss': f_loss,
            'r_metrics': tf.reduce_mean(rotation_metrics),
            't_metrics': tf.reduce_mean(translation_metrics),
            'depth_metrics': tf.reduce_mean(depth_metrics),
            'flow_metrics': tf.reduce_mean(flow_metrics),
            'ego_motion_recon': ego_recon,
            'r_pred': batch_rotation,
            't_pred': batch_translation,
            'uncertainty_map': tf.exp(batch_rt_unc),
            'flow_error_map': flow_metrics
        }
    
    @tf.function
    def train_VONet_step(self, net, optimizer, inputs):
        (batch_total_motion, batch_camera_motion, \
        batch_r, batch_t, \
        batch_depth_t0, batch_depth_t1, batch_K) = inputs
        
        batch_shape = batch_total_motion.get_shape().as_list()
    
        batch_inv_depth_t0 = tf.math.reciprocal_no_nan(batch_depth_t0)
        batch_inv_depth_t1 = tf.math.reciprocal_no_nan(batch_depth_t1)

        batch_normal_total_motion = z_normalize_data(batch_total_motion, self.args.data_mean, self.args.data_std)
        batch_normal_camera_motion = z_normalize_data(batch_camera_motion, self.args.data_mean, self.args.data_std)
        batch_K_inv = tf_invert_intrinsic(batch_K)

        with tf.GradientTape() as tape:
            if self.args.ego:
                batch_outputs = net(batch_normal_camera_motion, batch_inv_depth_t0, batch_inv_depth_t1, batch_K, training=True)
            else:
                batch_outputs = net(batch_normal_total_motion, batch_inv_depth_t0, batch_inv_depth_t1, batch_K, training=True)

            batch_rotation = batch_outputs[0]
            batch_translation = batch_outputs[1]

            ego_recon, rt_depth_t1 = compute_rigid_flow(batch_depth_t0, batch_rotation, batch_translation, \
                                                        batch_K, batch_K_inv)

            rot_loss = Rotation_Loss(batch_r, batch_rotation)
            trans_loss = Translation_L1_Loss(batch_t, batch_translation)

            rt_inv_depth_t1 = tf.math.reciprocal_no_nan(rt_depth_t1)

            depth_loss = Depth_Loss(batch_inv_depth_t1, rt_inv_depth_t1)

            flow_loss = Flow_Loss(batch_camera_motion, ego_recon, self.args.flow_delta)

            r_loss = rot_loss * self.args.r_weight
            t_loss = trans_loss * self.args.t_weight
            d_loss = depth_loss * self.args.d_weight
            f_loss = flow_loss * self.args.f_weight
            
            loss = 0
            
            if not self.args.no_extrinsic:
                loss += r_loss + t_loss
                
            if self.args.depth:
                loss += d_loss

            if self.args.ego_motion:
                loss += f_loss
                
        rotation_metrics = Rotation_Metrics(batch_r, batch_rotation)
        translation_metrics = Translation_Metrics(batch_t, batch_translation)
        depth_metrics = Depth_Metrics(batch_depth_t1, rt_depth_t1)
        flow_metrics = Flow_Metrics(batch_camera_motion, ego_recon)

        var_list = net.trainable_variables
        gradients = tape.gradient(loss, var_list)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        optimizer.apply_gradients(zip(gradients, var_list))

        return {
            'total_loss': loss,
            'rotation_loss': rot_loss,
            'translation_loss': trans_loss,
            'depth_loss': depth_loss,
            'flow_loss': f_loss,
            'r_metrics': tf.reduce_mean(rotation_metrics),
            't_metrics': tf.reduce_mean(translation_metrics),
            'depth_metrics': tf.reduce_mean(depth_metrics),
            'flow_metrics': tf.reduce_mean(flow_metrics),
            'ego_motion_recon': ego_recon,
            'r_pred': batch_rotation,
            't_pred': batch_translation,
            'flow_error_map': flow_metrics
        }