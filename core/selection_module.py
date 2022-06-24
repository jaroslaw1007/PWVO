import tensorflow as tf
import numpy as np

from core.utils.tf_utils import top_k_pred, patch_top_1_pred, Softmax_Selection 

def Selection_Module(pixel_rt_pred, pixel_rt_unc, top_k=5, patch_size=32, selection='mean_select'):
    r_pred, t_pred = tf.split(pixel_rt_pred, num_or_size_splits=2, axis=-1)
    r_unc, t_unc = tf.split(pixel_rt_unc, num_or_size_splits=2, axis=-1)
    
    rx, ry, rz = tf.split(r_pred, num_or_size_splits=3, axis=-1)
    tx, ty, tz = tf.split(t_pred, num_or_size_splits=3, axis=-1)
    urx, ury, urz = tf.split(r_unc, num_or_size_splits=3, axis=-1)
    utx, uty, utz = tf.split(t_unc, num_or_size_splits=3, axis=-1)
    
    if selection == 'mean_select':
        batch_rotation = tf.reduce_mean(r_pred, axis=[1, 2])
        batch_translation = tf.reduce_mean(t_pred, axis=[1, 2])
    elif selection == 'top_k_select':
        batch_rx = top_k_pred(rx, urx, top_k)
        batch_ry = top_k_pred(ry, ury, top_k)
        batch_rz = top_k_pred(rz, urz, top_k)
        batch_tx = top_k_pred(tx, utx, top_k)
        batch_ty = top_k_pred(ty, uty, top_k)
        batch_tz = top_k_pred(tz, utz, top_k)

        batch_rotation = tf.concat([batch_rx, batch_ry, batch_rz], axis=-1)
        batch_translation = tf.concat([batch_tx, batch_ty, batch_tz], axis=-1)
    elif selection == 'patch_select':
        batch_rotation, batch_translation = patch_top_1_pred(pixel_rt_pred, pixel_rt_unc, patch_size, False)
    elif selection == 'patch_soft_select':
        batch_rotation, batch_translation = patch_top_1_pred(pixel_rt_pred, pixel_rt_unc, patch_size, True)
    else:
        batch_rx = Softmax_Selection(rx, urx)
        batch_ry = Softmax_Selection(ry, ury)
        batch_rz = Softmax_Selection(rz, urz)
        batch_tx = Softmax_Selection(tx, utx)
        batch_ty = Softmax_Selection(ty, uty)
        batch_tz = Softmax_Selection(tz, utz)

        batch_rotation = tf.concat([batch_rx, batch_ry, batch_rz], axis=-1)
        batch_translation = tf.concat([batch_tx, batch_ty, batch_tz], axis=-1)
        
    return batch_rotation, batch_translation