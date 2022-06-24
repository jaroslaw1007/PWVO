import tensorflow as tf
import numpy as np

def Safe_Euclidean_Norm(x, epsilon=1e-12, axis=None, keepdims=False):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) + epsilon)

def Translation_Loss(gt, pred):
    euc_pred = Safe_Euclidean_Norm(pred, axis=-1)
    euc_gt = Safe_Euclidean_Norm(gt, axis=-1)
    err = tf.math.l2_normalize(gt, axis=-1) - tf.math.l2_normalize(pred, axis=-1)

    loss_1 = tf.reduce_mean(tf.square(euc_gt - euc_pred))
    loss_2 = tf.reduce_mean(Safe_Euclidean_Norm(err, axis=-1))
    
    return loss_1 + loss_2

def Translation_L1_Loss(gt, pred):
    return tf.reduce_mean(tf.abs(gt - pred) + 1e-12)

def Rotation_Loss(gt, pred):
    return tf.reduce_mean(tf.abs(gt - pred) + 1e-12)

def Pixelwise_Translation_Loss(gt, pred):
    pred_shape = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1), axis=1)
    gt = tf.tile(gt, [1, pred_shape[1], pred_shape[2], 1])
    
    euc_pred = Safe_Euclidean_Norm(pred, axis=-1, keepdims=True)
    euc_gt = Safe_Euclidean_Norm(gt, axis=-1, keepdims=True)
    err = tf.math.l2_normalize(gt, axis=-1) - tf.math.l2_normalize(pred, axis=-1)
    
    loss_1 = tf.reduce_mean(tf.square(euc_gt - euc_pred))
    loss_2 = tf.reduce_mean(Safe_Euclidean_Norm(err, axis=-1, keepdims=True))
    
    return loss_1 + loss_2

def Pixelwise_Rotation_Loss(gt, pred):
    pred_shape = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1), axis=1)
    gt = tf.tile(gt, [1, pred_shape[1], pred_shape[2], 1])
    
    return tf.reduce_mean(tf.abs(gt - pred) + 1e-12)
    
def Pixelwise_Uncertainty_Translation_Loss(gt, pred, logvar):
    pred_shape = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1), axis=1)
    gt = tf.tile(gt, [1, pred_shape[1], pred_shape[2], 1])
    
    uncertainty = tf.exp(-logvar)
    
    euc_pred = Safe_Euclidean_Norm(pred, axis=-1, keepdims=True)
    euc_gt = Safe_Euclidean_Norm(gt, axis=-1, keepdims=True)
    err = tf.math.l2_normalize(gt, axis=-1) - tf.math.l2_normalize(pred, axis=-1)
    
    loss_1 = tf.reduce_mean(uncertainty * tf.square(euc_gt - euc_pred))
    loss_2 = tf.reduce_mean(uncertainty * Safe_Euclidean_Norm(err, axis=-1, keepdims=True))
    loss_3 = tf.reduce_mean(logvar)
    
    return (loss_1 + loss_2) / 2. + loss_3
    
def Pixelwise_Uncertainty_Rotation_Loss(gt, pred, logvar):
    pred_shape = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1), axis=1)
    gt = tf.tile(gt, [1, pred_shape[1], pred_shape[2], 1])
    
    r_loss = tf.abs(gt - pred) + 1e-12
    
    uncertainty = tf.exp(-logvar)
    loss_1 = tf.reduce_mean(uncertainty * r_loss)
    loss_2 = tf.reduce_mean(logvar)
    
    return (loss_1 + loss_2) / 2.

def Flow_Loss(gt, pred, flow_delta=20.0):
    error = tf.abs(gt - pred) + 1e-12

    valid = error < flow_delta
    valid = tf.cast(valid, tf.float32)
    
    flow_loss = tf.reduce_mean(valid * error)
    
    return flow_loss

def Flow_Uncertainty_Loss(gt, pred, r_logvar, t_logvar, flow_delta=20.0, scale_factor=1.0):
    rx_logvar, ry_logvar, rz_logvar = tf.split(r_logvar * scale_factor, num_or_size_splits=3, axis=-1) 
    tx_logvar, ty_logvar, tz_logvar = tf.split(t_logvar * scale_factor, num_or_size_splits=3, axis=-1)
    
    rx_uncertainty = tf.exp(-rx_logvar)
    ry_uncertainty = tf.exp(-ry_logvar)
    rz_uncertainty = tf.exp(-rz_logvar)
    tx_uncertainty = tf.exp(-tx_logvar)
    ty_uncertainty = tf.exp(-ty_logvar)
    tz_uncertainty = tf.exp(-tz_logvar)
    
    pred_fx, pred_fy = tf.split(pred, num_or_size_splits=2, axis=-1)
    gt_fx, gt_fy = tf.split(gt, num_or_size_splits=2, axis=-1)
    
    fx_loss = tf.abs(gt_fx - pred_fx) + 1e-12
    fy_loss = tf.abs(gt_fy - pred_fy) + 1e-12
    
    valid_x = fx_loss < flow_delta
    valid_y = fy_loss < flow_delta
    
    valid_x = tf.cast(valid_x, tf.float32)
    valid_y = tf.cast(valid_y, tf.float32)
    
    loss_rx_fx = tf.reduce_mean(valid_x * rx_uncertainty * fx_loss)
    loss_rx_fy = tf.reduce_mean(valid_y * rx_uncertainty * fy_loss)
    
    loss_ry_fx = tf.reduce_mean(valid_x * ry_uncertainty * fx_loss)
    loss_ry_fy = tf.reduce_mean(valid_y * ry_uncertainty * fy_loss)
    
    loss_rz_fx = tf.reduce_mean(valid_x * rz_uncertainty * fx_loss)
    loss_rz_fy = tf.reduce_mean(valid_y * rz_uncertainty * fy_loss)
    
    loss_tx_fx = tf.reduce_mean(valid_x * tx_uncertainty * fx_loss)
    
    loss_ty_fy = tf.reduce_mean(valid_y * ty_uncertainty * fy_loss)
    
    loss_tz_fx = tf.reduce_mean(valid_x * tz_uncertainty * fx_loss)
    loss_tz_fy = tf.reduce_mean(valid_y * tz_uncertainty * fy_loss)
    
    loss_rx_log = tf.reduce_mean(rx_logvar)
    loss_ry_log = tf.reduce_mean(ry_logvar)
    loss_rz_log = tf.reduce_mean(rz_logvar)
    loss_tx_log = tf.reduce_mean(tx_logvar)
    loss_ty_log = tf.reduce_mean(ty_logvar)
    loss_tz_log = tf.reduce_mean(tz_logvar)
    
    loss_1 = (loss_rx_fx + loss_rx_fy + loss_ry_fx + loss_ry_fy + loss_rz_fx + loss_rz_fy + 
              loss_tx_fx + loss_ty_fy + loss_tz_fx + loss_tz_fy) / 2.0
    loss_2 = (loss_tx_log + loss_ty_log) / 2.0 + loss_rx_log + loss_ry_log + loss_rz_log + loss_tz_log
    
    return loss_1 + loss_2

def Depth_Loss(gt, pred):
    return tf.reduce_mean(tf.abs(gt - pred) + 1e-12)

def Depth_Smooth_Loss(gt, pred):
    l1_loss = tf.abs(gt - pred) + 1e-12
    
    loss = tf.where(l1_loss < 1, 0.5 * tf.square(l1_loss), l1_loss - 0.5)
    
    return loss

def Depth_Uncertainty_Loss(gt, pred, r_logvar, t_logvar, scale_factor=1.0):
    rx_logvar = tf.slice(r_logvar * scale_factor, [0, 0, 0, 0], [-1, -1, -1, 1])
    ry_logvar = tf.slice(r_logvar * scale_factor, [0, 0, 0, 1], [-1, -1, -1, 1])
    tz_logvar = tf.slice(t_logvar * scale_factor, [0, 0, 0, 2], [-1, -1, -1, 1])
    
    rx_uncertainty = tf.exp(-rx_logvar)
    ry_uncertainty = tf.exp(-ry_logvar)
    tz_uncertainty = tf.exp(-tz_logvar)
    
    l1_loss = tf.abs(gt - pred) + 1e-12
    
    loss_rx = tf.reduce_mean(rx_uncertainty * l1_loss)
    loss_ry = tf.reduce_mean(ry_uncertainty * l1_loss)
    loss_tz = tf.reduce_mean(tz_uncertainty * l1_loss)
    
    loss_rx_log = tf.reduce_mean(rx_logvar)
    loss_ry_log = tf.reduce_mean(ry_logvar)
    loss_tz_log = tf.reduce_mean(tz_logvar)
    
    return (loss_rx + loss_rx_log + loss_ry + loss_ry_log + loss_tz + loss_tz_log) / 2.0