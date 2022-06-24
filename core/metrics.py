import tensorflow as tf
import numpy as np

def Flow_Metrics(gt, pred, max_flow=400):
    mag = tf.sqrt(tf.reduce_sum(gt ** 2, axis=-1))
    
    valid = mag < max_flow
    valid = tf.expand_dims(tf.cast(valid, tf.float32), axis=-1)
    l1_loss = tf.abs(pred - gt)
    
    flow_loss = (valid * l1_loss)
    
    return flow_loss

def Rotation_Metrics(gt, pred):
    err = tf.abs(gt - pred)
    
    return err

def Translation_Metrics(gt, pred):
    err = tf.abs(gt - pred)
    
    return err

def Depth_Metrics(gt, pred):
    err = tf.abs(gt - pred)
    
    return err

def Rotation_Pixelwise_Metrics(gt, pred):
    batch, height, width, _ = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1),axis=1)
    gt = tf.tile(gt, [1, height, width, 1])
    err = tf.abs(gt - pred)
    
    return err

def Translation_Pixelwise_Metrics(gt, pred):
    batch, height, width, _ = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1),axis=1)
    gt = tf.tile(gt, [1, height, width, 1])
    err = tf.abs(gt - pred)
    
    return err

def Rotation_Pixelwise_Uncertainty_Metrics(gt, pred, logvar):
    uncertainty = tf.reduce_mean(tf.exp(-logvar), axis=-1, keepdims=True)
    batch, height, width, _ = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1),axis=1)
    gt = tf.tile(gt, [1, height, width, 1])
    err = tf.abs(gt - pred*uncertainty)
    # err = tf.reduce_mean(err, axis=[1, 2, 3])
    
    return err

def Translation_Pixelwise_Uncertainty_Metrics(gt, pred, logvar):
    uncertainty = tf.reduce_mean(tf.exp(-logvar), axis=-1, keepdims=True)
    batch, height, width, _ = pred.get_shape().as_list()
    gt = tf.expand_dims(tf.expand_dims(gt, axis=1),axis=1)
    gt = tf.tile(gt, [1, height, width, 1])
    err = tf.abs(gt - pred*uncertainty)
    # err = tf.reduce_mean(err, axis=[1, 2, 3])
    
    return err

def Pixelwise_Uncertainty_Metrics(pred, logvar, split=False):
    if split:
        uncertainty = tf.exp(-logvar)
    else:
        uncertainty = tf.reduce_mean(tf.exp(-logvar), axis=-1, keepdims=True)
    weight = pred*uncertainty
    weight = tf.reduce_sum(weight, axis=[1, 2])/tf.reduce_sum(uncertainty, axis=[1, 2])
    # weight = tf.reduce_mean(pred, axis=[1, 2])
    return weight
