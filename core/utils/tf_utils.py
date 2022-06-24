import tensorflow as tf
import numpy as np

def Safe_Softmax(x, axis):
    '''
    https://gist.github.com/raingo/a5808fe356b8da031837
    '''
    
    axis_max = tf.reduce_max(x, axis=axis, keepdims=True)
    exp_x = tf.exp(x - axis_max)
    denominator = tf.reduce_sum(exp_x, axis=axis, keepdims=True)
    softmax = exp_x / denominator
    
    return softmax
    
def tf_clip_value_preserve_gradient(x, clip_min, clip_max):
    clip_x = tf.clip_by_value(x, clip_min, clip_max)
    
    return x + tf.stop_gradient(clip_x - x)

def z_normalize_data(data, data_mean, data_std):
    return (data - data_mean) / data_std

def z_unnormalize_data(data, data_mean, data_std):
    return data * data_std + data_mean

def Softmax_Selection(pred, pred_unc):
    softmap = Safe_Softmax(-pred_unc)
    output = tf.reduce_sum(softmap * pred, axis=[1, 2])
    
    return output
    
def patch_top_1_pred(pixel_pred, unc_pred, kernel_size=32, soft=False):
    pred_shape = pixel_pred.get_shape().as_list()
    # b x (h/k) x (w/k) x (kkc), 100x4x8x6144
    patch_pred = tf.image.extract_patches(pixel_pred, [1, kernel_size, kernel_size, 1], \
                                          [1, kernel_size, kernel_size, 1], \
                                          [1, 1, 1, 1], padding='SAME')    
    # b x (hw / kk) x (kk) x c, 100x32x1024x6
    patch_pred = tf.reshape(patch_pred, [pred_shape[0], -1, kernel_size * kernel_size, pred_shape[3]])
        
    patch_unc = tf.image.extract_patches(unc_pred, [1, kernel_size, kernel_size, 1], \
                                         [1, kernel_size, kernel_size, 1], \
                                         [1, 1, 1, 1], padding='SAME')
    patch_unc = tf.reshape(patch_unc, [pred_shape[0], -1, kernel_size * kernel_size, pred_shape[3]])
    
    # Split rx, ry, rz, tx, ty, tz ( b x (hw / kk) x (kk) )
    patch_rx, patch_ry, patch_rz, patch_tx, patch_ty, patch_tz = tf.split(patch_pred, num_or_size_splits=6, axis=-1)
    unc_rx, unc_ry, unc_rz, unc_tx, unc_ty, unc_tz = tf.split(patch_unc, num_or_size_splits=6, axis=-1)
    
    # List(100) 32 x 1024
    p_rx_group = tf.split(patch_rx, pred_shape[0], axis=0)
    p_ry_group = tf.split(patch_ry, pred_shape[0], axis=0)
    p_rz_group = tf.split(patch_rz, pred_shape[0], axis=0)
    p_tx_group = tf.split(patch_tx, pred_shape[0], axis=0)
    p_ty_group = tf.split(patch_ty, pred_shape[0], axis=0)
    p_tz_group = tf.split(patch_tz, pred_shape[0], axis=0)
    
    u_rx_group = tf.split(unc_rx, pred_shape[0], axis=0)
    u_ry_group = tf.split(unc_ry, pred_shape[0], axis=0)
    u_rz_group = tf.split(unc_rz, pred_shape[0], axis=0)
    u_tx_group = tf.split(unc_tx, pred_shape[0], axis=0)
    u_ty_group = tf.split(unc_ty, pred_shape[0], axis=0)
    u_tz_group = tf.split(unc_tz, pred_shape[0], axis=0)
    
    pred_r = []
    pred_t = []
    
    for (prx, pry, prz, urx, ury, urz) in zip(p_rx_group, p_ry_group, p_rz_group, u_rx_group, u_ry_group, u_rz_group):
        if soft:
            rx = soft_pred(prx[0], urx[0]) 
            ry = soft_pred(pry[0], ury[0])
            rz = soft_pred(prz[0], urz[0])
        else:
            rx = top_1_pred(prx[0], urx[0])
            ry = top_1_pred(pry[0], ury[0])
            rz = top_1_pred(prz[0], urz[0])
        
        r = tf.concat([rx, ry, rz], axis=-1)
        
        pred_r.append(r)
        
    for (ptx, pty, ptz, utx, uty, utz) in zip(p_tx_group, p_ty_group, p_tz_group, u_tx_group, u_ty_group, u_tz_group):
        if soft:
            tx = soft_pred(ptx[0], utx[0])
            ty = soft_pred(pty[0], uty[0])
            tz = soft_pred(ptz[0], utz[0])
        else:
            tx = top_1_pred(ptx[0], utx[0])
            ty = top_1_pred(pty[0], uty[0]) 
            tz = top_1_pred(ptz[0], utz[0])
        
        t = tf.concat([tx, ty, tz], axis=-1)
        
        pred_t.append(t)
        
    pred_r = tf.concat(pred_r, axis=0)
    pred_t = tf.concat(pred_t, axis=0)
    
    pred_r = tf.reshape(pred_r, [pred_shape[0], 3])
    pred_t = tf.reshape(pred_t, [pred_shape[0], 3])
    
    return pred_r, pred_t

def soft_pred(pred, unc):
    batch = pred.get_shape().as_list()[0]
    
    indices = tf.math.argmin(unc, axis=1)
    
    batch_idx = tf.reshape(tf.range(batch, dtype=tf.int32), [batch, 1])
    unc_idx = tf.concat([batch_idx, tf.cast(indices, tf.int32)], axis=-1)
    
    unc_val = tf.gather_nd(unc, unc_idx)
    pred_val = tf.gather_nd(pred, unc_idx)
    
    unc_softmax = Safe_Softmax(-unc_val, axis=0)
    soft_pred = pred_val * unc_softmax
    final_pred = tf.reduce_sum(soft_pred, axis=0)
    
    return final_pred
    
def top_1_pred(pred, unc):
    # (hw / kk) x (kk)
    batch = pred.get_shape().as_list()[0]
    
    indices = tf.math.argmin(unc, axis=1)
    
    batch_idx = tf.reshape(tf.range(batch, dtype=tf.int32), [batch, 1])
    unc_idx = tf.concat([batch_idx, tf.cast(indices, tf.int32)], axis=-1)
        
    final_pred = tf.gather_nd(pred, unc_idx)
    final_pred = tf.reduce_mean(final_pred, axis=0)
    
    return final_pred
    
def top_k_pred(pixel_pred, unc_pred, k=5):
    batch = pixel_pred.get_shape().as_list()[0]
    
    flat_pixel_pred = tf.reshape(pixel_pred, [batch, -1])
    flat_unc_pred = tf.reshape(unc_pred, [batch, -1])
    
    vals, indices = tf.math.top_k(-flat_unc_pred, k=k)
    batch_idx = tf.tile(tf.reshape(tf.range(batch, dtype=tf.int32), [batch, 1, 1]), [1, k, 1])
    unc_idx = tf.concat([batch_idx, tf.expand_dims(indices, axis=-1)], axis=-1)
    pred = tf.gather_nd(flat_pixel_pred, unc_idx)
    pred = tf.reduce_mean(pred, axis=-1, keepdims=True)
    
    return pred

def tf_invert_intrinsic(K):
    '''
    https://github.com/google-research/google-research/blob/master/depth_and_motion_learning/intrinsics_utils.py
    '''
    
    K_cols = tf.unstack(K, axis=-1)
    
    fx, _, _ = tf.unstack(K_cols[0], axis=-1)
    _, fy, _ = tf.unstack(K_cols[1], axis=-1)
    x0, y0, _ = tf.unstack(K_cols[2], axis=-1)
    
    zeros = tf.zeros_like(fx)
    ones = tf.ones_like(fx)
    
    row1 = tf.stack([1.0 / fx, zeros, zeros], axis=-1)
    row2 = tf.stack([zeros, 1.0 / fy, zeros], axis=-1)
    row3 = tf.stack([-x0 / fx, -y0 / fy, ones], axis=-1)
    
    return tf.stack([row1, row2, row3], axis=-1)

def deg2rad(x):
    return x * (np.pi / 180.0)

def rad2deg(x):
    return x * (180.0 / np.pi)

def angle2mat(angles):
    eulers = deg2rad(angles)
    
    eulers_shape = eulers.get_shape().as_list()
    x, y, z = tf.split(eulers, num_or_size_splits=3, axis=1)
    
    # tf.clip_by_value(x, -np.pi, np.pi)
    x = tf_clip_value_preserve_gradient(x, -np.pi, np.pi)
    y = tf_clip_value_preserve_gradient(y, -np.pi, np.pi)
    z = tf_clip_value_preserve_gradient(z, -np.pi, np.pi)
    
    # Expand to B x N x 1 x 1
    x = tf.expand_dims(tf.expand_dims(x, -1), -1) 
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    
    zeros = tf.zeros([eulers_shape[0], 1, 1, 1])
    ones = tf.ones([eulers_shape[0], 1, 1, 1])
    
    cx = tf.math.cos(x)
    sx = tf.math.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cx, -sx], axis=3)
    rotx_3 = tf.concat([zeros, sx, cx], axis=3)
    rotx = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)
    
    cy = tf.math.cos(y)
    sy = tf.math.sin(y)
    roty_1 = tf.concat([cy, zeros, sy], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-sy, zeros, cy], axis=3)
    roty = tf.concat([roty_1, roty_2, roty_3], axis=2)
    
    cz = tf.math.cos(z)
    sz = tf.math.sin(z)
    rotz_1 = tf.concat([cz, -sz, zeros], axis=3)
    rotz_2 = tf.concat([sz, cz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    rotz = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)
    
    rot_mat = tf.matmul(tf.matmul(rotx, roty), rotz)
    
    return rot_mat

def quaternion_to_matrix(quat):
    quat_shape = quat.get_shape().as_list()
    w, x, y, z = tf.unstack(quat, axis=-1)
    
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                      axis=-1)
    matrix = tf.reshape(matrix, [quat_shape[0], 3, 3])
    
    return matrix

def transformation_matrix(R, T):
    R_shape = R.get_shape().as_list()
    
    trans_vec = tf.expand_dims(T, -1)
    # rot_mat = euler2mat(R)
    rot_mat = angle2mat(R)
    rot_mat = tf.squeeze(rot_mat, axis=1)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [R_shape[0], 1, 1])
    
    trans_mat = tf.concat([rot_mat, trans_vec], axis=2)
    trans_mat = tf.concat([trans_mat, filler], axis=1)
    
    return trans_mat

def tf_pixel_coord_generation(batch, height, width, is_homogeneous=True, is_rescale=True, is_norm=False):
    n_x = tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1)
    n_y = tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1)
    
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])), tf.transpose(n_x, [1, 0]))
    y_t = tf.matmul(n_y, tf.ones(shape=tf.stack([1, width])))
    
    if is_rescale:
        x_t = (x_t + 1.0) * 0.5
        y_t = (y_t + 1.0) * 0.5
        
        if not is_norm:
            x_t = x_t * tf.cast(width - 1, tf.float32)
            y_t = y_t * tf.cast(height - 1, tf.float32)
    
    xy_coord = tf.concat([tf.expand_dims(x_t, axis=2), tf.expand_dims(y_t, axis=2)], axis=2)
    
    if is_homogeneous:
        xy_coord = tf.concat([xy_coord, tf.ones(shape=tf.stack([height, width, 1]))], axis=-1)
        
    # B x h x w x 3
    xy_coord = tf.tile(tf.expand_dims(xy_coord, axis=0), [batch, 1, 1, 1])
    
    if is_rescale:
        xy_coord = tf.math.round(xy_coord)
    
    return xy_coord
    
def compute_rigid_flow(depth_t0, R, T, K, K_inverse):
    batch, height, width, _ = depth_t0.get_shape().as_list()
    # K_inverse = tf_invert_intrinsic(K)
    
    trans_mat = transformation_matrix(R, T)
    pixel_coord_t0 = tf_pixel_coord_generation(batch, height, width)
    
    flat_depth_t0 = tf.reshape(tf.transpose(depth_t0, [0, 3, 1, 2]), [batch, 1, height * width])
    flat_pixel_coord_t0 = tf.reshape(tf.transpose(pixel_coord_t0, [0, 3, 1, 2]), [batch, 3, height * width])
    
    # B x 3 x (hw)
    cam_coord_t0 = tf.matmul(K_inverse, flat_pixel_coord_t0) * flat_depth_t0
    # B x 4 x (hw)
    cam_coord_t0 = tf.concat([cam_coord_t0, tf.ones(shape=[batch, 1, height * width])], axis=1)
    
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    pad_K = tf.concat([K, tf.zeros(shape=[batch, 3, 1])], axis=2)
    pad_K = tf.concat([pad_K, filler], axis=1)
    
    # B x 4 x (hw)
    cam_coord_t1 = tf.matmul(trans_mat, cam_coord_t0)
    unnormal_pixel_coord_t1 = tf.matmul(pad_K, cam_coord_t1)
    
    x_u = tf.slice(unnormal_pixel_coord_t1, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormal_pixel_coord_t1, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormal_pixel_coord_t1, [0, 2, 0], [-1, 1, -1])
    
    # Absolute for avoiding reflection
    x_n = x_u / (tf.abs(z_u) + 1e-12)
    y_n = y_u / (tf.abs(z_u) + 1e-12)
    
    pixel_coord_t1 = tf.reshape(tf.concat([x_n, y_n], axis=1), [batch, 2, height, width])
    pixel_coord_t1 = tf.transpose(pixel_coord_t1, [0, 2, 3, 1])
    
    rigid_flow = pixel_coord_t1 - pixel_coord_t0[:, :, :, :2]
    
    depth_t1 = tf.transpose(tf.reshape(z_u, [batch, 1, height, width]), [0, 2, 3, 1])
    
    return rigid_flow, depth_t1