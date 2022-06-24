import numpy as np
from collections import Counter

def save_func(save_path, save_data):
    with open(save_path, 'wb') as f:
        np.save(f, save_data)
    
    f.close()

def intrinsic_inverse(K):
    fx = K[0, 0]
    fy = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]
    
    K_inv = np.array([
        1.0 / fx, 0.0, -x0 / fx,
        0.0, 1.0 / fy, -y0 / fy,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    return K_inv

def pixel_coord_generation(init_h, init_w, h, w, is_homogeneous=True):
    # Np array and Camera image plane are transpose (x <-> y)
    n_x = np.linspace(-1.0, 1.0, w)[..., np.newaxis]
    n_y = np.linspace(-1.0, 1.0, h)[..., np.newaxis]
    
    x_t = np.dot(np.ones((h, 1)), n_x.T)
    y_t = np.dot(n_y, np.ones((1, w)))
    
    x_t = (x_t + 1.0) * 0.5 * (init_w - 1.0)
    y_t = (y_t + 1.0) * 0.5 * (init_h - 1.0)
    
    xy_coord = np.concatenate([x_t[..., np.newaxis], y_t[..., np.newaxis]], axis=-1)
    
    if is_homogeneous:
        xy_coord = np.concatenate([xy_coord, np.ones((h, w, 1))], axis=-1)
    
    return xy_coord

def deg2mat_xyz(deg, left_hand=False):
    if left_hand:
        deg = -deg
    
    rad = np.deg2rad(deg)
    x, y, z = np.split(rad, 3, axis=0)
    
    sz = np.sin(z[0])
    cz = np.cos(z[0])
    sy = np.sin(y[0])
    cy = np.cos(y[0])
    sx = np.sin(x[0])
    cx = np.cos(x[0])

    rot_x = np.array([
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx
    ]).reshape(3, 3)
    
    rot_y = np.array([
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    ]).reshape(3, 3)
    
    rot_z = np.array([
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    R = np.dot(np.dot(rot_x, rot_y), rot_z)

    return R

def deg2mat_zyx(deg, left_hand=False):
    if left_hand:
        deg = -deg
    
    rad = np.deg2rad(deg)
    x, y, z = np.split(rad, 3, axis=0)
    
    sz = np.sin(z[0])
    cz = np.cos(z[0])
    sy = np.sin(y[0])
    cy = np.cos(y[0])
    sx = np.sin(x[0])
    cx = np.cos(x[0])

    rot_x = np.array([
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx
    ]).reshape(3, 3)
    
    rot_y = np.array([
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    ]).reshape(3, 3)
    
    rot_z = np.array([
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    R = np.dot(np.dot(rot_z, rot_y), rot_x)

    return R

def generate_depth_image(img_height, img_width, p_coord_t1):
    '''
    Reference to https://github.com/nianticlabs/monodepth2/blob/master/kitti_utils.py
    '''
    # Check if in bounds
    p_coord_t1[0, :] = np.round(p_coord_t1[0, :])
    p_coord_t1[1, :] = np.round(p_coord_t1[1, :])
    valid_indices = (p_coord_t1[0, :] >= 0) & (p_coord_t1[1, :] >= 0)
    valid_indices = valid_indices & (p_coord_t1[0, :] < img_width) & (p_coord_t1[1, :] < img_height) 
    p_coord_t1 = p_coord_t1[:, valid_indices]
    
    # Project to image
    depth_map_t1 = np.zeros((img_height, img_width, 1))
    depth_map_t1[p_coord_t1[1, :].astype(np.int), p_coord_t1[0, :].astype(np.int), :] = p_coord_t1[2, :][..., np.newaxis] 
    
    # Find the duplicate points and choose closest depth
    indices = p_coord_t1[1, :] * img_width + p_coord_t1[0, :]
    duplicate_indices = [item for item, count in Counter(indices).items() if count > 1]
    
    for d_idx in duplicate_indices:
        d_coord = np.where(indices == d_idx)[0]
        x_loc = int(p_coord_t1[0, d_coord[0]])
        y_loc = int(p_coord_t1[1, d_coord[0]])
        depth_map_t1[y_loc, x_loc, :] = p_coord_t1[2, d_coord].min()
        
    depth_map_t1[depth_map_t1 < 0] = 0
    
    return depth_map_t1

def create_motion(K_t0, K_t1, R, T, p_coord_t0, d_t0):
    p_shape = p_coord_t0.shape
    
    flat_p_coord_t0 = np.reshape(np.transpose(p_coord_t0, (2, 0, 1)), (3, -1))
    flat_d_t0 = np.reshape(np.transpose(d_t0, (2, 0, 1)), (-1))
    
    M = np.vstack((np.hstack((R, T[:, None])), [0, 0, 0 ,1]))
    
    K_t0_inv = intrinsic_inverse(K_t0)
    K_t1_pad = np.vstack((np.hstack((K_t1, np.zeros([3, 1]))), [0, 0, 0 ,1]))
    
    filler = np.ones((1, p_shape[0] * p_shape[1]))
    
    c_coord_t0 = np.matmul(K_t0_inv, flat_p_coord_t0) * flat_d_t0
    
    c_coord_t0 = np.vstack((c_coord_t0, filler))
    c_coord_t1 = np.matmul(M, c_coord_t0)
    
    flat_scene_motion = c_coord_t1[:3, :] - c_coord_t0[:3, :]
    
    unnormal_p_coord_t1 = np.matmul(K_t1_pad, c_coord_t1)
    # Absolute for avioding reflection
    p_coord_t1 = unnormal_p_coord_t1 / (np.abs(unnormal_p_coord_t1[2, :]) + 1e-12)
    
    flat_f = p_coord_t1[:2, :] - flat_p_coord_t0[:2, :]
    
    # if gen_depth:
        # p_coord_t1[2, :] = d_t1
        # depth_map_t1 = generate_depth_map(p_shape[0], p_shape[1], p_coord_t1)
    # else:
        # depth_map_t1 = np.zeros((p_shape[0], p_shape[1], 1))
        
    scene_motion = np.transpose(np.reshape(flat_scene_motion, (3, p_shape[0], p_shape[1])), (1, 2, 0))
    f = np.transpose(np.reshape(flat_f, (2, p_shape[0], p_shape[1])), (1, 2, 0))
    
    return scene_motion, f

def rotation_cancelation(f, K, R, T, img_height, img_width, p_coord_t0, d_t1, gen_depth=False):
    R_inv = np.transpose(R)
    K_inv = intrinsic_inverse(K)
    
    pad_R_inv = np.vstack((np.hstack((R_inv, np.zeros([3, 1]))), [0, 0, 0 ,1]))
    pad_K = np.vstack((np.hstack((K, np.zeros([3, 1]))), [0, 0, 0 ,1]))
    
    rot_prime = np.matmul(np.eye(3) - R_inv, T[:, None])
    rot_removal_mat = np.vstack([np.hstack([R_inv, rot_prime]), [0, 0, 0, 1]])
    
    filler = np.ones([1, p_coord_t0.shape[1]])
    p_coord_t1 = f + p_coord_t0[:2, :]
    p_coord_t1 = np.vstack([p_coord_t1, filler])
    
    c_coord_t1 = np.matmul(K_inv, p_coord_t1) * d_t1
    c_coord_t1 = np.vstack([c_coord_t1, filler])
    
    rot_removal_c_coord_t1 = np.matmul(rot_removal_mat, c_coord_t1)
    rot_removal_d_t1 = rot_removal_c_coord_t1[2, :]
    rot_removal_unnormal_p_coord_t1 = np.matmul(pad_K, rot_removal_c_coord_t1)
    rot_removal_p_coord_t1 = rot_removal_unnormal_p_coord_t1 / (np.abs(rot_removal_unnormal_p_coord_t1[2, :]) + 1e-12)
    
    rot_removal_flow = rot_removal_p_coord_t1[:2, :] - p_coord_t0[:2, :]
    if gen_depth:
        rot_removal_p_coord_t1[2, :] = rot_removal_d_t1
        rot_removal_depth_map_t1 = generate_depth_map(img_height, img_width, rot_removal_p_coord_t1)
    else:
        rot_removal_depth_map_t1 = np.zeros((img_height, img_width, 1))
    
    return rot_removal_flow, rot_removal_d_t1, rot_removal_depth_map_t1

def translation_cancelation(f, K, T, img_height, img_width, p_coord_t0, d_t1, gen_depth=False):
    K_inv = intrinsic_inverse(K)
    
    pad_K = np.vstack((np.hstack((K, np.zeros([3, 1]))), [0, 0, 0 ,1]))
    
    trans_removal_mat = np.vstack([np.hstack([np.eye(3), -T[:, None]]), [0, 0, 0, 1]])
    
    filler = np.ones([1, p_coord_t0.shape[1]])
    p_coord_t1 = f + p_coord_t0[:2, :]
    p_coord_t1 = np.vstack([p_coord_t1, filler])
    
    c_coord_t1 = np.matmul(K_inv, p_coord_t1) * d_t1
    c_coord_t1 = np.vstack([c_coord_t1, filler])
    
    trans_removal_c_coord_t1 = np.matmul(trans_removal_mat, c_coord_t1)
    trans_removal_d_t1 = trans_removal_c_coord_t1[2, :]
    trans_removal_unnormal_p_coord_t1 = np.matmul(pad_K, trans_removal_c_coord_t1) 
    trans_removal_p_coord_t1 = trans_removal_unnormal_p_coord_t1 / (np.abs(trans_removal_unnormal_p_coord_t1[2, :]) + 1e-12)
    
    trans_removal_flow = trans_removal_p_coord_t1[:2, :] - p_coord_t0[:2, :]
    
    if gen_depth:
        trans_removal_p_coord_t1[2, :] = trans_removal_d_t1
        trans_removal_depth_map_t1 = generate_depth_map(img_height, img_width, trans_removal_p_coord_t1)
    else:
        trans_removal_depth_map_t1 = np.zeros((img_height, img_width, 1))
    
    return trans_removal_flow, trans_removal_d_t1, trans_removal_depth_map_t1

