import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

from core.utils.tf_utils import tf_pixel_coord_generation, tf_invert_intrinsic

class SelfAttnBlock(layers.Layer):
    def __init__(self, in_channels):
        super(SelfAttnBlock, self).__init__()
        
        self.query_conv = layers.Conv2D(in_channels // 8, kernel_size=1)
        self.key_conv = layers.Conv2D(in_channels // 8, kernel_size=1)
        self.value_conv = layers.Conv2D(in_channels, kernel_size=1)
        
        self.gamma = tf.Variable(initial_value=[0.0], trainable=True)
        
    def call(self, x, training):
        b, h, w, c = x.get_shape().as_list()
        
        q = self.query_conv(x, training=training)
        k = self.key_conv(x, training=training)
        v = self.value_conv(x, training=training)
        
        proj_query = tf.reshape(q, [b, h * w, -1])
        proj_key = tf.transpose(tf.reshape(k, [b, h * w, -1]), [0, 2, 1])
        proj_value = tf.transpose(tf.reshape(v, [b, h * w, -1]), [0, 2, 1])
        
        energy = tf.matmul(proj_query, proj_key)
        attention = tf.nn.softmax(energy)
        out = tf.matmul(proj_value, tf.transpose(attention, [0, 2, 1]))
        out = tf.reshape(tf.transpose(out, [0, 2, 1]), [b, h, w, -1])
        
        return tf.add(tf.multiply(out, self.gamma), x)

class ResidualBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, k_size=3, strides=1, dilation=1, act_fn=None):
        super(ResidualBlock, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
            
        initializer = tf.keras.initializers.GlorotNormal()
        
        self.conv1 = layers.Conv2D(out_channels, kernel_size=k_size, strides=strides, dilation_rate=dilation, \
                                   padding='same', kernel_initializer=initializer)
        self.conv2 = layers.Conv2D(out_channels, kernel_size=k_size, strides=1, dilation_rate=dilation, \
                                   padding='same', kernel_initializer=initializer)
        
        if strides == 1 and in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = layers.Conv2D(out_channels, kernel_size=1, strides=strides, \
                                            kernel_initializer=initializer)
        
    def call(self, x, training):
        y = x
        y = self.act_fn(self.conv1(y, training=training))
        y = self.conv2(y, training=training)
        
        if self.downsample is not None:
            x = self.downsample(x, training=training)
            
        return self.act_fn(x + y)
    
class PredictionHead(layers.Layer):
    def __init__(self, out_channels=3, act_fn=None):
        super(PredictionHead, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
        
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(32)
        self.fc3 = layers.Dense(out_channels)
        
    def call(self, x, training):
        x_shape = x.get_shape().as_list()
        flatten_x = tf.reshape(x, [x_shape[0], -1])
        
        h_f = self.act_fn(self.fc1(flatten_x, training=training))
        h_f = self.act_fn(self.fc2(h_f, training=training))
        out = self.fc3(h_f, training=training)
        
        return out
    
class AvgPredictionHead(layers.Layer):
    def __init__(self, out_channels=3, act_fn=None):
        super(AvgPredictionHead, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
        
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(32)
        self.fc3 = layers.Dense(out_channels)
        
    def call(self, x, training):
        x = tf.reduce_mean(x, axis=[1, 2])
        
        h_f = self.act_fn(self.fc1(x, training=training))
        h_f = self.act_fn(self.fc2(h_f, training=training))
        out = self.fc3(h_f, training=training)
        
        return out
    
class FuseConv(layers.Layer):
    def __init__(self, out_channels=32, act_fn=None):
        super(FuseConv, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
            
        initializer = tf.keras.initializers.GlorotNormal()
        
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)
        
    def call(self, x, training):
        hf = self.act_fn(self.conv1(x, training=training))
        
        return hf
    
class ConvPredictionHead(layers.Layer):
    def __init__(self, hidden_dims=256, out_channels=3, act_fn=None):
        super(ConvPredictionHead, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
        
        self.conv1 = layers.Conv2D(hidden_dims, kernel_size=3, padding='same')
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, padding='same')
        
    def call(self, x, training):
        h_f = self.act_fn(self.conv1(x, training=training))
        out = self.conv2(h_f, training=training)
        
        return out 

class VOBasicEncoder(layers.Layer):
    def __init__(self, out_channels=256, act_fn=None):
        super(VOBasicEncoder, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
            
        initializer = tf.keras.initializers.GlorotNormal()
        
        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer) # 64x128
        self.conv2 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.conv3 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        
        self.layer1 = self._make_layer(32, 64, 3) # 32x64
        self.layer2 = self._make_layer(64, 128, 4) # 16x32
        self.layer3 = self._make_layer(128, 128, 6) # 8x16
        self.layer4 = self._make_layer(128, 256, 7) # 4x8
        self.layer5 = self._make_layer(256, 256, 3) # 2x4
        
        # self.attn = SelfAttnBlock(256)
        
    def _make_layer(self, in_channels, out_channels, block_num):
        model = tf.keras.Sequential()
        model.add(ResidualBlock(in_channels, out_channels, 3, 2, 1, self.act_fn))
        
        for _ in range(1, block_num):
            model.add(ResidualBlock(out_channels, out_channels, 3, 1, 1, self.act_fn))
            
        return model
        
    def call(self, x, training):
        h_f = self.act_fn(self.conv1(x, training=training))
        h_f = self.act_fn(self.conv2(h_f, training=training))
        h_f = self.act_fn(self.conv3(h_f, training=training))
        
        h_f = self.layer1(h_f, training=training)
        h_f = self.layer2(h_f, training=training)
        h_f = self.layer3(h_f, training=training)
        h_f = self.layer4(h_f, training=training)
        h_f = self.layer5(h_f, training=training)
        # h_f = self.attn(h_f, training=training)
        
        return h_f
    
class VONet(Model):
    def __init__(self, args):
        super(VONet, self).__init__()
        
        initializer = tf.keras.initializers.GlorotNormal()
        self.args = args
        
        self.encoder = VOBasicEncoder(act_fn=tf.nn.elu)
        
        self.depth_conv = FuseConv(8, act_fn=tf.nn.elu)
        self.coord_conv = FuseConv(8, act_fn=tf.nn.elu)
        
        self.r_block = PredictionHead(3, act_fn=tf.nn.elu)
        self.t_block = PredictionHead(3, act_fn=tf.nn.elu)
        
        self.p_coord_homo = tf_pixel_coord_generation(args.batch_size, args.height, args.width, True, True)
        self.pixel_coord_t0 = tf.reshape(tf.transpose(self.p_coord_homo, [0, 3, 1, 2]), [args.batch_size, 3, -1])
        
        self.filler = tf.ones([args.batch_size, 1, args.height * args.width])
                
    def call(self, flow, depth_t0, depth_t1, k, training=True):
        k_inv = tf_invert_intrinsic(k)
        
        c_coord_t0 = tf.matmul(k_inv, self.pixel_coord_t0)
        c_coord_t0 = tf.reshape(c_coord_t0, [self.args.batch_size, 3, self.args.height, self.args.width])
        c_coord_t0 = tf.transpose(c_coord_t0, [0, 2, 3, 1])
        c_coord_t0 = tf.slice(c_coord_t0, [0, 0, 0, 0], [-1, -1, -1, 2])
        
        depth_feature = self.depth_conv(tf.concat([depth_t0, depth_t1], axis=-1), training=training)
        coord_feature = self.coord_conv(c_coord_t0, training=training)
        
        if self.args.coord:
            x = tf.concat([flow, depth_feature, coord_feature], axis=-1)
        else:
            x = tf.concat([flow, depth_feature], axis=-1)
                    
        flow_feature = self.encoder(x, training=training)
            
        rot_pred = self.r_block(flow_feature, training=training) * 0.01
        trans_pred = self.t_block(flow_feature, training=training) * 0.01
        
        return rot_pred, trans_pred
