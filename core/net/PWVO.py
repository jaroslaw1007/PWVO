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
    def __init__(self, in_channels, out_channels, k_size=3, strides=1, dilation=1, act_fn=None, is_norm=False):
        super(ResidualBlock, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
            
        # initializer = tf.keras.initializers.HeNormal()
        initializer = tf.keras.initializers.GlorotNormal()
        
        if is_norm:
            self.conv1 = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=k_size, strides=strides, \
                              padding='same', kernel_initializer=initializer),
                BatchInstanceNormalization()
            ])
            
            self.conv2 = tf.keras.Sequential([
                layers.Conv2D(out_channels, kernel_size=k_size, strides=1, dilation_rate=dilation, \
                              padding='same', kernel_initializer=initializer),
                BatchInstanceNormalization()
            ])
        else:
            self.conv1 = layers.Conv2D(out_channels, kernel_size=k_size, strides=strides, \
                                       padding='same', kernel_initializer=initializer)
            self.conv2 = layers.Conv2D(out_channels, kernel_size=k_size, strides=1, dilation_rate=dilation, \
                                       padding='same', kernel_initializer=initializer)
        
        if strides == 1 and in_channels == out_channels:
            self.downsample = None
        else:
            if is_norm:
                self.downsample = tf.keras.Sequential([
                    layers.Conv2D(out_channels, kernel_size=1, strides=strides, kernel_initializer=initializer),
                    BatchInstanceNormalization()
                ])
            else:
                self.downsample = layers.Conv2D(out_channels, kernel_size=1, strides=strides, \
                                                kernel_initializer=initializer)
        
    def call(self, x, training):
        y = x
        y = self.act_fn(self.conv1(y, training=training))
        y = self.act_fn(self.conv2(y, training=training))
        
        if self.downsample is not None:
            x = self.downsample(x, training=training)
            
        return self.act_fn(x + y)
    
class PredictionHead(layers.Layer):
    def __init__(self, hidden_dims=256, out_channels=3, act_fn=None):
        super(PredictionHead, self).__init__()
        
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
    
class FuseConv(layers.Layer):
    def __init__(self, out_channels=32, act_fn=None):
        super(FuseConv, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
            
        # initializer = tf.keras.initializers.HeNormal()
        initializer = tf.keras.initializers.GlorotNormal()
        
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)
        
    def call(self, x, training):
        hf = self.act_fn(self.conv1(x, training=training))
        
        return hf

class VOAutoEncoder(layers.Layer):
    def __init__(self, out_channels=128, act_fn=None):
        super(VOAutoEncoder, self).__init__()
        
        if act_fn is not None:
            self.act_fn = act_fn
        else:
            self.act_fn = tf.nn.relu
            
        # initializer = tf.keras.initializers.HeNormal()
        initializer = tf.keras.initializers.GlorotNormal()
        
        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer) # 64x128
        self.conv2 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.conv3 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        
        self.layer1 = self._make_layer(32, 64, 3) # B x 32 x 64 x 64
        self.layer2 = self._make_layer(64, 128, 4) # B x 16 x 32 x 128
        self.layer3 = self._make_layer(128, 128, 6) # B x 8 x 16 x 128
        self.layer4 = self._make_layer(128, 256, 7) # B x 4 x 8 x 256
        self.layer5 = self._make_layer(256, 256, 3) # B x 2 x 4 x 256
        
        self.deconv5 = layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), \
                                              padding='same', kernel_initializer=initializer) # 4 x 8
        self.deconv4 = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), \
                                              padding='same', kernel_initializer=initializer) # 8 x 16
        self.deconv3 = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), \
                                              padding='same', kernel_initializer=initializer) # 16 x 32
        self.deconv2 = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), \
                                              padding='same', kernel_initializer=initializer) # 32 x 64
        self.deconv1 = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), \
                                              padding='same', kernel_initializer=initializer)
        self.deconv0 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), \
                                              padding='same', kernel_initializer=initializer)
        
        self.convblock5 = ResidualBlock(512, 128, 3, 1, 1, self.act_fn)
        self.convblock4 = ResidualBlock(256, 128, 3, 1, 1, self.act_fn)
        self.convblock3 = ResidualBlock(256, 128, 3, 1, 1, self.act_fn)
        self.convblock2 = ResidualBlock(256, 64, 3, 1, 1, self.act_fn)
        self.convblock1 = ResidualBlock(128, 64, 3, 1, 1, self.act_fn)
        self.convblock0 = ResidualBlock(32, 32, 3, 1, 1, self.act_fn)
        
        # self.attn = SelfAttnBlock(256)
        
        self.outconv = layers.Conv2D(32, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)
        
    def _make_layer(self, in_channels, out_channels, block_num):
        model = tf.keras.Sequential()
        model.add(ResidualBlock(in_channels, out_channels, 3, 2, 1, self.act_fn))
        
        for _ in range(1, block_num):
            model.add(ResidualBlock(out_channels, out_channels, 3, 1, 1, self.act_fn))
            
        return model
    
    def call(self, x, training):
        hf = self.act_fn(self.conv1(x, training=training))
        hf = self.act_fn(self.conv2(hf, training=training))
        hf = self.act_fn(self.conv3(hf, training=training))
        
        hf1 = self.layer1(hf, training=training)
        hf2 = self.layer2(hf1, training=training)
        hf3 = self.layer3(hf2, training=training)
        hf4 = self.layer4(hf3, training=training)
        hf5 = self.layer5(hf4, training=training)
        # hf5 = self.attn(hf5, training=training)
        
        up5 = self.deconv5(hf5, training=training) # 4 x 8
        up5 = tf.concat([up5, hf4], axis=-1) # 256 + 256
        up5 = self.convblock5(up5, training=training)
        
        up4 = self.deconv4(up5, training=training) # 8 x 16
        up4 = tf.concat([up4, hf3], axis=-1) # 128 + 128
        up4 = self.convblock4(up4, training=training) 
        
        up3 = self.deconv3(up4, training=training) # 16 x 32
        up3 = tf.concat([up3, hf2], axis=-1) # 128 + 128
        up3 = self.convblock3(up3, training=training) 
        
        up2 = self.deconv2(up3, training=training) # 32 x 64
        up2 = tf.concat([up2, hf1], axis=-1) # 64 + 64
        up2 = self.convblock2(up2, training=training)
        
        up1 = self.deconv1(up2, training=training) # 64 x 128
        up1 = self.convblock1(up1, training=training)
        
        up0 = self.deconv0(up1, training=training) # 128 x 256
        up0 = self.convblock0(up0, training=training)
        out = self.outconv(up0, training=training)
        
        hidden_feature = hf5
        final_feature = out
        
        return hidden_feature, final_feature
    
class PWVO(Model):
    def __init__(self, args):
        super(PWVO, self).__init__()
        
        initializer = tf.keras.initializers.HeNormal()
        self.args = args
        
        self.depth_conv = FuseConv(8, act_fn=tf.nn.elu)
        self.coord_conv = FuseConv(8, act_fn=tf.nn.elu)
        
        self.autoencoder = VOAutoEncoder(act_fn=tf.nn.elu)
        
        self.rt_prediction_block = PredictionHead(32, 6, act_fn=tf.nn.elu)
        self.rt_uncertainty_block = PredictionHead(32, 6, act_fn=tf.nn.elu)
        
        self.p_coord_homo = tf_pixel_coord_generation(args.batch_size, args.height, args.width, True, True)
        self.pixel_coord_t0_k = tf.reshape(tf.transpose(self.p_coord_homo, [0, 3, 1, 2]), [args.batch_size, 3, -1])
                
    def call(self, flow, depth_t0, depth_t1, K, training=True):
        k_inv = tf_invert_intrinsic(K)
        c_coord_t0 = tf.matmul(k_inv, self.pixel_coord_t0_k)
        c_coord_t0 = tf.reshape(c_coord_t0, [self.args.batch_size, 3, self.args.height, self.args.width])
        c_coord_t0 = tf.transpose(c_coord_t0, [0, 2, 3, 1])
        c_coord_t0 = tf.slice(c_coord_t0, [0, 0, 0, 0], [-1, -1, -1, 2])
        
        depth_feature = self.depth_conv(tf.concat([depth_t0, depth_t1], axis=-1), training=training)
            
        coord_feature = self.coord_conv(c_coord_t0, training=training)
        
        if self.args.coord:
            x = tf.concat([flow, depth_feature, coord_feature], axis=-1)
        else:
            x = tf.concat([flow, depth_feature], axis=-1)
        
        hidden_feature, final_feature = self.autoencoder(x, training=training)

        rt_pixel_pred = self.rt_prediction_block(final_feature, training=training)
        rot_pred, trans_pred = tf.split(rt_pixel_pred, num_or_size_splits=2, axis=-1)
        rt_pixel_uncertainty = self.rt_uncertainty_block(final_feature, training=training)

        rot_out = rot_pred * 0.01
        trans_out = trans_pred * 0.01
        
        return rot_out, trans_out, rt_pixel_uncertainty
