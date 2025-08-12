import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Flatten, Conv2DTranspose,AlphaDropout,  Dropout, SeparableConv2D,Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute, LocallyConnected2D
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import MultiHeadAttention, Add, LayerNormalization, Flatten

class PCALayer(tf.keras.layers.Layer):
    def __init__(self, n_components, **kwargs):
        super(PCALayer, self).__init__(**kwargs)
        self.n_components = n_components

    def build(self, input_shape):
        self.shape = input_shape
        self.input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.n_components),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x):
        # Flatten the input tensor
        batch_size = tf.shape(x)[0]
        flattened = tf.reshape(x, [batch_size, -1, self.input_dim])

        # Compute the mean and subtract it from the input tensor
        mean = tf.reduce_mean(flattened, axis=1, keepdims=True)
        centered = flattened - mean

        # Compute the covariance matrix
        cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flattened)[1] - 1, tf.float32)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = tf.linalg.eigh(cov)

        #sorted_indices = tf.argsort(eigenvalues, direction='DESCENDING')
        #eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=1)

        # Take the top n_components eigenvectors and multiply them by the input tensor
        top_eigenvectors = eigenvectors[:, -self.n_components:]
        projected = tf.matmul(centered, top_eigenvectors)

        # Reshape the flattened tensor to the original shape and return it
        output_shape = tf.concat([tf.shape(x)[:-1], [self.n_components]], axis=0)
        output = tf.reshape(projected, output_shape)

        return output


    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.n_components,)

    def get_config(self):
        config = super(PCALayer, self).get_config()
        config.update({'n_components': self.n_components})
        return config

# class PCASortLayer(tf.keras.layers.Layer):
#     def __init__(self, output_filters, **kwargs):
#         super(PCASortLayer, self).__init__(**kwargs)
#         self.output_filters = output_filters

#     def build(self, input_shape):
#         self.shape = input_shape
#         self.input_dim = int(input_shape[-1])
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(self.input_dim, self.output_filters),
#                                       initializer='glorot_uniform',
#                                       trainable=False)

#     def call(self, x):
#         # Flatten the input tensor
#         batch_size = tf.shape(x)[0]
#         flattened = tf.reshape(x, [batch_size, -1, self.input_dim])

#         # Compute the mean and subtract it from the input tensor
#         mean = tf.reduce_mean(flattened, axis=1, keepdims=True)
#         centered = flattened - mean

#         # Compute the covariance matrix
#         cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flattened)[1] - 1, tf.float32)

#         # Compute the eigenvalues and eigenvectors of the covariance matrix
#         eigenvalues, eigenvectors = tf.linalg.eigh(cov)

#         # Sort the eigenvectors in descending order of eigenvalues
#         sorted_indices = tf.argsort(eigenvalues, direction='DESCENDING')
#         eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=1)



#         # Take the top n_components eigenvectors and multiply them by the input tensor
#         top_eigenvectors = eigenvectors[:, -self.output_filters:]
#         projected = tf.matmul(centered, top_eigenvectors)

#         # Reshape the flattened tensor to the original shape and return it
#         output_shape = tf.concat([tf.shape(x)[:-1], [self.output_filters]], axis=0)
#         output = tf.reshape(projected, output_shape)

#         return output

#     def compute_output_shape(self, input_shape):
#         return tuple(input_shape[:-1]) + (self.output_filters,)


#     def get_config(self):
#         config = super(PCASortLayer, self).get_config()
#         config.update({'output_filters': self.output_filters})
#         return config




def attention_block(inputs, ratio=4):
    x1 = attention_through_filters(inputs)
    x2 = attention_through_spatial(inputs)
    x = add([x1, x2])
    #x = PCALayer(filters)(inputs)
    return x
    

def conv2d_bn(x, filters, num_row, num_col):
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    #x = depthwise_convblock(x, filters, num_row, num_col)
    return x


def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='gelu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, (num_row, num_col), strides=strides, kernel_initializer="he_normal", padding=padding, use_bias=False, name=name)(x)
    x = tfa.layers.GroupNormalization(groups=filters//2, axis= channel_axis)(x)
    if(activation == None):
        return x

    x = tfa.activations.gelu(x, approximate=True)
    #x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return x

def attention_block(inputs, ratio=4):
    
    #inputs = iterLBlock(inputs, filters)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(inputs)
    
    x_3 = multiply([inputs, se])

    filters = x_3.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    # Use global average pooling to reduce the spatial dimensions
    x_1 = GlobalAveragePooling2D()(x_3)
    x_1 = Reshape(se_shape)(x_1)
    x_1 = Dense(filters, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(x_1)
    x_2 = GlobalMaxPooling2D()(x_3)
    x_2 = Reshape(se_shape)(x_2)
    x_2 = Dense(filters, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(x_1)

    x = add([x_1, x_2])

    se_2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(x)
    
    x = multiply([x_3, se_2])

    return x



def conv2d_bn(x, filters, num_row, num_col):
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    x = depthwise_convblock(x, filters, num_row, num_col)
    return x


def depthwise_convblock(inputs, filters, num_row, num_col, alpha=1, depth_multiplier=1, strides=(1,1), block_id=1, SE=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(filters * alpha)

    x = DepthwiseConv2D((num_row, num_col),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        kernel_initializer='he_normal',
                        use_bias=False)(inputs)
    x = BatchNormalization(axis=channel_axis,)(x)
    x = Activation('elu')(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),
                padding='same',
                kernel_initializer='he_normal',
                use_bias=False,
                strides=(1, 1))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('elu')(x)

    if(SE == True):
        x = attention_block(x)
        return x

    return x

 
def iterLBlock(x, filters, name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj = filters//8, filters//8, filters//2, filters//8, filters//4, filters//8

    conv_1x1 = initial_conv2d_bn(x, filters_1x1, 1, 1, padding='same', activation='selu')

    conv_3x3 = initial_conv2d_bn(x, filters_3x3, 1, 1, padding='same', activation='selu')
    conv_3x3 = conv2d_bn(conv_3x3, filters_3x3, 3, 3)

    conv_5x5 = initial_conv2d_bn(x, filters_5x5, 1, 1, padding='same', activation='selu')
    conv_5x5 = conv2d_bn(conv_5x5, filters_5x5, 3, 3)
    conv_5x5_3x1 = initial_conv2d_bn(conv_5x5, filters_5x5//2, 3,1, padding='same', activation='selu')
    conv_5x5_1x3 = initial_conv2d_bn(conv_5x5, filters_5x5//2, 1,3, padding='same', activation='selu')

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = initial_conv2d_bn(pool_proj, filters_pool_proj, 1, 1)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5_3x1, conv_5x5_1x3, pool_proj], axis=3, name=name)
    output = tfa.layers.GroupNormalization(groups=filters//2, axis=channel_axis)(output)
    output = tfa.activations.gelu(output, approximate=True)
    return output

def ResPath(inp, filters, length):
    shortcut = inp
    shortcut = initial_conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = initial_conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = initial_conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = initial_conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def encoder_block(x, filters, num_row, num_col, length):
    x = iterLBlock(x, filters)
    pca = PCALayer(filters)(x)
    output = add([x, pca]) 
    output = ResPath(output, filters, length)
    return output

def decoder_block(inputs, skip, filters):
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    x = iterLBlock(x, filters)
    output = depthwise_convblock(x, filters*2, 3, 3, SE=True)
    return output


#['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block6_1_conv']


def PCABoilNet(input_filters, height, width, n_channels):
     #inputs = Input((height, width, n_channels), name = "input_image")
    filters = input_filters
    model_input = Input(shape=(height, width, n_channels))
    """ Pretrained resnet"""
    tf.keras.backend.clear_session()
    #base_model = tf.keras.applications.MobileNetV2(input_tensor=model_input, include_top=False, weights="imagenet",  alpha=1.3)
    base_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_tensor=model_input, pooling=max)

    #resnet50 = keras.applications.ResNet50(
     #   weights="imagenet", include_top=False, input_tensor=model_input
    #)
    print("Number of layers in the base model: ", len(base_model.layers))
 

    base_model.trainable = True
    
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    for i, layer in enumerate(base_model.layers[:-48]):
        layer.trainable = False


        """    # Iteration 2
    s11 = attention_block(base_model.get_layer("input_1").output)
    s11 = encoder_block(s11, filters, 3, 3, 4) 
    #s11 = tf.keras.layers.SpatialDropout2D(0.06)(s11)              ## (256 x 256)
    #s11 = iterLBlock(s11, filters)
    s21 = attention_block(base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    s21 = encoder_block(s21, filters*2, 3, 3, 3) 
    #s21 = tf.keras.layers.SpatialDropout2D(0.06)(s21)
    #s21 = iterLBlock(s21, filters*2)
    s31 = attention_block(base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    #s31 = iterLBlock(s31, filters*4)
    s31 = encoder_block(s31, filters*4, 3, 3, 2) 
    s41 = attention_block(base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    #s41 = iterLBlock(s41, filters*8)
    s41 = encoder_block(s41, filters*8, 3, 3, 1) 

    
    b11 = attention_block(base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    b11 = encoder_block(b11, filters*16, 3, 3, 0) 
    #b11= iterLBlock(b11, filters*16)
    #depthwise_convblock(b11, filters*16, 3,3, depth_multiplier=1, SE=True)

    d11 = decoder_block(b11, s41, filters*8)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*4)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*2)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters*1)                          ## (256 x 256)"""

        # Iteration 2
    """ Encoder """
    s11 = attention_block(base_model.get_layer("input_1").output)
    pca1 = PCALayer(n_channels)(s11)
    s11 = add([s11, pca1])
    s11 = ResPath(s11, filters, 4) 
    #s11 = tf.keras.layers.SpatialDropout2D(0.06)(s11)              ## (256 x 256)
    #s11 = iterLBlock(s11, filters)
    s21 = attention_block(base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    pca2 = PCALayer(filters*2)(s21)
    s21 = add([s21, pca2])
    s21 = ResPath(s21, filters*2, 3) 
    #s21 = tf.keras.layers.SpatialDropout2D(0.06)(s21)
    #s21 = iterLBlock(s21, filters*2)
    s31 = attention_block(base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    pca3 = PCALayer(filters*2)(s31)
    s31 = add([s31, pca3])
    s31 = ResPath(s31, filters*4, 2)
    
    s41 = attention_block(base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    pca4 = PCALayer(filters*4)(s41)
    s41 = add([s41, pca4])
    s41 = ResPath(s41, filters*8, 1)

    """ Bridge """
    b11 = attention_block(base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    #b11 = encoder_block(b11, filters*16, 3, 3) 
    b11= iterLBlock(b11, filters*16)
    #depthwise_convblock(b11, filters*16, 3,3, depth_multiplier=1, SE=True)

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*8)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*4)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*2)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters*1)                          ## (256 x 256)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="PCABoilNet")

    return model



def main():

# Define the model

    model = PCABoilNet(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main() 