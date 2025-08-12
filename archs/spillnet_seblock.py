import tensorflow as tf
# import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Flatten,Lambda, Conv2DTranspose,AlphaDropout,  Dropout, SeparableConv2D,Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import MultiHeadAttention, Add, LayerNormalization, Flatten

seed = 2023
from seed_utils import set_global_seed
set_global_seed(seed)


class PCALayer(tf.keras.layers.Layer):
    def __init__(self, n_components, **kwargs):
        super(PCALayer, self).__init__(**kwargs)
        self.n_components = n_components

    def build(self, input_shape):
        self.shape = input_shape
        self.input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.n_components), dtype="float32",
                                      initializer='glorot_uniform',
                                      trainable=False)

    def call(self, x):
        # Flatten the input tensor
        #x = tf.linalg.normalize(x,axis=-1)
        #print(x.shape)
        # assumption is that the feature vector is normalized
        #x = tf.math.l2_normalize(x, axis=-1)
        batch_size = tf.shape(x)[0]
        flattened = tf.reshape(x, [batch_size, -1, self.input_dim])
        
        # Compute the mean and subtract it from the input tensor
        mean = tf.reduce_mean(flattened, axis=1, keepdims=True)
        centered = flattened - mean
        

        # Compute the covariance matrix
        cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flattened)[1] - 1, tf.float32)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = tf.linalg.eigh(cov)

        # Sort the eigenvectors based on the eigenvalues
        idx = tf.argsort(eigenvalues, axis=-1, direction='DESCENDING')
        top_eigenvectors = tf.gather(eigenvectors, idx, batch_dims=1, axis=-1)
        top_eigenvectors = top_eigenvectors[:, :, :self.n_components]

        # Transpose the eigenvectors to match the input shape
        top_eigenvectors = tf.transpose(top_eigenvectors, perm=[0, 1, 2])
        
        # Project centered data onto top principal components
        projected = tf.matmul(centered, top_eigenvectors)

        # Reshape projected data and return as output
        output_shape = tf.concat([tf.shape(x)[:-1], [self.n_components]], axis=0)
        #output = tf.reshape(projected, shape=(-1, *self.output_shape))
        output = tf.reshape(projected, output_shape)
        return output



    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.n_components,)

    def get_config(self):
        config = super(PCALayer, self).get_config()
        config.update({'n_components': self.n_components})
        return config

def spatial_pooling_block(inputs, ratio=4):
    
    #inputs = iterLBlock(inputs, filters)
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    
    se_shape = (1, 1, filters)
    

    spp_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(inputs)
    spp_1 = layers.GlobalMaxPooling2D()(spp_1)
    spp_1 = Reshape(se_shape)(spp_1)
    spp_1 = Dense(filters, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=True, bias_initializer='zeros')(spp_1)
    #

    spp_2 = MaxPooling2D(pool_size=(4,4), strides=(4,4), padding='same')(inputs)    
    spp_2 = layers.GlobalMaxPooling2D()(spp_2)
    spp_2 = Reshape(se_shape)(spp_2)
    spp_2 = Dense(filters, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=True, bias_initializer='zeros')(spp_2)
    #print(f'shape of spp2 {spp_2.shape}')
        

    spp_3 = MaxPooling2D(pool_size=(8,8), strides=(8,8), padding='same')(inputs)
    spp_3 = layers.GlobalMaxPooling2D()(spp_3)
    spp_3 = Reshape(se_shape)(spp_3)
    spp_3 = Dense(filters, activation='relu',kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=True, bias_initializer='zeros')(spp_3)
    #print(f'shape of spp3 {spp_3.shape}')
    

    feature = Add()([spp_1,spp_2, spp_3])
    #feature = Dense(filters, activation='sigmoid',kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=True, bias_initializer='zeros')(feature)
    feature = Activation('sigmoid')(feature)
    
    x = multiply([inputs, feature])

    return x

def attention_block(input_tensor):
    channel_attention = spatial_pooling_block(input_tensor)

    # Compute the spatial attention
    spatial_attention = Conv2D(filters=1, kernel_size=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), activation='sigmoid')(channel_attention)
    channel_attention = multiply([channel_attention, spatial_attention])

    # Output the channel-spatial attention block
    output_tensor = add([channel_attention, input_tensor])
    return output_tensor


# CBAM Attention Block

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer=tf.keras.initializers.HeNormal(seed=2023),
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer=tf.keras.initializers.HeNormal(seed=2023),
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
    
    #avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    avg_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(cbam_feature)
    #assert avg_pool.shape[-1] == 1
    #max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(cbam_feature)
    #assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    #assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])


# SE Attention Block

def squeeze_excite_block(input, ratio=8, name=None):

    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)


    se = GlobalMaxPooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se], name=name)
    return x


def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='gelu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters,(num_row,num_col),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    x = tf.keras.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
    if(activation == None):
        return x

    #x = tfa.activations.gelu(x, approximate=True)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return x


def conv2d_bn(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    return x

 
def iterLBlock(x, filters, name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj = filters//8, filters//8, filters//2, filters//8, filters//4, filters//8

    conv_1x1 = initial_conv2d_bn(x, filters_1x1, 1, 1, padding='same', activation='selu')
    conv_3x3 = conv2d_bn(x, filters_3x3, 3, 3)
    conv_5x5 = conv2d_bn(x, filters_5x5, 5, 5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = initial_conv2d_bn(pool_proj, filters_pool_proj, 1, 1)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    output = tf.keras.layers.GroupNormalization(groups=filters, axis=channel_axis)(output)
    output = tf.keras.layers.LeakyReLU(alpha=0.02)(output)
    return output


def decoder_block(inputs, skip, filters):
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    x = iterLBlock(x, filters)
    x = squeeze_excite_block(x)
    return x


#['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block6_1_conv']


def SandBoilNet_SE(input_filters, height, width, n_channels):
     #inputs = Input((height, width, n_channels), name = "input_image")
    filters = input_filters
    model_input = Input(shape=(height, width, n_channels))
    """ Pretrained resnet"""
    tf.keras.backend.clear_session()
    base_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_tensor=model_input, pooling=max)
    print("Number of layers in the base model: ", len(base_model.layers))
 

    base_model.trainable = True
    
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    for i, layer in enumerate(base_model.layers[:-48]):
        layer.trainable = False

# Iteration 2
    """ Encoder """
    s11 = (base_model.get_layer("input_1").output)
    s11 =  initial_conv2d_bn(s11, filters, 3, 3)
    s11 =  iterLBlock(s11, filters)
    s11 = squeeze_excite_block(s11)

    
    s21 = (base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    pca2 = PCALayer(64)(s21)
    s21 = squeeze_excite_block(s21)
    s21 = add([s21, pca2])
    s21 = iterLBlock(s21, filters*2)
    #s21 = tf.keras.layers.SpatialDropout2D(rate = 0.06)(s21)
    
    s31 = (base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    pca3 = PCALayer(64)(s31)
    s31 = squeeze_excite_block(s31)
    s31 = add([s31, pca3])
    s31 = iterLBlock(s31, filters*2)
    s31 = tf.keras.layers.SpatialDropout2D(rate = 0.02)(s31)

    
    s41 = (base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    pca4 = PCALayer(128)(s41)
    s41 = squeeze_excite_block(s41)
    s41 = add([s41, pca4])
    s41 = iterLBlock(s41, filters*2)
    s41 = tf.keras.layers.SpatialDropout2D(rate = 0.02)(s41)
    

    """ Bridge """
    b11 = (base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    pcb11 = PCALayer(256)(b11)
    b11 = squeeze_excite_block(b11)
    b11 = add([b11, pcb11])
    b11= iterLBlock(b11, filters*4)
    b11 = tf.keras.layers.SpatialDropout2D(rate = 0.04)(b11)


    """ Decoder """
    d11 = decoder_block(b11, s41, filters*4)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*2)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*1)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters//2)                          ## (256 x 256)
    #d41 = conv2d_bn(d41, filters//2, 3,3)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="SandBoilNet_SE")

    return model



def main():

# Define the model

    model = SandBoilNet_SE(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main() 