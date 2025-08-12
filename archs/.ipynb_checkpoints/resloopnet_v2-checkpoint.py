import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute
from tensorflow.keras.models import Model, model_from_json

def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='gelu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, (num_row, num_col), strides=strides, kernel_initializer="he_normal", padding=padding, use_bias=False, name=name)(x)
    x = tfa.layers.GroupNormalization(groups=filters//2, axis= channel_axis)(x)
    if(activation == None):
        return x

    x = tfa.activations.gelu(x, approximate=True)
    return x

def spatial_squeeze_excite_block(input):
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='glorot_uniform')(input)
    x = multiply([input, se])
    return x

def squeeze_excite_block(input, ratio=16):

    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalMaxPooling2D()(init)
    se = Reshape(se_shape)(se)
    
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid',  kernel_initializer='glorot_uniform', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def channel_spatial_squeeze_excite(input, ratio=16):
    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x

def conv2d_bn(x, filters, num_row, num_col):
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    x = initial_conv2d_bn(x, filters, num_row, num_col)
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

    x = tfa.layers.GroupNormalization(groups=filters//2, axis=channel_axis)(x)
    x = tfa.activations.gelu(x, approximate=True)
    x = Conv2D(pointwise_conv_filters, (1, 1),
                padding='same',
                kernel_initializer='he_normal',
                use_bias=False,
                strides=(1, 1))(x)

    x = tfa.layers.GroupNormalization(groups=filters//2, axis=channel_axis)(x)
    x = tfa.activations.gelu(x, approximate=True)

    if(SE == True):
        x = channel_spatial_squeeze_excite(x)
        return x

    return x
 
def attention_block(inputs, filters):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='glorot_uniform')(inputs)
    x = multiply([inputs, se])
    
    x = iterLBlock(x, filters)
    
    
    # Use global average pooling to reduce the spatial dimensions
    x_1 = GlobalAveragePooling2D()(x)
    x_1 = Reshape(se_shape)(x_1)
    x_2 = GlobalMaxPooling2D()(x)
    x_2 = Reshape(se_shape)(x_2)
    
    if K.image_data_format() == 'channels_first':
        x_1 = Permute((3, 1, 2))(x_1)
        x_2 = Permute((3, 1, 2))(x_2)
    
    
    # Use a fully connected layer to compute the attention weights
    x_1 = Dense(filters, activation='relu', kernel_initializer='he_normal', use_bias=False)(x_1)
    x_1 = Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform', use_bias=False)(x_1)
    
    x_2 = Dense(filters, activation='relu', kernel_initializer='he_normal', use_bias=False)(x_2)
    x_2 = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=False)(x_2)
    
    
    ## Use a reshape layer to resize the attention weights
   # x_1 = Reshape((1, 1, 1))(x_1)
   # x_2 = Reshape((1, 1, 1))(x_2)
    
    
    # Use a multiply layer to apply the attention weights to the input
    x_1 = multiply([inputs, x_1])
    x_2 = multiply([inputs, x_2])
    
    output = add([x_1, x_2])
    
    
    return output
    
    ## original v2 version have regular division of filters and no multiply operation
    
def iterLBlock(x, filters, name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    shortcut = x
    filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj = filters//8, filters//8, filters//2, filters//8, filters//4, filters//8
    conv_1x1 = initial_conv2d_bn(x, filters_1x1, 1, 1, padding='same', activation='relu')

    conv_3x3 = initial_conv2d_bn(x, filters_3x3_reduce, 1, 1, padding='same', activation='relu')
    conv_3x3 = conv2d_bn(conv_3x3, filters_3x3, 3, 3)
    

    conv_5x5 = initial_conv2d_bn(x, filters_5x5_reduce, 1, 1, padding='same', activation='relu')
    conv_5x5 = conv2d_bn(conv_5x5, filters_5x5, 3, 3)
    conv_5x5_3x1 = initial_conv2d_bn(conv_5x5, filters_5x5//2, 3,1, padding='same', activation='relu')
    conv_5x5_1x3 = initial_conv2d_bn(conv_5x5, filters_5x5//2, 1,3, padding='same', activation='relu')

    
    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = conv2d_bn(pool_proj, filters_pool_proj, 1, 1)
    
    output = concatenate([conv_1x1, conv_3x3, conv_5x5_3x1, conv_5x5_1x3, pool_proj], axis=3, name=name)
    output = tfa.layers.GroupNormalization(groups=filters//2, axis=channel_axis)(output)
    output = tfa.activations.gelu(output, approximate=True)
    return output

def concat_feature_rank(encoder, decoder, filters):
    feature_vector = iterLBlock(encoder, filters*2)
    feature_vector = concatenate([feature_vector, decoder])
    feature_vector = depthwise_convblock(feature_vector, filters, 3,3, depth_multiplier=1, SE=True)
    return feature_vector

def decoder_block(inputs, skip, filters):
    #x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    x = iterLBlock(x, filters)
    return x

#['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block6_1_conv']


def ResLoopNet_V2(input_filters, height, width, n_channels):
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



    #for i, layer in enumerate(base_model.layers):
     #   layer.trainable = False


    # Iteration 1
    s1 = channel_spatial_squeeze_excite(base_model.get_layer("input_1").output)                ## (256 x 256)
    s1 = iterLBlock(s1, filters*1)
    s2 = channel_spatial_squeeze_excite(base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    s2 = iterLBlock(s2, filters*2)
    s3 = channel_spatial_squeeze_excite(base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    s3 = iterLBlock(s3, filters*4)

    """ Bridge """
    b1 = channel_spatial_squeeze_excite(base_model.get_layer("conv3_block4_1_conv").output)   ## (32 x 32)
    b1 = attention_block(b1, filters*8)

    """ Decoder """
    d1 = decoder_block(b1, s3, filters*4)                         ## (64 x 64)
    d2 = decoder_block(d1, s2, filters*2)                         ## (128 x 128)
    d3 = decoder_block(d2, s1, filters*1)                         ## (256 x 256)


    # Iteration 2
    """ Encoder """
    s11 = channel_spatial_squeeze_excite(base_model.get_layer("input_1").output)
    s11 = concat_feature_rank(s11, d3, filters*1)                ## (256 x 256)
    s21 = channel_spatial_squeeze_excite(base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    s21 = concat_feature_rank(s21, d2, filters*2)
    s31 = channel_spatial_squeeze_excite(base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    s31 = concat_feature_rank(s31, d1, filters*4)
    s41 = channel_spatial_squeeze_excite(base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    s41 = concat_feature_rank(s41, b1, filters*8)

    """ Bridge """
    b11 = channel_spatial_squeeze_excite(base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    b11 = attention_block(b11, filters*16)
    #depthwise_convblock(b11, filters*16, 3,3, depth_multiplier=1, SE=True)

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*8)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*4)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*2)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters*1)                          ## (256 x 256)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="ResLoopNet_V2")

    return model



def main():

# Define the model

    model = ResLoopNet_V2(16, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main()