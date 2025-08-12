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

    #x = tfa.activations.gelu(x, approximate=True)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return x

def attention_block(inputs, ratio=4):
    
    #inputs = iterLBlock(inputs, filters)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    #se = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal')(inputs)
    #inputs_1 = initial_conv2d_bn(inputs, filters, num_row, num_col)
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(inputs)
    
    x_3 = multiply([inputs, se])

    filters = x_3.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    # Use global average pooling to reduce the spatial dimensions
    x_1 = GlobalAveragePooling2D()(x_3)
    x_1 = Reshape(se_shape)(x_1)
    x_2 = GlobalMaxPooling2D()(x_3)
    x_2 = Reshape(se_shape)(x_2)
    
    attention_weights_1 = Conv2D(filters, (1, 1), activation='relu', kernel_initializer='he_normal')(x_1)
    attention_weights_1 = Conv2D(filters, (1, 1), activation='sigmoid', use_bias=False)(attention_weights_1)
    attention_weights_2 = Conv2D(filters, (1, 1), activation='relu', kernel_initializer='he_normal')(x_2)
    attention_weights_2 = Conv2D(filters, (1, 1), activation='sigmoid', use_bias=False)(attention_weights_2)
    #print(f'shape of attention weights : {attention_weights_1.shape} and {attention_weights_2.shape}')
    if K.image_data_format() == 'channels_first':
        x_1 = Permute((3, 1, 2))(x_1)
        x_2 = Permute((3, 1, 2))(x_2)
    
 
    # Use a multiply layer to apply the attention weights to the input
    #x_1 = Reshape(se_shape)(attention_weights_1)
    #x_2 = Reshape(se_shape)(attention_weights_2)
    x_1 = multiply([x_3, x_1])
    x_2 = multiply([x_3, x_2])
    
    x = add([x_1, x_2])

    se_2 = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(x)
    
    x = multiply([x, se_2])
    #output = initial_conv2d_bn(x, filters, num_row, num_col)
    #output=iterLBlock(x, filters)
    return x

def spatial_squeeze_excite_block(input):
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input)

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
    se = Dense(filters, activation='sigmoid',  kernel_initializer='he_normal', use_bias=False)(se)

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
    #x = initial_conv2d_bn(x, filters, num_row, num_col)
    return x


def depthwise_convblock(inputs, filters, num_row, num_col, alpha=1, depth_multiplier=1, strides=(1,1), block_id=1, SE=False):
    ''' Depthwise Separable Convolution (DSC) layer
    Args:
        inputs: input tensor
        filters: number of output filters 
        (num_row, num_col): filter size
    Returns: a keras tensor
    References
    -   [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861v1.pdf) 
    '''
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

    conv_3x3 = initial_conv2d_bn(x, filters_3x3_reduce, 1, 1, padding='same', activation='selu')
    conv_3x3 = conv2d_bn(conv_3x3, filters_3x3, 3, 3)

    conv_5x5 = initial_conv2d_bn(x, filters_5x5_reduce, 1, 1, padding='same', activation='selu')
    conv_5x5 = conv2d_bn(conv_5x5, filters_5x5, 3, 3)
    conv_5x5_3x1 = initial_conv2d_bn(conv_5x5, filters_5x5//2, 3,1, padding='same', activation='selu')
    conv_5x5_1x3 = initial_conv2d_bn(conv_5x5, filters_5x5//2, 1,3, padding='same', activation='selu')

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = conv2d_bn(pool_proj, filters_pool_proj, 1, 1)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5_3x1, conv_5x5_1x3, pool_proj], axis=3, name=name)
    output = tfa.layers.GroupNormalization(groups=filters//2, axis=channel_axis)(output)
    #output = tfa.activations.gelu(output, approximate=True)
    output = tf.keras.layers.LeakyReLU(alpha=0.01)(output)
    return output


def encoder_block(x, filters, num_row, num_col):
    x = iterLBlock(x, filters)
    shortcut = x
    x = depthwise_convblock(x, filters, 3, 3, SE=True)
    output = add([shortcut, x])
    output = depthwise_convblock(output, filters, num_row, num_row, SE=False)
    return output

def decoder_block(inputs, skip, filters):
    #x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    x = iterLBlock(x, filters)
    shortcut = x
    output = depthwise_convblock(x, filters, 3, 3, SE=True)
    output = add([shortcut, x])
    output = depthwise_convblock(output, filters, 3, 3, SE=False)
    return x

#['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block6_1_conv']


def BestSoFar(input_filters, height, width, n_channels):
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


    # Iteration 2
    """ Encoder """
    s11 = attention_block(base_model.get_layer("input_1").output)
    s11 = encoder_block(s11, filters, 3, 3)               ## (256 x 256)
    #s11 = iterLBlock(s11, filters)
    s21 = attention_block(base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    s21 = encoder_block(s21, filters*2, 3, 3) 
    #s21 = iterLBlock(s21, filters*2)
    s31 = attention_block(base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    #s31 = iterLBlock(s31, filters*4)
    s31 = encoder_block(s31, filters*4, 3, 3) 
    s41 = attention_block(base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    #s41 = iterLBlock(s41, filters*8)
    s41 = encoder_block(s41, filters*8, 3, 3) 

    """ Bridge """
    b11 = attention_block(base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    b11 = encoder_block(b11, filters*16, 3, 3) 
    #b11= iterLBlock(b11, filters*16)
    #depthwise_convblock(b11, filters*16, 3,3, depth_multiplier=1, SE=True)

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*8)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*4)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*2)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters*1)                          ## (256 x 256)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="LastTryNet")

    return model



def main():

# Define the model

    model = LastTryNet(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main()