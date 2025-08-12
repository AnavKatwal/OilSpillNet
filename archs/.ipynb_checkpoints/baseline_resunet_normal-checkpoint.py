import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute
from tensorflow.keras.models import Model, model_from_json

def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, (num_row, num_col), strides=strides, kernel_initializer="he_normal", padding=padding, use_bias=False, name=name)(x)
    #x = tfa.layers.GroupNormalization(groups=filters//2, axis= channel_axis)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    #x = tfa.activations.gelu(x, approximate=True)
    x = Activation(activation)(x)

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

    #x = tfa.layers.GroupNormalization(groups=filters//2, axis=channel_axis)(x)
    #x = tfa.activations.gelu(x, approximate=True)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('elu')(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),
                padding='same',
                kernel_initializer='he_normal',
                use_bias=False,
                strides=(1, 1))(x)

    #x = tfa.layers.GroupNormalization(groups=filters//2, axis=channel_axis)(x)
    #x = tfa.activations.gelu(x, approximate=True)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation('elu')(x)

    if(SE == True):
        x = channel_spatial_squeeze_excite(x)
        return x

    return x
 

def decoder_block(inputs, skip, filters):
    #x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    #x = depthwise_convblock(x, filters, 3, 3) 
    x = conv2d_bn(x, filters, 3, 3)

    return x

#['conv1_conv', 'conv2_block3_1_conv', 'conv3_block4_1_conv', 'conv4_block6_1_conv']


def Baseline_ResUNet_Normal(input_filters, height, width, n_channels):
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


    # # Iteration 2
    # """ Encoder """
    # s11 = channel_spatial_squeeze_excite(base_model.get_layer("input_1").output)
    # s11 = conv2d_bn(s11, filters, 3, 3)               ## (256 x 256)
    # s21 = channel_spatial_squeeze_excite(base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    # s21 = conv2d_bn(s21, filters*2, 3, 3) 
    # s31 = channel_spatial_squeeze_excite(base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    # s31 = conv2d_bn(s31, filters*4, 3, 3) 
    # s41 = channel_spatial_squeeze_excite(base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)
    # s41 = conv2d_bn(s41, filters*8, 3, 3) 

    # """ Bridge """
    # b11 = channel_spatial_squeeze_excite(base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    # b11 = conv2d_bn(b11, filters*16, 3, 3) 
    #depthwise_convblock(b11, filters*16, 3,3, depth_multiplier=1, SE=True)

    # Iteration 2
    """ Encoder """
    s11 = base_model.get_layer("input_1").output               ## (256 x 256)
    s21 = base_model.get_layer("conv1_conv").output    ## (128 x 128)
    
    s31 = base_model.get_layer("conv2_block3_1_conv").output   ## (64 x 64)
    
    s41 = base_model.get_layer("conv3_block4_1_conv").output    ## (32 x 32)
    

    """ Bridge """
    b11 = base_model.get_layer("conv4_block6_1_conv").output   ## (16 x 16)
    

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*8)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*4)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*2)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters*1)                          ## (256 x 256)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="Baseline_ResUNet_Normal")

    return model



def main():

# Define the model

    model = Baseline_ResUNet_Normal(16, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main()