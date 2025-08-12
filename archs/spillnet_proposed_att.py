import tensorflow as tf
# import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute, Add, Concatenate
from tensorflow.keras.models import Model, model_from_json

seed = 2023
from seed_utils import set_global_seed
set_global_seed(seed)


def spatial_pooling_block(inputs, ratio=4):
    
    #inputs = iterLBlock(inputs, filters)
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = inputs.shape[channel_axis]
    
    se_shape = (1, 1, filters)
    

    spp_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(inputs)
    spp_1 = layers.GlobalMaxPooling2D()(spp_1)
    spp_1 = Reshape(se_shape)(spp_1)
    spp_1 = Dense(filters, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=6446), use_bias=True, bias_initializer='zeros')(spp_1)
    #

    spp_2 = MaxPooling2D(pool_size=(4,4), strides=(4,4), padding='same')(inputs)    
    spp_2 = layers.GlobalMaxPooling2D()(spp_2)
    spp_2 = Reshape(se_shape)(spp_2)
    spp_2 = Dense(filters, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=6446), use_bias=True, bias_initializer='zeros')(spp_2)
    #print(f'shape of spp2 {spp_2.shape}')
        

    spp_3 = MaxPooling2D(pool_size=(8,8), strides=(8,8), padding='same')(inputs)
    spp_3 = layers.GlobalMaxPooling2D()(spp_3)
    spp_3 = Reshape(se_shape)(spp_3)
    spp_3 = Dense(filters, activation='relu',kernel_initializer=tf.keras.initializers.HeNormal(seed=6446), use_bias=True, bias_initializer='zeros')(spp_3)
    #print(f'shape of spp3 {spp_3.shape}')
    

    feature = Add()([spp_1,spp_2, spp_3])
    #feature = Dense(filters, activation='sigmoid',kernel_initializer=tf.keras.initializers.HeNormal(seed=6446), use_bias=True, bias_initializer='zeros')(feature)
    feature = Activation('sigmoid')(feature)
    
    x = multiply([inputs, feature])

    return x

def attention_block(input_tensor):
    # Compute the channel attention
    #avg_pool = K.mean(input_tensor, axis=(1, 2), keepdims=True)
    #max_pool = K.max(input_tensor, axis=(1, 2), keepdims=True)

    #channel_attention = Conv2D(filters=1, kernel_size=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=6446), activation='sigmoid')(add([avg_pool, max_pool]))
    #channel_attention = multiply([input_tensor, channel_attention])

    channel_attention = spatial_pooling_block(input_tensor)

    # Compute the spatial attention
    spatial_attention = Conv2D(filters=1, kernel_size=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed=6446), activation='sigmoid')(channel_attention)
    channel_attention = multiply([channel_attention, spatial_attention])

    # Output the channel-spatial attention block
    output_tensor = add([channel_attention, input_tensor])
    return output_tensor


# def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='gelu', name=None):
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

#     x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
#     x = tfa.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
#     if(activation == None):
#         return x
#     x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
#     return x

def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='gelu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters,(num_row, num_col),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    x = tf.keras.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
    # if(activation == None):
    #     return x
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    # x = BatchNormalization(axis=3, scale=False)(x)
    # x = Activation(activation, name=name)(x)
    return x

def depthwise_conv(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = SeparableConv2D(filters, (num_row, num_col), padding='same',kernel_regularizer=None,kernel_initializer=tf.keras.initializers.HeNormal(seed=6446))(x)
    x = tf.keras.layers.GroupNormalization(groups=filters, axis= channel_axis)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return x

 
def iterLBlock(x, filters, name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj = filters//8, filters//8, filters//2, filters//8, filters//4, filters//8

    conv_1x1 = initial_conv2d_bn(x, filters_1x1, 1, 1)

    #conv_3x3 = initial_conv2d_bn(x, filters_3x3, 1, 1, padding='same', activation='selu')
    conv_3x3 =  depthwise_conv(x, filters_3x3, 3, 3)
    conv_3x3 =  initial_conv2d_bn(conv_3x3, filters_3x3, 3, 3)

    #conv_5x5 = initial_conv2d_bn(x, filters_5x5_reduce, 1, 1, padding='same', activation='selu')
    conv_5x5 = depthwise_conv(x, filters_5x5, 5, 5)
    conv_5x5 = initial_conv2d_bn(conv_5x5, filters_5x5, 5, 5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = initial_conv2d_bn(pool_proj, filters_pool_proj, 1, 1)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    output = tf.keras.layers.GroupNormalization(groups=filters, axis=channel_axis)(output)
    output = tf.keras.layers.LeakyReLU(alpha=0.02)(output)
    return output


def conv2d_bn(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    return x

def decoder_block(inputs, skip, filters):
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    x = iterLBlock(x, filters)
    x = attention_block(x)
    return x


def Boilnet_Proposed_Att(input_filters, height, width, n_channels):
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

    """ Encoder """
    s11 = attention_block(base_model.get_layer("input_1").output)
    s21 = attention_block(base_model.get_layer("conv1_conv").output)    ## (128 x 128)
    s31 = attention_block(base_model.get_layer("conv2_block3_1_conv").output)    ## (64 x 64)
    s41 = attention_block(base_model.get_layer("conv3_block4_1_conv").output)    ## (32 x 32)

    """ Bridge """
    b11 = attention_block(base_model.get_layer("conv4_block6_1_conv").output)   ## (16 x 16)
    b11 = iterLBlock(b11, filters*8)
    

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*4)                         ## (32 x 32)
    d21 = decoder_block(d11, s31, filters*2)                         ## (64 x 64)
    d31 = decoder_block(d21, s21, filters*1)                         ## (128 x 128)
    d41 = decoder_block(d31, s11, filters//2)                          ## (256 x 256)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="Boilnet_Proposed_Att")

    return model



def main():

# Define the model

    model = Boilnet_Proposed_Att(32, 256, 256, 3)
    #mnet = MobileNetV2(input_tensor=inputs, input_shape = (256, 256, 3), include_top=False, weights="imagenet", alpha=1)

    print(model.summary())



if __name__ == '__main__':
    main()