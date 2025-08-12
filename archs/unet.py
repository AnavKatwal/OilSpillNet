import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers.advanced_activations import ELU, LeakyReLU
#from tensorflow.keras.utils.vis_utils import plot_model

# set seed
seed = 2023
from seed_utils import set_global_seed
set_global_seed(seed)



def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation(activation, name=name)(x)

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023), use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation(activation, name=name)(x)

    return x

def UNet(input_filters, height, width, n_channels):

    inputs = Input((height, width, n_channels))
    filters = input_filters

    block1 = conv2d_bn(inputs, filters, 3, 3, activation='relu', padding='same')
    pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
    

    block2 = conv2d_bn(pool1, filters*2, 3, 3, activation='relu', padding='same')
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    

    block3 = conv2d_bn(pool2, filters*4, 3, 3, activation='relu', padding='same')
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    

    block4 = conv2d_bn(pool3, filters*8, 3, 3, activation='relu', padding='same')
    pool4 = MaxPooling2D(pool_size=(2, 2))(block4)
    

    block5 = conv2d_bn(pool4, filters*16, 3, 3, activation='relu', padding='same')

    up6 = concatenate([Conv2DTranspose(
        filters*8, (2, 2), strides=(2, 2), padding='same')(block5), block4], axis=3)
    block6 = conv2d_bn(up6, filters*8, 3, 3, activation='relu', padding='same')

    up7 = concatenate([Conv2DTranspose(
        filters*4, (2, 2), strides=(2, 2), padding='same')(block6), block3], axis=3)
    block7 = conv2d_bn(up7, filters*4, 3, 3, activation='relu', padding='same')

    up8 = concatenate([Conv2DTranspose(
        filters*2, (2, 2), strides=(2, 2), padding='same')(block7), block2], axis=3)
    block8 = conv2d_bn(up8, filters*2, 3, 3, activation='relu', padding='same')

    up9 = concatenate([Conv2DTranspose(filters, (2, 2), strides=(
        2, 2), padding='same')(block8), block1], axis=3)
    block9 = conv2d_bn(up9, filters, 3, 3, activation='relu', padding='same')
    conv10 = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(block9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def main():

# Define the model

    model = UNet(32, 256, 256, 3)
    print(model.summary())



if __name__ == '__main__':
    main()
        