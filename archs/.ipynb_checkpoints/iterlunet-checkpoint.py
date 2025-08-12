from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, model_from_json


def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
	'''InitialBlock
	Args: 
		x: input tensor
		filters: number of output filters
		(num_row, num_col): filter size
	Returns: a keras tensor
	'''
	x = Conv2D(filters, (num_row, num_col), strides=strides, kernel_initializer="he_normal", padding=padding, use_bias=False)(x)
	x = BatchNormalization(axis=3, scale=False)(x)

	if(activation == None):
		return x

	x = Activation(activation, name=name)(x)
	return x

def spatial_squeeze_excite_block(input):
	''' sSE Block
	Args:
		input: input tensor
	Returns: a keras tensor
	References
	-   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
	'''
	se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(input)

	x = multiply([input, se])
	return x

def squeeze_excite_block(input, ratio=16):
	''' SE Block
	Args:
		input: input tensor
	Returns: a keras tensor
	References
	-   [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
'''
	init = input
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	filters = init.shape[channel_axis]
	se_shape = (1, 1, filters)


	se = GlobalAveragePooling2D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
	se = Dense(filters, activation='sigmoid',  kernel_initializer='he_normal', use_bias=False)(se)

	if K.image_data_format() == 'channels_first':
		se = Permute((3, 1, 2))(se)

	x = multiply([init, se])
	return x

def channel_spatial_squeeze_excite(input, ratio=16):
	''' csSE Block
	Args:
		input: input tensor
		filters: number of output filters
	Returns: a keras tensor
	References
	-   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
	-   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
	'''

	cse = squeeze_excite_block(input, ratio)
	sse = spatial_squeeze_excite_block(input)

	x = add([cse, sse])
	return x

def conv2d_bn(x, filters, num_row, num_col):
	x = initial_conv2d_bn(x, filters, num_row, num_col)
	return x


def depthwise_convblock(inputs, filters, num_row, num_col, alpha=1, depth_multiplier=1, strides=(1,1), block_id=1, SE=False):
	''' Depthwise Separable Convolution (DSC) layer
	Args:
		inputs: input tensor
		filters: number of output filters 
		(num_row, num_col): filter size
	Returns: a keras tensor
	References
	-	[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861v1.pdf) 
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
		x = channel_spatial_squeeze_excite(x)
		return x

	return x


def iterLBlock(x, filters, name=None):
	''' Iterative Loop Block (IterLBlock)
	Args:
		inputs: input tensor
		filters: number of output filters
	Returns: a keras tensor 
	'''
	shortcut = x
	filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj = filters//16, filters//16, filters//4, filters//16, filters//8, filters//16
	conv_1x1 = initial_conv2d_bn(x, filters_1x1, 1, 1, padding='same', activation='relu')

	conv_3x3 = initial_conv2d_bn(x, filters_3x3_reduce, 1, 1, padding='same', activation='relu')
	conv_3x1 = conv2d_bn(conv_3x3, filters_3x3//2, 3,1)
	conv_1x3 = conv2d_bn(conv_3x3, filters_3x3//2, 1,3)

	conv_5x5 = initial_conv2d_bn(x, filters_5x5_reduce, 1, 1, padding='same', activation='relu')
	conv_5x5 = conv2d_bn(conv_5x5, filters_5x5, 3,3)
	conv_5x5_3x1 = conv2d_bn(conv_5x5, filters_5x5//2, 3,1)
	conv_5x5_1x3 = conv2d_bn(conv_5x5, filters_5x5//2, 1,3)

	pool_proj = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	pool_proj = initial_conv2d_bn(pool_proj, filters_pool_proj, 1, 1, padding='same', activation='relu')

	output = squeeze_excite_block(concatenate([conv_1x1, conv_3x1, conv_1x3, conv_5x5_3x1, conv_5x5_1x3, pool_proj], axis=3, name=name))
	output = BatchNormalization(axis=3)(output)
	output = Activation('relu')(output)
	return output

def IterLUNet(input_filters, height, width, n_channels):
	''' Iterative Loop U-Net (IterLUNet) - Deep Learning Architecture for Pixel-Wise Crack Detection in Levee Systems 
	Args:
		input_filters: number of input filters in current keras tensor [16, 32, 64, 128, 256, 512]
		(height, width, n_channels): input images with height, width and number of channels (3 in case of RGB images)
	'''

	inputs = Input((height, width, n_channels))
	filters = input_filters


	# Iteration 1
	block1 = initial_conv2d_bn(inputs, filters*1, 3, 3, padding='same', activation='relu')
	pool1 = MaxPooling2D(pool_size=(2, 2))(block1)

	bottleneck1 = iterLBlock(pool1, filters*2, name = 'iterLBlock1')


	up1 = concatenate([Conv2DTranspose(
		filters*1, (2, 2), strides=(2, 2), padding='same')(bottleneck1), block1], axis=3)
	level1 = iterLBlock(up1, filters*1,name = 'iterLBlock2' )
    print(f'shape of level 1 {level1.shape})


	# Iteration 2

	encoder2 = initial_conv2d_bn(inputs, filters*1, 3, 3, padding='same', activation='relu')
	inter1 = concatenate([encoder2, level1], axis=3)
	inter1 = depthwise_convblock(inter1, filters, 3,3, depth_multiplier=1, SE=True)
	block2 = iterLBlock(inter1, filters*2, name = 'iterLBlock3')
	pool2 = MaxPooling2D(pool_size=(2, 2))(block2)


	inter21 = concatenate([pool2, bottleneck1], axis=3)
	inter21 = depthwise_convblock(inter21, filters*2, 3,3, depth_multiplier=1, SE=True)
	block21 = iterLBlock(inter21, filters*4,name = 'iterLBlock4')
	pool21 = MaxPooling2D(pool_size=(2, 2))(block21)

	bottleneck2 = iterLBlock(pool21, filters*8, name = 'iterLBlock5')

	up21 = concatenate([Conv2DTranspose(
		filters*4, (2, 2), strides=(2, 2), padding='same')(bottleneck2), block21], axis=3)
	block22 = iterLBlock(up21, filters*4, name = 'iterLBlock6')


	up22 = concatenate([Conv2DTranspose(
		filters*2, (2, 2), strides=(2, 2), padding='same')(block22), block2], axis=3)
	                                   
	level2 = iterLBlock(up22, filters*2, name = 'iterLBlock7')
print(f'shape of level 2 {level2.shape})



	# Iteration 3
	encoder3 = initial_conv2d_bn(inputs, filters*2, 3, 3, padding='same', activation='relu')
	inter3 = concatenate([encoder3, level2], axis=3)
	inter3 = depthwise_convblock(inter3, filters*2, 3,3, depth_multiplier=1, SE=True)
	block3 = iterLBlock(inter3, filters*2, name = 'iterLBlock8')
	pool3 = MaxPooling2D(pool_size=(2, 2))(block3)


	inter31 = concatenate([pool3, block22], axis=3)
	inter31 = depthwise_convblock(inter31, filters*4, 3,3, depth_multiplier=1, SE=True)
	block31 = iterLBlock(inter31, filters*4, name = 'iterLBlock9')
	pool31 = MaxPooling2D(pool_size=(2, 2))(block31)


	inter32 = concatenate([pool31, bottleneck2], axis=3)
	inter32 = depthwise_convblock(inter32, filters*8, 3,3, depth_multiplier=1, SE=True)
	block32 = iterLBlock(inter32, filters*8, name = 'iterLBlock10')
	pool32 = MaxPooling2D(pool_size=(2, 2))(block32)

	bottleneck3 = iterLBlock(pool32, filters*16, name = 'iterLBlock11')

	up3 = concatenate([Conv2DTranspose(
		filters*8, (2, 2), strides=(2, 2), padding='same')(bottleneck3), block32], axis=3)
	block33 = iterLBlock(up3, filters*8, name = 'iterLBlock12')


	up31 = concatenate([Conv2DTranspose(
		filters*4, (2, 2), strides=(2, 2), padding='same')(block33), block31], axis=3)
	block34 = iterLBlock(up31, filters*4, name = 'iterLBlock13')


	up32 = concatenate([Conv2DTranspose(
		filters*2, (2, 2), strides=(2, 2), padding='same')(block34), block3], axis=3)
	level3 = iterLBlock(up32, filters*2, name = 'iterLBlock14')
    print(f'shape of level 3 {level3.shape})

	conv10 = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(level3)
	model = Model(inputs=[inputs], outputs=[conv10])
	return model

def main():

# Define the model

	model = IterLUNet(64, 256, 256, 3)
	#print(model.summary())



if __name__ == '__main__':
	main()