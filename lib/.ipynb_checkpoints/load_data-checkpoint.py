# Method to load data; both images and their masks
import os
import numpy as np
import cv2
from tqdm import tqdm
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# Get and resize train images and masks
def get_data(ids, path, im_height, im_width, train):
    
    #ids = sorted(next(os.walk(path + "/images"))[2])
    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    print(X.shape)

    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
        
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        id_ = id_.split('/')[-1]
        # Load images
        img = cv2.imread(path + '/images/'+ '{}'.format(id_), cv2.IMREAD_COLOR)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 3), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = cv2.imread(path + '/masks/' + '{}'.format(id_.split('.')[0]+'.png'), cv2.IMREAD_GRAYSCALE)
            mask = img_to_array(mask)
            mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Save images
        #print(X[n, ..., 0].shape)
        #print(y[n].shape)
        X[n] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        print(X.shape)
        return X, y
    else:
        return X


def get_train_test_augmented(X, Y, validation_split=0.2, batch_size=4, seed=2022):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1-validation_split, test_size=validation_split, random_state=seed)

	img_data_gen_args = dict(rotation_range=45.,
							width_shift_range=0.2,
							height_shift_range=0.2,
							shear_range=0.2,
							zoom_range=0.2,
							horizontal_flip=True,
							vertical_flip=True,
							brightness_range=[0.2, 0.8])

	mask_data_gen_args = dict(preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype),
							rotation_range=45.,
							width_shift_range=0.2,
							height_shift_range=0.2,
							shear_range=0.2,
							zoom_range=0.2,
							horizontal_flip=True,
							vertical_flip=True,
							brightness_range=[0.2, 0.8]) 

	# Train data, provide the same seed and keyword arguments to the fit and flow methods
	X_datagen = ImageDataGenerator()
	Y_datagen = ImageDataGenerator()

	X_datagen.fit(X_train, augment=False, seed=seed)
	Y_datagen.fit(Y_train, augment=False, seed=seed)

	X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
	Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)


	# Test data, no data augmentation, but we create a generator anyway
	X_datagen_val = ImageDataGenerator()
	Y_datagen_val = ImageDataGenerator()

	X_datagen_val.fit(X_test, augment=False, seed=seed)
	Y_datagen_val.fit(Y_test, augment=False, seed=seed)

	X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=False, seed=seed)
	Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=False, seed=seed)


	# combine generators into one which yields image and masks
	train_generator = zip(X_train_augmented, Y_train_augmented)
	test_generator = zip(X_test_augmented, Y_test_augmented)

	return train_generator, test_generator
	
