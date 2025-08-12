import random
import numpy as np
import cv2
import os
import tensorflow as tf

from seed_utils import set_global_seed

seed = 2023
set_global_seed(seed)

class OilSpillDataGen(tf.keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=4, img_height=256, img_width=256, **kwargs):
        super().__init__(**kwargs)
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.on_epoch_end()

    def __load__(self, id_name):
        ## Path
        id_name = id_name.split("/")[-1]
        
        # Assuming id_name is in the format 'image_X_Y.png'
        base_name = id_name.split('.')[0]  # This will be 'image_X_Y'
        image_number_Y = base_name.split('_')[-1] # this will extract 'Y'

        image_path = os.path.join(self.path, "images", f"{base_name}.jpg")
        mask_path = os.path.join(self.path, "masks", f"mask_{image_number_Y}.png")
        
        dim = (self.img_width, self.img_height)
        
        ## Reading Image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        ## Reading Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        
        ## Normalizing
        image = image / 255.0
        mask = mask / 255.0

        return image, mask
    
    def __getitem__(self, index):
        
        current_batch_size = self.batch_size

        if (index + 1) * self.batch_size > len(self.ids):
            current_batch_size = len(self.ids) - index * self.batch_size
        
        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        images, masks = [], []
        for id_name in files_batch:
            img, mask = self.__load__(id_name)
            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)
    
    def on_epoch_end(self):
        random.shuffle(self.ids)
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))