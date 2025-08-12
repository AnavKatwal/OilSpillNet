import random
import numpy as np
import tensorflow as tf
seed = 646
random.seed = seed
np.random.seed = seed
#tf.seed = seed

class DataGen(tf.keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=4, img_height=256, img_width=256):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        id_name = id_name.split("/")[-1]
        
        image_path = os.path.join(self.path, "images", id_name) 
        print(image_path)
        mask_path = os.path.join(self.path, "masks/",'{}'.format(id_name.split('.')[0]+'.png'))
        print(mask_path)
        dim = (self.img_width, self.img_height)
    
        ## Reading Image
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        #image = img_to_array(image)
        #image = resize(image, (self.img_height, self.img_width, 3), mode='constant', preserve_range=True)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
        mask = np.zeros((self.img_height, self.img_width, 1))
                                 
        _mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #_mask_image = img_to_array(_mask_image)
        #_mask_image = resize(_mask_image, (self.img_height, self.img_width, 1), mode='constant', preserve_range=True)
        _mask_image = cv2.resize(_mask_image, dim, interpolation = cv2.INTER_AREA)
        _mask_image = np.expand_dims(_mask_image, axis=-1)                         
        
        mask = np.maximum(mask, _mask_image)
                                 
        ## Normalizaing 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))