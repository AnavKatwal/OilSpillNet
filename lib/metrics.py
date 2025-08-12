import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

seed = 2023
from seed_utils import set_global_seed
set_global_seed(seed)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mcc_loss(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)

    tp = tf.reduce_sum(y_true * y_pred)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fp = tf.reduce_sum(y_pred * (1 - y_true))
    fn = tf.reduce_sum((1 - y_pred) * y_true)

    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + tf.keras.backend.epsilon())

    mcc = numerator / denominator
    return 1 - mcc


# def mcc_metric(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
    
#     tp = tf.reduce_sum(y_true * y_pred)
#     tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
#     fp = tf.reduce_sum(y_pred * (1 - y_true))
#     fn = tf.reduce_sum((1 - y_pred) * y_true)

#     numerator = tp * tn - fp * fn
#     denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + tf.keras.backend.epsilon())

#     mcc = numerator / denominator
#     return mcc

@tf.function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true = K.cast(K.flatten(y_true), 'float32')
    y_pred = K.cast(K.flatten(y_pred), 'float32')

    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# it generates dicecoefficient over entire batch because of axis=0
def dice_coefficient(y_true, y_pred):
    smooth = 1e-7
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f, axis=0)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f, axis=0) + K.sum(y_pred_f, axis=0) + smooth)
    
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


# def f1(y_true, y_pred):
#     smooth = 1e-7
    
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
    
#     true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)), axis=0)
#     possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)), axis=0)
#     predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)), axis=0)
    
#     precision = true_positives / (predicted_positives + smooth)
#     recall = true_positives / (possible_positives + smooth)
    
#     f1_score = 2 * ((precision * recall) / (precision + recall + smooth))
    
#     f1_score = tf.reduce_mean(f1_score)
#     precision = tf.reduce_mean(precision)
#     recall = tf.reduce_mean(recall)
#     return precision.numpy(), recall.numpy(), f1_score.numpy()


# def confusion(y_true, y_pred):
#     smooth=1
#     y_pred_pos = K.clip(y_pred, 0, 1)
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = K.clip(y_true, 0, 1)
#     y_neg = 1 - y_pos
#     tp = K.sum(y_pos * y_pred_pos)
#     fp = K.sum(y_neg * y_pred_pos)
#     fn = K.sum(y_pos * y_pred_neg)
#     prec = (tp + smooth)/(tp+fp+smooth)
#     recall = (tp+smooth)/(tp+fn+smooth)
#     return prec, recall

# def tp(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
#     return tp


# def tn(y_true, y_pred):
#     smooth = 1
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pred_neg = 1 - y_pred_pos
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_neg = 1 - y_pos
#     tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
#     return 


# def sensitivity(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     return true_positives / (possible_positives + K.epsilon())


# def specificity(y_true, y_pred):
#     true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
#     possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
#     return true_negatives / (possible_negatives + K.epsilon())


@tf.function
def jaccard(y_true, y_pred, smooth=1e-6):

    y_true = K.cast(K.flatten(y_true), 'float32')
    y_pred = K.cast(K.flatten(y_pred), 'float32')

    intersection = K.sum(y_true * y_pred)
    total = K.sum(y_true) + K.sum(y_pred)
    union = total - intersection

    return (intersection + smooth) / (union + smooth)

# def bce_dice_loss_old(y_true,  y_pred, smooth=1e-6):    
       
#     #flatten label and prediction tensors
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
    
#     BCE =  tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)   
#     dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
#     Dice_BCE = BCE + dice_loss
    
#     return Dice_BCE

def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Crossentropy Loss and Dice Loss function for semantic segmentation.
    :param y_true: ground truth mask.
    :param y_pred: predicted mask.
    :return: the combined loss.
    """
    # Binary Crossentropy Loss
    epsilon = K.epsilon()
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Dice Loss
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred) 
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth) 
    
    # Combine the two losses with equal weights
    combined_loss = (0.5 * bce) + (0.5 * dice_loss)
    
    return combined_loss

# def jacard_dice(Y_test, yp):
#     jacard = 0
#     dice = 0
#     smooth =  1.0
#     for i in range(len(Y_test)):
#         flat_pred = K.flatten(Y_test[i])
#         flat_label = K.flatten(yp[i])
        
#         intersection_i = K.sum(flat_label * flat_pred)
#         union_i = K.sum( flat_label + flat_pred - flat_label * flat_pred)
        
#         dice_i = (2. * intersection_i + smooth) / (K.sum(flat_label) + K.sum(flat_pred) + smooth)
#         jacard_i = intersection_i / union_i
        
#         jacard += jacard_i
#         dice += dice_i

#     jacard /= len(Y_test)
#     dice /= len(Y_test)
#     print(jacard.numpy())
#     print(dice.numpy())
    
#     return jacard.numpy(), dice.numpy()

def tversky(y_true, y_pred):
    epsilon = 1e-5
    smooth = 1
    alpha = 0.6
    beta = 0.6
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + beta*false_pos + smooth)

# def tversky_loss(y_true, y_pred):
#     return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.7 # higher value of gamma amplifies penalty on misclassified pixels or regions
    return K.pow((1-pt_1), gamma)


# def focal_dice_loss(y_true, y_pred, alpha=0.6, gamma=3):
#     """
#     Combined Focal Loss and Dice Loss function.
#     :param y_true: ground truth mask.
#     :param y_pred: predicted mask.
#     :param alpha: focal loss alpha parameter.
#     :param gamma: focal loss gamma parameter.
#     :return: the combined loss.
#     """
#     # Binary Crossentropy Loss
#     epsilon = K.epsilon()
#     bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
#     bce = K.expand_dims(bce, axis=-1)
    
#     # Dice Loss
#     smooth = 1.0
#     intersection = K.sum(y_true * y_pred)
#     dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
#     dice_loss = 1 - dice
    
#     # Focal Loss
#     p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
#     alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
#     modulating_factor = K.pow(1 - p_t, gamma)

#     focal_loss = K.mean(alpha_factor * modulating_factor * bce, axis=-1)
#     # Combine the two losses with equal weights
#     combined_loss = (0.5 * focal_loss) + (0.5 * dice_loss)
    
#     return combined_loss

# def iou_metric(y_true_in, y_pred_in):
#     labels = y_true_in
#     y_pred = y_pred_in

#     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))

#     intersection = temp1[0]

#     area_true = np.histogram(labels,bins=[0,0.5,1])[0]
#     area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
#     area_true = np.expand_dims(area_true, -1)
#     area_pred = np.expand_dims(area_pred, 0)

#     # Compute union
#     union = area_true + area_pred - intersection
  
#     # Exclude background from the analysis
#     intersection = intersection[1:,1:]
#     intersection[intersection == 0] = 1e-9
    
#     union = union[1:,1:]
#     union[union == 0] = 1e-9

#     iou = intersection / union
#     return iou

def iou_metric_batch(y_true_in, y_pred_in):
    y_pred_in = y_pred_in
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

# def tversky_metric_batch(y_true_in, y_pred_in):
#     y_pred_in = y_pred_in
#     batch_size = y_true_in.shape[0]
#     metric = []
#     for batch in range(batch_size):
#         value = tversky(y_true_in[batch], y_pred_in[batch])
#         metric.append(value)
#     return np.mean(metric)

# def specificity_metric_batch(y_true_in, y_pred_in):
#     y_pred_in = y_pred_in
#     batch_size = y_true_in.shape[0]
#     metric = []
#     for batch in range(batch_size):
#         value = specificity(y_true_in[batch], y_pred_in[batch])
#         metric.append(value)
#     return np.mean(metric)

# def sensitivity_metric_batch(y_true_in, y_pred_in):
#     y_pred_in = y_pred_in
#     batch_size = y_true_in.shape[0]
#     metric = []
#     for batch in range(batch_size):
#         value = sensitivity(y_true_in[batch], y_pred_in[batch])
#         metric.append(value)
#     return np.mean(metric)