import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU
from csv import writer
from lib.metrics import iou_metric_batch

seed = 2023
from seed_utils import set_global_seed
set_global_seed(seed)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_csv_file(path_to_file):
    path = Path(path_to_file)
    if not path.is_file():
        list_names = ['Models', 'Binary Acc.', 'BA', 'mIoU', 'IoU', 'DC', 'Precision', 'Recall', 'Macro_F1', 'Micro_F1', 'MCC', 'Sensitivity', 'Specificity']
        with open(path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(list_names)


def evaluate_model(yp, Y_test):
    flat_pred = K.flatten(yp)
    flat_label = K.flatten(Y_test)

    binary_acc = BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
    mean_iou = MeanIoU(num_classes=2)

    binary_acc.update_state(flat_label, flat_pred)
    r1 = binary_acc.result().numpy()

    r3 = mean_iou.update_state(flat_label,flat_pred)
    r3 = mean_iou.result().numpy()

    return r1, r3


def micro_f1(y_true, y_pred):
    smooth = 1e-7

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + smooth)
    recall = true_positives / (possible_positives + smooth)

    f1_score = (2* (precision * recall) / (precision + recall + smooth))

    return precision, recall, f1_score



def macro_f1(avg_precision, avg_recall):
    macro_f1 = (2 *  avg_precision * avg_recall ) / (avg_precision + avg_recall)
   
    return macro_f1

def matthews_correlation(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    tp = K.sum(y_true * y_pred)
    tn = K.sum((1 - y_true) * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + K.epsilon())

    mcc = numerator / (denominator + K.epsilon())
    return mcc.numpy()



# dice coefficient and iou adapted from https://github.com/shyamfec/NucleiSegNet/blob/https/github.com/DevikalyanDas/NucleiSegnet-Paper-with-Codemaster/scores_comp.py
def dice_coefficient(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))

    intersection = temp1[0]

    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    dice_m = (2*intersection) / union
    return dice_m


def iou_score(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))

    intersection = temp1[0]

    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    iou = intersection / union
    return iou

def mean_iou(y_true, y_pred):
    intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    union = K.sum(K.round(K.clip(y_true + y_pred, 0, 1))) + K.epsilon()
    mean_iou = intersection / union
    return mean_iou


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def balanced_accuracy(y_true, y_pred):
    sensit = sensitivity(y_true, y_pred)
    specifi = specificity(y_true, y_pred)
    BA = (sensit + specifi) / 2
    return sensit, specifi, BA


def compute_metrics(model, X_test, Y_test, threshold):
    precisions = []
    recalls = []
    micro_f1_scores = []
    macro_f1_scores = []
    mcc_scores = []
    dice_scores = []
    iou_scores = []
    mean_iou_scores = []
    binary_acc_scores = []
    bal_acc_scores = []
    sensitivity_scores = []
    specificity_scores = []
    

    for i in range(len(X_test)):
        image = X_test[i]
        label = Y_test[i]

        if len(image.shape) != 4:
            # since there is single image for prediction
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)

        prediction = model.predict(image, verbose=0)
        #prediction = np.round(prediction,0)
        prediction = (prediction > threshold).astype('float32')

        p_batch, r_batch, micro_f1_score = micro_f1(label, prediction)
        mcc_score = matthews_correlation(label, prediction)
        dice_batch = dice_coefficient(label, prediction)
        iou_batch = iou_score(label, prediction)
        sensitivity, specificity, bal_acc = balanced_accuracy(label, prediction)
        binary_acc_batch, mean_iou_batch = evaluate_model(prediction, label)

        precisions.append(p_batch)
        recalls.append(r_batch)
        micro_f1_scores.append(micro_f1_score)
        mcc_scores.append(mcc_score)
        dice_scores.append(dice_batch)
        iou_scores.append(iou_batch)
        bal_acc_scores.append(bal_acc)
        mean_iou_scores.append(mean_iou_batch)
        binary_acc_scores.append(binary_acc_batch)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)

    average_precision = tf.reduce_mean(precisions).numpy()
    average_recall = tf.reduce_mean(recalls).numpy()
    average_macro_f1 = macro_f1(average_precision, average_recall)
    average_micro_f1 = tf.reduce_mean(micro_f1_scores).numpy()
    average_mcc = tf.reduce_mean(mcc_scores).numpy()
    average_dice = tf.reduce_mean(dice_scores).numpy()
    average_iou = tf.reduce_mean(iou_scores).numpy()
    average_mean_iou = tf.reduce_mean(mean_iou_scores).numpy()
    average_binary_acc = tf.reduce_mean(binary_acc_scores).numpy()
    average_balanced_acc = tf.reduce_mean(bal_acc_scores).numpy()
    average_sensitivity = tf.reduce_mean(sensitivity_scores).numpy()
    average_specificity = tf.reduce_mean(specificity_scores).numpy()
    
    #iou_all = iou_metric_batch(Y_test, yp)

    return (
        average_precision,
        average_recall,
        average_macro_f1,
        average_micro_f1,
        average_mcc,
        average_dice,
        average_iou,
        average_mean_iou,
        average_binary_acc,
        average_balanced_acc,
        average_sensitivity,
        average_specificity
    )


def test_model(model, X_test, Y_test, experiment_name, filename, threshold):
    result_folder = "../results/"
    create_dir(result_folder)

    csv_filename = "../results/" + str(filename) +".csv"
    create_csv_file(csv_filename)

    results = []

    (
        precision,
        recall,
        macro_f1,
        micro_f1,
        mcc,
        dice,
        iou,
        mean_iou,
        binary_acc,
        bal_acc,
        sensitivity,
        specificity,
    ) = compute_metrics(model, X_test, Y_test, threshold)

    results.append(experiment_name)
    results.append("{:.2%}".format(binary_acc))
    results.append("{:.2%}".format(bal_acc))
    results.append("{:.2%}".format(mean_iou))
    results.append("{:.2%}".format(iou))
    results.append("{:.2%}".format(dice))
    results.append("{:.2%}".format(precision))
    results.append("{:.2%}".format(recall))
    results.append("{:.2%}".format(macro_f1))
    results.append("{:.2%}".format(micro_f1))
    results.append("{:.2%}".format(mcc))
    results.append("{:.2%}".format(sensitivity))
    results.append("{:.2%}".format(specificity))
    

    with open(csv_filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)

    print('Test IoU Score (foreground):', iou)
    print('Test Dice Coefficient (foreground):', dice)
    print('Test Mean IoU (foreground):', mean_iou)
    print('Test Binary Accuracy:', binary_acc)
    print('Test Balanced Accuracy:', bal_acc)
    return results
