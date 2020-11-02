
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import math
import numpy as np

def multi_focal_loss(gamma, alpha, y_true, y_pred):

    epsilon = 1.e-7
    #gamma=5.
    alpha = tf.constant(alpha, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    focal_loss = tf.multiply(tf.multiply(weight, ce), alpha_t)
    #print('f1:    ', fl.shape)
    focal_loss = tf.reduce_mean(focal_loss, axis=1)
    return focal_loss

def confusion(ground_truth, predictions):
    np.seterr(invalid='ignore')
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    sensitivity = (tp)/(tp+fn)
    specificty = (tn)/(tn+fp)

    if math.isnan(accuracy) == True: accuracy = 0
    if math.isnan(sensitivity) == True: sensitivity = 0
    if math.isnan(specificty) == True: specificty = 0

    return accuracy,sensitivity,specificty