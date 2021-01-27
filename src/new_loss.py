import tensorflow as tf

def DiceLoss(y_true, y_pred, smooth = 1e-6):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), 8)
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - (numerator + smooth) / (denominator + smooth)

def IoULoss(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), 8)
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU