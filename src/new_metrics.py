import tensorflow as tf

def new_mIoU(y_true, y_pred):
    mIoU = tf.keras.metrics.MeanIoU(num_classes=8)
    y_pred = tf.argmax(y_pred,-1)
    mIoU.update_state(y_true, y_pred)
    return mIoU.result()

def my_mIoU(y_true, y_pred):
    return tf.py_function(new_mIoU, inp=[y_true,y_pred],Tout=tf.float32)