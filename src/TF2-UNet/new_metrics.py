import tensorflow as tf

def new_mIoU():
    tf.config.run_functions_eagerly(True)
    @tf.function
    def get_mIoU(y_true, y_pred):
        mIoU = tf.keras.metrics.MeanIoU(num_classes=8)
        y_pred = tf.argmax(y_pred,-1)
        mIoU.update_state(y_true, y_pred)
        return mIoU.result()
    return get_mIoU
