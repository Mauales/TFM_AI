import tensorflow as tf

def build_model(num_classes = 8, img_size =(256,256,3)):
    OUTPUT_CHANNELS = num_classes

    base_model = tf.keras.applications.MobileNetV2(input_shape=img_size, include_top=False)