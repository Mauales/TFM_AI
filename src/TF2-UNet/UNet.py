import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate

def build_model(num_classes = 8, input_size =(256,256,3)):
    OUTPUT_CHANNELS = num_classes
    dropout_rate = 0.5

    initializer = 'he_normal'

    # -- Encoder -- #
    # Block encoder 1
    inputs = Input(shape=input_size)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

    # Block encoder 2
    max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_2)
    conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_2)

    # Block  encoder 3
    max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_3)
    conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)

    # Block  encoder 4
    max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_4)
    conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_4)
    # -- Encoder -- #

    # ----------- #
    maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
    conv = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(maxpool)
    conv = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv)
    # ----------- #

    # -- Decoder -- #
    # Block decoder 1
    up_dec_1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv))
    merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis=3)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_1)
    conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)

    # Block decoder 2
    up_dec_2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv_dec_1))
    merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis=3)
    conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_2)
    conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)

    # Block decoder 3
    up_dec_3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv_dec_2))
    merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis=3)
    conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_3)
    conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)

    # Block decoder 4
    up_dec_4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(
        UpSampling2D(size=(2, 2))(conv_dec_3))
    merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis=3)
    conv_dec_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_4)
    conv_dec_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
    conv_dec_4 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
    # -- Dencoder -- #

    output = Conv2D(OUTPUT_CHANNELS, 1, activation='softmax')(conv_dec_4)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model