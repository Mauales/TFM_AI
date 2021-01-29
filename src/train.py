import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
import model as basic_model
import model_pretrained as pret_model
from new_loss import *
from new_metrics import *

if __name__ == "__main__":
    ## Dataset
    n_classes = 8
    img_size = (128,128,3)
    path = r"C:\Users\mauro\OneDrive\Escritorio\CaDISv2"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    ## Hyperparameters
    batch = 32
    lr = 1e-4
    epochs = 20
    train_dataset = tf_dataset(train_x, train_y, batch, shuffle= True, augment= True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch)

    model = pret_model.build_model(n_classes,img_size)

    opt = tf.keras.optimizers.Adam(lr)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy', my_mIoU])

    callbacks = [
          ModelCheckpoint("files/model.h5"),
          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
          CSVLogger("files/data.csv"),
          TensorBoard(),
          EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
     ]

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    with tf.device('/GPU:0'):
        model.fit(train_dataset,
                  validation_data=valid_dataset,
                  epochs=epochs,
                  steps_per_epoch=train_steps,
                  validation_steps=valid_steps,
                  callbacks=callbacks)