import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset
from new_metrics import *
from cadis_visualization import *

IMG_SIZE = (128,128)

def read_image(path):
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, IMG_SIZE)
    x = x / 255.0
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    x, _, _ = remap_experiment1(x)
    x = cv2.resize(x, IMG_SIZE)
    x = x / 1.0
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    ## Dataset
    path = r"C:\Users\mauro\OneDrive\Escritorio\CaDISv2"
    batch_size = 16
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch_size)

    test_steps = (len(test_x) // batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'my_mIoU': my_mIoU}):
        model = tf.keras.models.load_model("files/model.h5")

    #model.evaluate(test_dataset, steps=test_steps)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        if i % 20 == 0:
            x = read_image(x)
            y = read_mask(y)
            y_pred = model.predict(np.expand_dims(x, axis=0))[0]
            y_pred = tf.argmax(y_pred,-1).numpy()
            plot = plot_experiment(x, y_pred, 1, False)
            plot.savefig("./files/test"+ str(i)+".png")