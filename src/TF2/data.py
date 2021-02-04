from cadis_visualization import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def load_params(images_folder):
    """
    Loads the splits settings from the specified folder
    :param images_folder: Path to the splits.txt file
    :return: the split information in a dictionary where each key is a set (e.g. Training, Validation, Test)
    """
    f = open(images_folder + "\splits.txt", "r")
    splits = {}
    clase = ""
    for line in f:
        if line.startswith("#"):
            clase = line.split("#")[1].split(":")[0].strip()
            splits[clase] = []
        elif line.strip() != "":
            splits[clase].append(images_folder + "\\" + line.strip())
    return splits


def get_ImagesBySet(trainingSet, splits):
    """
    Loads image's names from the specified set
    :param trainingSet: Name of the set (e.g. Training, Validation, Test)
    :param splits: dictionary obtained from the execution of load_params method
    :return: the list of image names (which is the same that the label images)
    """
    if trainingSet not in splits.keys():
        raise Exception("Select one of the following options: " + ", ".join(splits.keys()))
    else:
        list_of_X = []
        list_of_Y = []
        for video in splits[trainingSet]:
            pathToImage = video + "\\Images"
            pathToLabel = video + "\\Labels"
            list_of_X += [pathToImage + "\\" + f for f in os.listdir(pathToImage) if
                          os.path.isfile(pathToImage + "\\" + f)]
            list_of_Y += [pathToLabel + "\\" + f for f in os.listdir(pathToLabel) if
                          os.path.isfile(pathToLabel + "\\" + f)]
        return list_of_X, list_of_Y

IMG_SIZE = (128,128)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, IMG_SIZE)
    x = x / 255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x, _, _ = remap_experiment1(x)
    x = cv2.resize(x, IMG_SIZE)
    x = x / 1.0
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape(IMG_SIZE + (3,))
    y.set_shape(IMG_SIZE)
    return x, y

def load_data(path=r"C:\Users\mauro\OneDrive\Escritorio\CaDISv2"):
    splits = load_params(path)
    train_x, train_y = get_ImagesBySet("Training", splits)
    valid_x, valid_y = get_ImagesBySet("Validation", splits)
    test_x, test_y = get_ImagesBySet("Test", splits)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def tf_dataset(x, y, batch_size = 8, shuffle = True, augment = False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse,num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1000)

    # Batch all datasets
    dataset = dataset.batch(batch_size)

    # Use data augmentation only on the training set
    if augment:
        dataset = dataset.map(random_right_left_flip,num_parallel_calls=AUTOTUNE)

    # Use buffered prefecting on all datasets
    return dataset.prefetch(buffer_size=AUTOTUNE)

@tf.function
def random_right_left_flip(x,y):
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    return tf.cond(choice < 0.5, lambda: (tf.image.flip_left_right(x), tf.image.flip_left_right(y)), lambda: (x,y))