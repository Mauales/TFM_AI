from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
from .cadis_visualization import *


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        globs = []
        if len(imgs_dir) > 1:
            inter = [glob(img_dir) for img_dir in imgs_dir]
            globs = [item for sublist in inter for item in sublist]
        else:
            globs += self.imgs_dir
        self.ids = [splitext(file)[0].split("Images")[-1][1:] for file in globs if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, cv2_img, scale, isMask = False):
        hwc = cv2_img.shape
        newW, newH = int(scale * hwc[1]), int(scale * hwc[0])
        assert newW > 0 and newH > 0, 'Scale is too small'
        cv2_img = cv2.resize(cv2_img, (newW, newH))

        img_nd = np.array(cv2_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        if not isMask:
            img_trans = img_nd / 255
        else:
            img_trans, _, _ = remap_experiment1(img_nd)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir[0].split("\\")[0] + "/*/Labels/" + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir[0].split("\\")[0] + "/*/Images/" + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_file[0], cv2.IMREAD_COLOR)

        assert img.size/3 == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        mask, _, _ = remap_experiment1(mask)

        #print(np.unique(mask), mask_file)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

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

def load_data(path=r"C:\Users\mauro\OneDrive\Escritorio\CaDISv2"):
    splits = load_params(path)
    train_x, train_y = get_ImagesBySet("Training", splits)
    valid_x, valid_y = get_ImagesBySet("Validation", splits)
    test_x, test_y = get_ImagesBySet("Test", splits)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
