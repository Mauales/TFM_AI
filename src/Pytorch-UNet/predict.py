import logging
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset, load_data
from utils.cadis_visualization import *


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
            probs = torch.argmax(probs,dim=1)
        else:
            probs = torch.sigmoid(output)

        full_mask = torch.squeeze(probs)

    return full_mask.cpu()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./checkpoints/CP_epoch5.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', default=[])

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(list_files):
    in_files = list_files

    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    path = r"C:\Users\mauro\OneDrive\Escritorio\CaDISv2"

    if len(in_files) == 0:
        _, _, (test_x, test_y) = load_data(path)
        in_files = test_x
        out_files = get_output_filenames(in_files)
    else:
        out_files = get_output_filenames(in_files)
    net = UNet(n_channels=3, n_classes=8)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

        # CHW to HWC
        colormap, classes_exp = remap_experiment1(mask, True)

        # BGR to RGB
        im2 = img.copy()
        im2[:, :, 0] = img[:, :, 2]
        im2[:, :, 2] = img[:, :, 0]

        if not args.no_save:
            out_fn = out_files[i].split("\\")[-1].split(".")[0]
            plt.close('all')
            myplt = plot_images(im2, mask, colormap, classes_exp)
            myplt.savefig("./output/" + out_fn + ".png")
            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
