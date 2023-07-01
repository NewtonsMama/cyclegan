import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
#import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_A_DIR = "/kaggle/input/mr-ct-data/mr-ct-data/train/trainA"
TRAIN_B_DIR = "/kaggle/input/mr-ct-data/mr-ct-data/train/trainB"
VAL_A_DIR = "/kaggle/input/mr-ct-data/mr-ct-data/test/testA"
VAL_B_DIR = "//kaggle/input/mr-ct-data/mr-ct-data/test/testB"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 2
IN_CHANNELS = 1
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"
SAVE_IMG_TRAIN_DIR = "/kaggle/working/cyclegan/CycleGAN/saved_images"

transforms = A.Compose(
    [
        #A.Resize(width=64, height=64),
        #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
