import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def eval_fn(gen_Z, gen_H, loader):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        torch.cuda.empty_cache()
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # fake_horse = gen_H(zebra)
        # fake_zebra = gen_Z(horse)
        fake_zebra = gen_Z(gen_H(zebra))
        fake_horse = gen_H(gen_Z(horse))
        fake_horse = fake_horse*0.5+0.5
        zebra = zebra*0.5 +0.5
        horse = horse*0.5 + 0.5
        fake_zebra = fake_zebra*0.5+0.5

        save_image(fake_horse, f"{config.SAVE_IMG_TEST_DIR}/test/gen_horse/{idx}.png")
        save_image(fake_zebra, f"{config.SAVE_IMG_TEST_DIR}/test/gen_zebra/{idx}.png")
        save_image(horse, f"{config.SAVE_IMG_TEST_DIR}/test/ori_horse/{idx}.png")
        save_image(zebra, f"{config.SAVE_IMG_TEST_DIR}/test/ori_zebra/{idx}.png")
        


        # save_image(torch.cat((zebra,fake_horse),0), f"{config.SAVE_IMG_TEST_DIR}/compare_test/horse_{idx}.png")
        # save_image(torch.cat((horse, fake_zebra),0), f"{config.SAVE_IMG_TEST_DIR}/compare_test/zebra_{idx}.png")
        save_image(torch.cat((zebra,fake_zebra),0), f"{config.SAVE_IMG_TEST_DIR}/compare_test/zebra_{idx}.png")
        save_image(torch.cat((horse, fake_horse),0), f"{config.SAVE_IMG_TEST_DIR}/compare_test/horse_{idx}.png")

        loop.set_postfix()


def main():
    gen_Z = Generator(img_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=config.IN_CHANNELS, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    load_checkpoint(
        config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
    )

    


    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_A_DIR ,
        root_zebra=config.VAL_B_DIR,
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    gen_Z.eval()
    gen_H.eval()
    eval_fn(gen_Z, gen_H, val_loader)

if __name__ == "__main__":
    main()