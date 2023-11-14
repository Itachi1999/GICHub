import config
from torchvision.transforms import transforms
from generator_model import Generator
from utils import load_checkpoint
import torch.optim as optim
from PIL import Image
from torchvision.utils import save_image
# gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
# gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

# opt_gen = optim.Adam(
#     list(gen_Z.parameters()) + list(gen_H.parameters()),
#     lr=config.LEARNING_RATE,
#     betas=(0.5, 0.999),
# )

# load_checkpoint(
#     config.CHECKPOINT_GEN_H,
#     gen_H,
#     opt_gen,
#     config.LEARNING_RATE,
# )
# load_checkpoint(
#     config.CHECKPOINT_GEN_Z,
#     gen_Z,
#     opt_gen,
#     config.LEARNING_RATE,
# )

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def opAPI(animal, img_path):
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    img = Image.open(img_path)
    transformed_img = transform(img).to(config.DEVICE)
    load_checkpoint(
        config.CHECKPOINT_GEN_Z,
        gen_Z,
        opt_gen,
        config.LEARNING_RATE,
    )

    load_checkpoint(
        config.CHECKPOINT_GEN_Z,
        gen_H,
        opt_gen,
        config.LEARNING_RATE,
    )

    if animal == 'Z' or 'z':
        op_img = gen_Z.forward(transformed_img)
    else:
        op_img = gen_H.forward(transformed_img)

    save_image(op_img, 'GAN_model/cycleGAN_standard/op.jpg')


if __name__ == '__main__':
    opAPI('Z', 'GAN_model/cycleGAN_standard/n02381460_120.jpg')
