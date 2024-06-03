import torch
from torch.optim import Adam
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from model import VQVAE



sample_dir = './train/sample/'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

idxs = torch.randint(0, len(mnist_test), (100, ))
ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()
ims = ims

model = VQVAE()
ckpt =torch.load('./train/checkpoint/ckpt_20.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()
generated_im, _ = model(ims)


utils.save_image(
    ims.cpu().data,
    sample_dir + f'original.png',
    normalize=True,
    nrow=10,
    range=(-1, 1)
)

utils.save_image(
    generated_im.cpu().data,
    sample_dir + f'gen.png',
    normalize=True,
    nrow=10,
    range=(-1, 1)
)