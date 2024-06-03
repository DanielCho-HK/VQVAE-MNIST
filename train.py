import torch
from torch.optim import Adam
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from model import VQVAE

log_dir = './logs/'
checkpoint_dir = './train/checkpoint/'
sample_dir = './train/sample/'

writer = SummaryWriter(log_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def train_vqvae():

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=4)

    model = VQVAE().to(device)
    num_epoches = 20
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    iteration = 0
    for epoch_idx in range(num_epoches):
        for im, _ in tqdm(mnist_loader):
            im = im.to(device)
            out, quantize_loss = model(im)

            recon_loss = criterion(out, im)
            loss = recon_loss + quantize_loss
            writer.add_scalar('train_loss', loss, global_step=iteration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

        print('Finished epoch {}'.format(epoch_idx+1))
        torch.save({
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    }, checkpoint_dir + f"ckpt_{str(epoch_idx+1)}.pth")

train_vqvae()