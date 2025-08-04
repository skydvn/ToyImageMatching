import argparse
import io
import math
from pathlib import Path
import sys

sys.path.append('../taming-transformers')
sys.path.append('..')

from IPython import display
from omegaconf import OmegaConf
from PIL import Image
import requests
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from tqdm.auto import tqdm
from utils import *


args = argparse.Namespace(
    size=[32, 32],
    init_image=None,
    init_weight=0.,
    clip_model='ViT-B/32',
    vqgan_config='../vqgan_imagenet_f16_1024.yaml',
    vqgan_checkpoint='../vqgan_imagenet_f16_1024.ckpt',
    step_size=0.05,
    cutn=64,
    cut_pow=1.,
    display_freq=50,
    seed=0,
    batch_size=32,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

VQGAN = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)

cut_size = perceptor.visual.input_resolution
e_dim = model.quantize.e_dim
f = 2**(model.decoder.num_resolutions - 1)
n_toks = model.quantize.n_e
toksX, toksY = args.size[0] // f, args.size[1] // f
sideX, sideY = toksX * f, toksY * f
z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

if args.seed is not None:
    torch.manual_seed(args.seed)

if args.init_image:
    pil_image = Image.open(fetch(args.init_image)).convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
z_orig = z.clone()
z.requires_grad_(True)
opt = optim.Adam([z], lr=args.step_size)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

""" ================== Get Dataset ================== """
# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download CIFAR-10
data_path = './data'
train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

# Split into support and query
support_size = int(0.1 * len(train_dataset))
query_size = len(train_dataset) - support_size
support_set, query_set = random_split(train_dataset, [support_size, query_size])

support_loader = DataLoader(support_set, batch_size=args.batch_size, shuffle=True)

""" ================== Get Dataset ================== """
def train_epoch(epoch, z, encoder, decoder, opt, args, z_min, z_max, support_loader):
    for batch_idx, (support_images, support_labels) in enumerate(support_loader):
        support_images = support_images.to(z.device)
        support_labels = support_labels.to(z.device)

        opt.zero_grad()

        loss = proto_loss(z, support_images, support_labels, encoder, decoder, temperature=args.temperature)

        if batch_idx % args.display_freq == 0:
            checkin(epoch, batch_idx, loss)

        loss.backward()
        opt.step()

        with torch.no_grad():
            z.copy_(torch.clamp(z, min=z_min, max=z_max))


for epoch in range(args.num_epochs):
    train_epoch(epoch, z, encoder, decoder, opt, args, z_min, z_max, support_loader)
