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
from torchvision.transforms import functional as TF
from IPython.display import display, Image
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
    temperature=0.5,
    num_epochs=1000,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

VQGAN = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)

e_dim = VQGAN.quantize.e_dim
f = 2**(VQGAN.decoder.num_resolutions - 1)
n_toks = VQGAN.quantize.n_e
toksX, toksY = args.size[0] // f, args.size[1] // f
sideX, sideY = toksX * f, toksY * f
z_min = VQGAN.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
z_max = VQGAN.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

if args.seed is not None:
    torch.manual_seed(args.seed)

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

""" ================== Make z ~ class ================== """
z_all = []
num_classes = 10

for cls_idx in range(num_classes):
    # Generate random tokens for this class
    token_indices = torch.randint(n_toks, [toksY * toksX], device=device)

    # One-hot encode and project to embedding space
    one_hot = F.one_hot(token_indices, n_toks).float()
    z_cls = one_hot @ VQGAN.quantize.embedding.weight

    # Reshape to latent format
    z_cls = z_cls.view(1, toksY, toksX, e_dim).permute(0, 3, 1, 2)  # [1, C, H, W]

    z_all.append(z_cls)

# Stack all class latents into a single tensor
z = torch.cat(z_all, dim=0)  # [num_classes, C, H, W]

z_orig = z.clone()
z.requires_grad_(True)
opt = optim.Adam([z], lr=args.step_size)
""" ================== ========== ================== """

""" ================== Loss & Epoch ================== """
def checkin(i, losses, z, synth, save_path='output'):
    os.makedirs(save_path, exist_ok=True)

    # Handle both single loss or list of losses
    if isinstance(losses, (list, tuple)):
        losses_str = ', '.join(f'{loss.item():.4f}' for loss in losses)
        total_loss = sum(losses).item()
    else:
        losses_str = f'{losses.item():.4f}'
        total_loss = losses.item()

    tqdm.write(f'Epoch/Step: {i}, Total Loss: {total_loss:.4f}, Individual Losses: {losses_str}')

    # Generate and save synthesized image
    for cls in num_classes:
        syn_image = network.decoder(z[cls].unsqueeze(0))  # [1, C, H, W]
        img = TF.to_pil_image(syn_image[0].cpu().clamp(0, 1))  # Clamp to [0, 1] for valid image

        filename = f"image_class_{idx}.png"
        filepath = os.path.join(save_path, filename)

        # Save the image
        img.save(filepath)

    # Display image
    display(Image(save_path))


def proto_loss(z, support_images, support_labels, network, temperature=0.1):
    """
    Compute prototype matching loss using contrastive learning.

    Args:
        z (Tensor): [C, latent_dim] - learnable latent vectors for each class
        support_images (Tensor): [N, C, H, W] - support images
        support_labels (Tensor): [N] - labels for support images
        network.encoder (nn.Module): encoder to extract features
        network.decoder (nn.Module): decoder to reconstruct image from latent
        temperature (float): scaling factor for contrastive loss

    Returns:
        Tensor: scalar loss value
    """
    classes = z.size(0)
    syn_images = []
    z_protos = []

    # Step 1: Decode each latent z to synthetic image and re-encode to get z_proto
    for cls_idx in range(classes):
        syn_image = network.decoder(z[cls_idx].unsqueeze(0))  # [1, C, H, W]
        syn_images.append(syn_image)
        z_proto = network.encoder(syn_image)  # [1, D]
        z_protos.append(z_proto.squeeze(0))  # [D]

    z_protos = torch.stack(z_protos)  # [C, D]

    # Step 2: Encode support images and compute true prototypes
    support_features = network.encoder(support_images)  # [N, D]
    unique_labels = torch.unique(support_labels)
    true_protos = torch.stack([
        support_features[support_labels == lbl].mean(dim=0)
        for lbl in unique_labels
    ])  # [C, D]

    # Flatten spatial dimensions: [10, 256, 2, 2] → [10, 1024]
    z_protos_flat = z_protos.view(z_protos.size(0), -1)
    true_protos_flat = true_protos.view(true_protos.size(0), -1)

    # Normalize for cosine similarity (optional but common in contrastive learning)
    z_protos_flat = F.normalize(z_protos_flat, dim=-1)
    true_protos_flat = F.normalize(true_protos_flat, dim=-1)

    # Matrix multiplication: [10, 1024] @ [1024, 10] → [10, 10]
    logits = torch.matmul(z_protos_flat, true_protos_flat.T) / temperature

    targets = torch.arange(classes).to(z.device)

    loss = F.cross_entropy(logits, targets)

    return loss


def train_epoch(epoch, z, network, opt, args, z_min, z_max, spt_loader):
    for batch_idx, (support_images, support_labels) in enumerate(spt_loader):
        support_images = support_images.to(z.device)
        support_labels = support_labels.to(z.device)

        opt.zero_grad()

        loss = proto_loss(z, support_images, support_labels, network, temperature=args.temperature)

        if batch_idx % args.display_freq == 0:
            checkin(epoch, batch_idx, loss)

        loss.backward()
        opt.step()

        with torch.no_grad():
            z.copy_(torch.clamp(z, min=z_min, max=z_max))


for epoch in range(args.num_epochs):
    train_epoch(epoch, z, VQGAN, opt, args, z_min, z_max, support_loader)
