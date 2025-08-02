import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from taming.models.vqgan import VQModel  # pip install taming-transformers
from PIL import Image


# ---- Compute Class Prototypes from K-shot CIFAR Images ----
def get_class_prototypes(dataset, k=5):
    prototypes = []
    for class_idx in range(len(cifar_classes)):
        imgs = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == class_idx][:k]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        img_embeds = clip_model.get_image_features(**inputs)
        prototypes.append(F.normalize(img_embeds.mean(dim=0), dim=-1))
    return torch.stack(prototypes)

prototypes = get_class_prototypes(train_dataset, k=5)

# ---- Generate Synthetic Image from VQGAN ----
def generate_image_from_z():
    z = torch.randn(1, 256, 16, 16).to(device)  # latent space
    with torch.no_grad():
        x_hat = vqgan.decode(z)
    x_hat_img = transforms.ToPILImage()(x_hat.squeeze().cpu().clamp(0,1))
    return x_hat, x_hat_img

# ---- Match Image to Classes ----
def classify_image(image_tensor):
    inputs = clip_processor(images=image_tensor, return_tensors="pt").to(device)
    img_embed = clip_model.get_image_features(**inputs)
    img_embed = F.normalize(img_embed, dim=-1)

    sim_text = torch.matmul(img_embed, text_embeds.T)
    sim_proto = torch.matmul(img_embed, prototypes.T)

    pred_text = sim_text.argmax(dim=-1).item()
    pred_proto = sim_proto.argmax(dim=-1).item()

    return pred_text, pred_proto, sim_text, sim_proto