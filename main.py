import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from taming.models.vqgan import VQModel  # pip install taming-transformers
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load CLIP ----
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load VQGAN ----
vqgan = VQModel.from_pretrained("CompVis/vqgan-f16-16384").to(device)

# ---- CIFAR-10 Class Texts (User1) ----
cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# ---- Dataset (User2) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# ---- Encode Class Texts ----
text_inputs = clip_processor(text=cifar_classes, return_tensors="pt", padding=True).to(device)
text_embeds = clip_model.get_text_features(**text_inputs)
text_embeds = F.normalize(text_embeds, dim=-1)



# ===== Example Usage =====
# Generate synthetic image (like middle part of figure)
x_hat_tensor, x_hat_img = generate_image_from_z()

pred_text, pred_proto, sim_text, sim_proto = classify_image(x_hat_img)

print("Predicted class (Text Embedding):", cifar_classes[pred_text])
print("Predicted class (Prototype):", cifar_classes[pred_proto])
