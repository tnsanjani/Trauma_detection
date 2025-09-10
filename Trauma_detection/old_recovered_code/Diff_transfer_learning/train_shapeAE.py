###############################################################################
# 2. PHASE A: Train the Shape Autoencoder (Shape-AE)
###############################################################################
import os
import zipfile
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.nn.utils import spectral_norm
import numpy as np
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''dataset_path = os.path.join('/media/scratch/datasets/nihcc_xray')

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),  
    transforms.Normalize([0.5], [0.5]),  
])

class nihcc_xray(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = glob.glob(os.path.join(image_folder, '**', '*.png'), recursive=True) + \
                           glob.glob(os.path.join(image_folder, '**', '*.jpg'), recursive=True) + \
                           glob.glob(os.path.join(image_folder, '**', '*.jpeg'), recursive=True)

        self.transform = transform

        print(f"Using {len(self.image_paths)} images from {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

dataset = nihcc_xray(dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)'''

#new dataset
################################################################################
# 2. Find a Folder with Chest X-Ray Images
################################################################################
'''def find_vinbigdata_folder():
    search_root = "/media/scratch/datasets/vinbigdata_extracted"
    for root, dirs, files in os.walk(search_root):
        # If 'train' or 'images' is a subfolder containing PNG/JPG images
        for d in dirs:
            d_path = os.path.join(root, d)
            d_lower = d.lower()
            if d_lower in ["train","images"]:
                # Check if it has some .png/.jpg
                exts = (".png",".jpg",".jpeg")
                sample_imgs = [f for f in os.listdir(d_path) if f.lower().endswith(exts)]
                if len(sample_imgs)>0:
                    return d_path
    raise FileNotFoundError(
        "Could not find subfolder named 'train' or 'images' with .png/.jpg. "
    )

################################################################################
# 3. Main: Download, Locate, DataLoader
################################################################################
vinbig_root = find_vinbigdata_folder()
print(f"Detected VinBigData images folder: {vinbig_root}")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

class SingleFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Gather .png/.jpg
        exts = (".png",".jpg",".jpeg")
        self.img_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(exts)
        ]
        self.img_paths.sort()
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0 

dataset = SingleFolderDataset(vinbig_root, transform=transform)


indices = list(range(len(dataset)))
random.shuffle(indices)
keep_count = int(1.00* len(dataset))
subset_idxs = indices[:keep_count]
subset_ds   = Subset(dataset, subset_idxs)'''




#using pokemon dataset

import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset_path = os.path.join('/media/scratch/datasets/PokemonData')
base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

augment_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

class AugmentedPokemonDataset(Dataset):
    def __init__(self, image_folder, base_transform=None, augment_transform=None, target_size=10000):
        self.base_transform = base_transform
        self.augment_transform = augment_transform
        self.image_folder = image_folder
        self.target_size = target_size
        self.subfolders = [f.path for f in os.scandir(image_folder) if f.is_dir()]

        self.subfolder_images = {}
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        original_image_paths = []
        for idx, subfolder in enumerate(self.subfolders):
            folder_name = os.path.basename(subfolder)
            self.class_to_idx[folder_name] = idx
            self.idx_to_class[idx] = folder_name
            
            image_paths = glob.glob(os.path.join(subfolder, '*.png')) + \
                         glob.glob(os.path.join(subfolder, '*.jpg')) + \
                         glob.glob(os.path.join(subfolder, '*.jpeg'))
            
            self.subfolder_images[folder_name] = image_paths
            original_image_paths.extend([(path, idx) for path in image_paths])
        
        self.original_count = len(original_image_paths)
        self.original_image_paths = original_image_paths

        if self.original_count >= target_size:
            self.all_image_paths = original_image_paths[:target_size]
            self.augmentation_factor = 1
        else:
            self.augmentation_factor = int(np.ceil(target_size / self.original_count))
            self.all_image_paths = []
            for i in range(self.augmentation_factor):
                if i == 0:
                    self.all_image_paths.extend([(path, idx, 0) for path, idx in original_image_paths])
                else:
                    self.all_image_paths.extend([(path, idx, i) for path, idx in original_image_paths])
            self.all_image_paths = self.all_image_paths[:target_size]
        
        class_counts = {}
        for item in self.all_image_paths:
            if isinstance(item, tuple) and len(item) >= 2:
                class_idx = item[1]
                class_name = self.idx_to_class[class_idx]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        item = self.all_image_paths[idx]
        image_path, class_idx, aug_idx = item
        original_image = Image.open(image_path).convert('RGB')
    
        image = original_image.copy()
        if aug_idx > 0:
            seed = int(hash(f"{image_path}_{aug_idx}") % 10000)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if self.augment_transform:
                transformed_image = self.augment_transform(image)
        else:
            if self.base_transform:
                transformed_image = self.base_transform(image)
        original_transformed = self.base_transform(original_image) if self.base_transform else original_image
        
        return transformed_image, original_transformed, class_idx, aug_idx > 0

dataset = AugmentedPokemonDataset(
    dataset_path, 
    base_transform=base_transform, 
    augment_transform=augment_transform, 
    target_size=10000
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)
print(f"Total images in dataset: {len(dataset)}")

class TrainingPokemonDataset(Dataset):
    def __init__(self, augmented_dataset):
        self.augmented_dataset = augmented_dataset
    
    def __len__(self):
        return len(self.augmented_dataset)
    
    def __getitem__(self, idx):
        transformed_image, _, class_idx, _ = self.augmented_dataset[idx]
        return transformed_image, class_idx

training_dataset = TrainingPokemonDataset(dataset)
loader = DataLoader(training_dataset, batch_size=16, shuffle=True)
accelerator = Accelerator(mixed_precision='fp16')
set_seed(42)
device = accelerator.device 

class ShapeAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ShapeAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),  # 128x128
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),  # 64x64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),  # 32x32
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),  # 16x16
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(True),  # 8x8
        )
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        return mu

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 512, 8, 8)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

# Create and prepare components
shape_ae = ShapeAE(latent_dim=128).to(device)
opt_ae = optim.Adam(shape_ae.parameters(), lr=1e-4)

# Convert to grayscale
def to_grayscale(imgs):
    return imgs.mean(dim=1, keepdim=True)


shape_ae, opt_ae, loader = accelerator.prepare(shape_ae, opt_ae, loader)
EPOCHS_SHAPEAE = 50
print("\n=== PHASE A: Training Shape-AE ===")

for epoch in range(EPOCHS_SHAPEAE):
    shape_ae.train()
    total_loss = 0.0
    
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        gray = to_grayscale(imgs)
        recons = shape_ae(gray)
        recons = torch.clamp(recons, -1.0, 1.0)
        
        loss_ae = F.l1_loss(recons, gray)
        
        accelerator.backward(loss_ae)
        opt_ae.step()
        opt_ae.zero_grad() 
        
        total_loss += loss_ae.item()
    
    avg_loss = total_loss / len(loader)

    if accelerator.is_local_main_process:
        print(f"[ShapeAE Epoch {epoch+1}/{EPOCHS_SHAPEAE}] Recon Loss: {avg_loss:.4f}")


accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(shape_ae)
    torch.save(unwrapped_model.state_dict(), "/media/scratch/Trauma_Detection/code/AE_model/shape_ae_pokeman.pth")
    print("Shape-AE training complete & saved.\n")

