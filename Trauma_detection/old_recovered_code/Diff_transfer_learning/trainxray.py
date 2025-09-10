import os
import zipfile
import tqdm
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,Subset
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.nn.utils import spectral_norm
import numpy as np
from PIL import Image
from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import accelerate
import glob
from torch.utils.data import DataLoader, Dataset
import time
import gc
import math

accelerator = Accelerator()
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

################################################################################
# 3. Main: Download, Locate, DataLoader
################################################################################
'''dataset_path = os.path.join('/media/scratch/Trauma_Detection/code/chest_xray_kaggle/chest_xray/train')
transform_1= transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class xrays(Dataset):
    def __init__(self, image_folder, transform=None, max_images=None):
        self.image_paths = glob.glob(os.path.join(image_folder, '**', '*.png'), recursive=True) + \
                           glob.glob(os.path.join(image_folder, '**', '*.jpg'), recursive=True) + \
                           glob.glob(os.path.join(image_folder, '**', '*.jpeg'), recursive=True)
                           
        #self.image_paths = self.image_paths[:10000]

        self.transform = transform

        print(f"Using {len(self.image_paths)} images from {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

dataset = xrays(dataset_path, transform=transform, max_images=None)
loader = DataLoader(dataset, batch_size=16, shuffle=True)'''




#using  pokemon dataset
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



###############################################################################
# 3. Load & Freeze Stable Diffusion + Hook for ADN-CDCR
###############################################################################
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base",subfolder="unet").to(device)
vae  = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base",subfolder="vae").to(device)
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base",subfolder="text_encoder").to(device)
tokenizer    = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base",subfolder="tokenizer")
scheduler    = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base",subfolder="scheduler")

unet.eval(); vae.eval(); text_encoder.eval()
for p in unet.parameters():
    p.requires_grad=False
for p in vae.parameters():
    p.requires_grad=False
for p in text_encoder.parameters():
    p.requires_grad=False

class ShapeAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ShapeAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(True),  # 128x128
            nn.Conv2d(32,64,4,2,1), nn.ReLU(True), # 64x64
            nn.Conv2d(64,128,4,2,1), nn.ReLU(True),# 32x32
            nn.Conv2d(128,256,4,2,1), nn.ReLU(True),#16x16
            nn.Conv2d(256,512,4,2,1), nn.ReLU(True),#8x8
        )
        self.fc_mu = nn.Linear(512*8*8,latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim,512*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1),   nn.ReLU(True),
            nn.ConvTranspose2d(32,1,4,2,1),    nn.Tanh()
        )

    def encode(self, x):
        # x: (B,1,256,256)
        h = self.enc(x)              # (B,512,8,8)
        h = h.view(h.size(0),-1)     # (B,512*8*8)
        mu= self.fc_mu(h)            # (B, latent_dim)
        return mu

    def decode(self, z):
        # z: (B, latent_dim)
        h = self.dec_fc(z)           # (B,512*8*8)
        h = h.view(h.size(0),512,8,8)
        return self.dec(h)           # (B,1,256,256)

    def forward(self,x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

shape_ae2 = ShapeAE(latent_dim=128).to(device) 
shape_ae2.load_state_dict(torch.load("/media/scratch/Trauma_Detection/code/AE_model/shape_ae_pokeman.pth", map_location=device))
shape_ae2.eval()
for p in shape_ae2.parameters():
    p.requires_grad=False

###############################################################################
# 4. Define Adaptation Layers for ALL UNet blocks
###############################################################################
class AdaptationLayer(nn.Module):
    def __init__(self, channels):
        super(AdaptationLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.norm= nn.GroupNorm(32, channels)
        self.relu= nn.ReLU(inplace=False)
    def forward(self,x):
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out + residual

def get_layer_channels(unet_model):
    dummy_in = torch.randn(1, unet_model.config.in_channels, 64, 64, device=device)
    dummy_ts = torch.tensor([0], dtype=torch.long,device=device)
    dummy_txt= torch.randn(1,77,unet_model.config.cross_attention_dim,device=device)

    out_ch = {}
    def hook_fn(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            out_ch[name] = out.shape[1]
        return hook

    hks=[]
    # Down blocks
    for i, block in enumerate(unet_model.down_blocks):
        h = block.register_forward_hook(hook_fn(f"down_block_{i}"))
        hks.append(h)
    # Mid block
    h_mid = unet_model.mid_block.register_forward_hook(hook_fn("mid_block"))
    hks.append(h_mid)
    # Up blocks
    for i, block in enumerate(unet_model.up_blocks):
        h = block.register_forward_hook(hook_fn(f"up_block_{i}"))
        hks.append(h)

    with torch.no_grad():
        _= unet_model(dummy_in, dummy_ts, dummy_txt)

    for h in hks:
        h.remove()
    return out_ch

layer_info = get_layer_channels(unet)
adaptation_layers = nn.ModuleDict()
for name,chs in layer_info.items():
    adaptation_layers[name] = AdaptationLayer(chs).to(device)

###############################################################################
# 5. Pixel-level Discriminator for domain alignment
###############################################################################
class PixelDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super(PixelDiscriminator, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch,64,4,2,1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64,128,4,2,1)),   nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128,256,4,2,1)),  nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256,1,1)
        )
    def forward(self,x):
        out = self.main(x)
        return out.view(x.size(0),-1)

pixel_disc = PixelDiscriminator(in_ch=3).to(device)

###############################################################################
# 6. Full Hooking Mechanism
###############################################################################
def create_hook(block_name):
    def hook_fn(module, inp, out):
        adapter_module = adaptation_layers.module if hasattr(adaptation_layers, 'module') else adaptation_layers
        
        if isinstance(out, tuple):
            out_0 = out[0]
            out_0 = adapter_module[block_name](out_0)
            return (out_0,) + out[1:]
        else:
            return adapter_module[block_name](out)
    return hook_fn

def register_all_hooks(unet_model, adaptation_layers, hooks_container):
    # Down blocks
    for i, block in enumerate(unet_model.down_blocks):
        h = block.register_forward_hook(create_hook(f"down_block_{i}"))
        hooks_container.append(h)
    
    # Mid block
    mid_h = unet_model.mid_block.register_forward_hook(create_hook("mid_block"))
    hooks_container.append(mid_h)
    
    # Up blocks
    for i, block in enumerate(unet_model.up_blocks):
        h = block.register_forward_hook(create_hook(f"up_block_{i}"))
        hooks_container.append(h)

def remove_all_hooks(hooks_container):
    for h in hooks_container:
        h.remove()
    hooks_container.clear()

###############################################################################
# 7. Training Setup for Phase B with Accelerate
###############################################################################

from accelerate import Accelerator
from accelerate.utils import set_seed

accelerator = Accelerator(mixed_precision='fp16') 
device = accelerator.device
set_seed(42) 

optimizer_adapters = optim.Adam(adaptation_layers.parameters(), lr=1e-4)
optimizer_disc = optim.Adam(pixel_disc.parameters(), lr=1e-5)
bce_loss = nn.BCEWithLogitsLoss()
EPOCHS_ADAPT = 50

unet, vae, text_encoder, shape_ae2, pixel_disc, adaptation_layers = accelerator.prepare(
    unet, vae, text_encoder, shape_ae2, pixel_disc, adaptation_layers
)
optimizer_adapters, optimizer_disc = accelerator.prepare(optimizer_adapters, optimizer_disc)
loader = accelerator.prepare(loader)

def encode_prompts(prompts):
    toks = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = text_encoder(toks.input_ids)[0]
    return out

# Load and prepare ShapeAE
shape_ae2 = ShapeAE(latent_dim=128)
shape_ae2.load_state_dict(torch.load("/media/scratch/Trauma_Detection/code/AE_model/shape_ae_pokeman.pth", map_location=device))
shape_ae2.eval()
for p in shape_ae2.parameters():
    p.requires_grad = False
shape_ae2 = accelerator.prepare(shape_ae2)

print("\n=== PHASE B: Training ADN-CDCR with shape-latent penalty & pixel disc ===")
unet.eval()
vae.eval()
text_encoder.eval()
shape_ae2.eval()

hooks_container = []

'''for epoch in range(EPOCHS_ADAPT):
    total_loss_gen = 0.0
    total_loss_disc = 0.0

    #for batch_idx, (imgs) in enumerate(loader):
        #imgs = imgs.to(device)
        #bsz = imgs.size(0)
    


    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, tuple) or isinstance(batch, list):
            imgs = batch[0]  
        else:
            imgs = batch 
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.stack(imgs) if isinstance(imgs, list) else torch.tensor(imgs)
        
        imgs = imgs.to(device)
        bsz = imgs.size(0)

        # (A) Encode -> latents
        with torch.no_grad():
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

        # (B) Sample noise + timesteps
        noise = torch.randn_like(latents)
        t_samples = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, t_samples)

    

                # (C) Build text prompts
        # Instead of referencing dataset.classes:bsz = imgs.size(0)
        prompts = ["a Normal chest x-ray"] * bsz
        txt_embeds = encode_prompts(prompts)


        # (D) Register hooking for ALL blocks
        register_all_hooks(unet, adaptation_layers, hooks_container)

        # (E) Forward pass => predicted noise
        with torch.no_grad():
            unet.eval()
        model_pred = unet(noisy_latents, t_samples, txt_embeds).sample

        # (F) Remove hooks
        remove_all_hooks(hooks_container)

        # (G) Diffusion loss
        diff_loss = F.mse_loss(model_pred, noise)

        # (H) Decode to get fake images
        with torch.no_grad():
            denoised_lat = noisy_latents - model_pred
            fake_imgs = vae.decode(denoised_lat/0.18215).sample

        # (I) Pixel Discriminator Step
        with accelerator.accumulate(pixel_disc):
            pixel_disc.train()
            optimizer_disc.zero_grad()

            real_logits = pixel_disc(imgs)
            real_labels = torch.ones_like(real_logits)
            loss_disc_real = bce_loss(real_logits, real_labels)

            fake_logits = pixel_disc(fake_imgs.detach())
            fake_labels = torch.zeros_like(fake_logits)
            loss_disc_fake = bce_loss(fake_logits, fake_labels)

            disc_loss = 0.5 * (loss_disc_real + loss_disc_fake)
            accelerator.backward(disc_loss)
            optimizer_disc.step()
            total_loss_disc += disc_loss.item()

        # (J) Generator Adversarial
        gen_logits = pixel_disc(fake_imgs)
        loss_adv = bce_loss(gen_logits, torch.ones_like(gen_logits))

        # (K) Shape-latent penalty
        def to_gray(x):
            return x.mean(dim=1, keepdim=True)

        with torch.no_grad():
            real_gray = to_gray(imgs)
        z_real = shape_ae2.encode(real_gray)

        register_all_hooks(unet, adaptation_layers, hooks_container)

        fake_gray = to_gray(fake_imgs)
        z_fake = shape_ae2.encode(fake_gray)

        shape_loss = F.mse_loss(z_fake, z_real)

        # (L) Combine generator(adapter) loss
        #total_gen_loss = diff_loss + 0.1*loss_adv + 0.01*shape_loss -initial
        
        #adv loss as o.oo1
        #total_gen_loss = diff_loss + 0.1*0.0001 + 0.01*shape_loss

        #total_gen_loss = diff_loss + 1*loss_adv #ablation (no shape loss)


        total_gen_loss = diff_loss + 0.01 * shape_loss #no avd loss


        # (M) Optimize Adapters
        with accelerator.accumulate(adaptation_layers):
            optimizer_adapters.zero_grad()
            accelerator.backward(total_gen_loss)
            accelerator.clip_grad_norm_(adaptation_layers.parameters(), 1.0) 
            optimizer_adapters.step()

        remove_all_hooks(hooks_container)

        total_loss_gen += total_gen_loss.item()

        if accelerator.is_main_process and batch_idx % 50 == 0:
            print(
                f"[Ep {epoch+1}/{EPOCHS_ADAPT}, Batch {batch_idx}] "
                f"Diff:{diff_loss.item():.4f} | Disc:{disc_loss.item():.4f} | "
                f"Adv:{loss_adv.item():.4f} | Shape:{shape_loss.item():.4f} | "
                f"G_Tot:{total_gen_loss.item():.4f}"
            )

    avg_gen = total_loss_gen/len(loader)
    avg_disc = total_loss_disc/len(loader)
    
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} -> GenLoss: {avg_gen:.4f}, DiscLoss: {avg_disc:.4f}")
        #checkpoint_dir = f"checkpoint_25epoch_{epoch+1}"
        #accelerator.save_state(checkpoint_dir)

if accelerator.is_main_process:
    print("Finished training adaptation layers with shape-latent + pixel disc.")
    adapt_layers = accelerator.unwrap_model(adaptation_layers)
    torch.save(adapt_layers.state_dict(), 
               "/media/scratch/Trauma_Detection/code/AE_model/adapters_shapeae_pokeman.pth")'''

###############################################################################
#8. Generating and saving images
###############################################################################
#ADAPTER_PATH = "/media/scratch/Trauma_Detection/code/AE_model/adapters_shapeae_nihcc_no_advloss.pth"
ADAPTER_PATH ='/media/scratch/Trauma_Detection/code/AE_model/adapters_shapeae_pokeman.pth'


accelerator = Accelerator()
device = accelerator.device

print("Building adaptation layers...")
layer_channels = get_layer_channels(unet)
adaptation_layers = nn.ModuleDict()
for name, ch in layer_channels.items():
    adaptation_layers[name] = AdaptationLayer(ch)  # Remove .to(device) as accelerator will handle this

# Load the trained adapter weights
adaptation_layers.load_state_dict(torch.load(ADAPTER_PATH, map_location=device))

# Prepare all models with accelerator
unet, vae, text_encoder, adaptation_layers = accelerator.prepare(unet, vae, text_encoder, adaptation_layers)

def generate_images(prompts, num_inference_steps=50, guidance_scale=7.5, height=256, width=256):
    if isinstance(prompts, str):
        prompts=[prompts]
    bsz= len(prompts)

    # Encode text
    tokens= tokenizer(prompts, padding="max_length",
                      max_length=tokenizer.model_max_length,
                      truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds= text_encoder(tokens.input_ids)[0]

    uncond_tokens= tokenizer([""]*bsz, padding="max_length",
                             max_length=tokenizer.model_max_length,
                             return_tensors="pt").to(device)
    with torch.no_grad():
        uncond_embeds= text_encoder(uncond_tokens.input_ids)[0]

    # Merge for CFG
    enc_hid= torch.cat([uncond_embeds, text_embeds], dim=0)  # (2B, seq_len, hidden_dim)

    # Latents
    lat_shape= (bsz, unet.config.in_channels, height//8, width//8)
    latents= torch.randn(lat_shape, device=device)

    # Sampler steps
    scheduler.set_timesteps(num_inference_steps)
    t_steps= scheduler.timesteps

    # Register hooking for all blocks
    hooks_container_infer=[]
    def create_hook_fn(block_name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                o0 = out[0]
                # Access the underlying model using .module
                o0 = adaptation_layers.module[block_name](o0)
                return (o0,) + out[1:]
            else:
                # Access the underlying model using .module
                return adaptation_layers.module[block_name](out)
        return hook

    for i, block in enumerate(unet.down_blocks):
        h= block.register_forward_hook(create_hook_fn(f"down_block_{i}"))
        hooks_container_infer.append(h)
    mid_h= unet.mid_block.register_forward_hook(create_hook_fn("mid_block"))
    hooks_container_infer.append(mid_h)
    for i, block in enumerate(unet.up_blocks):
        h= block.register_forward_hook(create_hook_fn(f"up_block_{i}"))
        hooks_container_infer.append(h)

    # Denoising loop
    for t in t_steps:
        latent_in= torch.cat([latents, latents], dim=0)
        with torch.no_grad():
            noise_pred= unet(latent_in, t, enc_hid).sample

        noise_pred_uncond, noise_pred_text= noise_pred.chunk(2)
        noise_pred= noise_pred_uncond + guidance_scale*(noise_pred_text - noise_pred_uncond)

        latents= scheduler.step(noise_pred, t, latents).prev_sample

    # remove inference hooks
    for hh in hooks_container_infer:
        hh.remove()

    # Decode
    with torch.no_grad():
        scaled_lat= latents/0.18215
        images   = vae.decode(scaled_lat).sample

    # Convert to PIL
    images= (images/2+0.5).clamp(0,1).cpu().permute(0,2,3,1).numpy()
    pil_imgs= [Image.fromarray((img*255).astype(np.uint8)) for img in images]
    return pil_imgs


example_prompts = [
    "a normal pokemon image"]


TOTAL_IMAGES = 500
BATCH_SIZE = 70
DELAY_SECONDS = 60 

parent_dir = "/media/scratch/Trauma_Detection/code/Pokemon_500recent"
os.makedirs(parent_dir, exist_ok=True)

for idx, prompt in enumerate(example_prompts):
    print(f"Generating for prompt: '{prompt}'")
    folder_name = prompt.replace(" ", "_").replace("-", "_").replace(",", "")
    save_dir = os.path.join(parent_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    num_batches = math.ceil(TOTAL_IMAGES / BATCH_SIZE)
    
    for batch_idx in range(num_batches):
        current_batch_size = min(BATCH_SIZE, TOTAL_IMAGES - batch_idx * BATCH_SIZE)
        print(f"\nProcessing batch {batch_idx+1}/{num_batches} ({current_batch_size} images)")
        out_imgs = generate_images(
            [prompt] * current_batch_size,
            num_inference_steps=100,
            guidance_scale=5)
        
        for img_idx, img in enumerate(out_imgs):
            fname = os.path.join(save_dir,f"final_{idx}_b{batch_idx:02d}_i{img_idx:03d}.png")
            img.save(fname)
        
        del out_imgs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if batch_idx < num_batches - 1:
            print(f"Waiting {DELAY_SECONDS} seconds for GPU cleanup...")
            time.sleep(DELAY_SECONDS)

print("\nAll done! Check output directory for images.")