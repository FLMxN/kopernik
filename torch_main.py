import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from predictor import predict_country, predict_region
from datasets import load_dataset
import sys
import dotenv
import os
import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# full_dataset = load_dataset("stochastic/random_streetview_images_pano_v0.0.2", 
#                           split="train", 
#                           streaming=True)
# imgs = []
# for i, example in enumerate(full_dataset):
#     if i < 1234:
#         print(i)
#         imgs.append(example["image"])


dotenv_file = Path(__file__).parent / '.env'
dotenv.load_dotenv(dotenv_file, override=True)
env = os.environ.copy()

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ------------------------------------------------------------- CONFIG -------------------------------------------------------
IMGS = [env['INPUT_IMG']]
# IMGS = ["pics/black-screen-4288342275.png"]
# IMGS = ["pics/image.png", "pics/zahodryazan.jpg", "pics/ryazan-russia-city-view-3628679470.jpg", "pics/t1.png", "pics/t2.png", "pics/t3.png", "pics/t4.png", "pics/Ryazan-03.jpg", "pics/5df12e8f9e3d0-5140-sobornaja-ploschad.jpeg"]
HEIGHT = 561
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_PRETTY = True if os.environ['PRETTY']=="1" else False
# IS_PRETTY = True
# ----------------------------------------------------------------------------------------------------------------------------

try:
    if 'verbose' in sys.argv:
        IS_PRETTY = False
except:
    pass
    
id2label_map = {
    7: "AD", 16: "AE", 15: "AR", 43: "AU", 30: "BD",
    26: "BE", 4: "BG", 46: "BR", 48: "BT", 31: "BW",
    3: "CA", 49: "CH", 34: "CL", 17: "CO", 45: "CZ",
    25: "DE", 36: "DK", 12: "EE", 41: "ES", 23: "FI",
    28: "FR", 0: "GB", 53: "GR", 22: "HK", 24: "HR",
    14: "HU", 42: "ID", 13: "IE", 51: "IL", 6: "IS",
    27: "IT", 35: "JP", 10: "KH", 1: "KR", 32: "LT",
    50: "LV", 29: "MX", 9: "MY", 2: "NL", 5: "NO",
    37: "NZ", 44: "PE", 47: "PL", 21: "PT", 38: "RO",
    52: "RU", 40: "SE", 19: "SG", 55: "SI", 8: "SK",
    11: "SZ", 18: "TH", 33: "TW", 39: "UA", 54: "US",
    20: "ZA"
}

id2label_map_reg = {
    0: "anglosphere", 1: "tropical", 2: "arid_african", 3: "mediterranean", 4: "post_socialist", 5: "nordic", 6: "east_asia"
}

def crop_resize(image: Image.Image, size = (HEIGHT, HEIGHT)) -> Image.Image:
    w, h = image.size
    new_h = HEIGHT
    new_w = int(w * (new_h / h))
    out = image.resize((new_w, new_h), Image.BICUBIC)
    return out

def stretch_resize(image: Image.Image, size = (round(HEIGHT*(16/9)), HEIGHT)) -> Image.Image:
    res = image.resize(size)
    return res
 
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def clean_state_dict_keys(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_sd[new_key] = v
    return new_sd

class ResNet50Country(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.id2label = id2label_map
        self.country_head = nn.Linear(in_features, num_classes)
        self.coordinate_head = nn.Linear(in_features, 2)  # (longitude, latitude)
    
    def forward(self, x):
        features = self.resnet(x)
        country_logits = self.country_head(features)
        coordinates = self.coordinate_head(features)
        return country_logits, coordinates
    
class ResNet50Region(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.id2label = id2label_map_reg
        self.region_head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        features = self.resnet(x)
        region_logits = self.region_head(features)
        return region_logits

def load_model_checkpoint(is_reg, path: str, device: torch.device, num_classes=56):
    if is_reg:
        model = ResNet50Region(num_classes=num_classes)
    else:
        model = ResNet50Country(num_classes=num_classes)
    
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    
    if not IS_PRETTY:
        print("Checkpoint structure:")
        if isinstance(checkpoint, dict):
            for k, v in checkpoint.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape {v.shape}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict with {len(v)} keys")
                else:
                    print(f"  {k}: {type(v).__name__} = {v}")
    
    # Extract the actual model weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\n✅ Extracted model_state_dict from checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\n✅ Extracted state_dict from checkpoint")
        else:
            # If the checkpoint itself is the state_dict
            state_dict = checkpoint
            print("\n✅ Checkpoint is already the state_dict")
    else:
        state_dict = checkpoint
    
    # Clean keys (remove 'module.' prefix if present)
    state_dict = clean_state_dict_keys(state_dict)
    
    # Debug: Show first few keys
    if not IS_PRETTY:
        print("\n📋 State dict keys (first 10):")
        for i, (k, v) in enumerate(list(state_dict.items())[:10]):
            if hasattr(v, 'shape'):
                print(f"  {k}: shape {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
    
    # Check if we have the new multi-task structure
    has_country_head = any('country_head' in k for k in state_dict.keys())
    has_coordinate_head = any('coordinate_head' in k for k in state_dict.keys())
    
    if not IS_PRETTY:
        if has_country_head and has_coordinate_head:
            print("\n✅ Multi-task checkpoint detected (both heads present)")
        elif has_country_head:
            print("\n⚠️ Only country_head found, coordinate_head missing")
        else:
            print("\n⚠️ Old checkpoint detected (no custom heads)")
    
    # Load the weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Checkpoint loaded successfully ({'regional' if is_reg else 'country'})")
    except Exception as e:
        print(f"⚠️ Strict load failed: {e}")
        print("Trying non-strict load...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"\n📊 Load summary:")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            for m in missing[:5]:  # Show first 5
                print(f"    - {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing)-5} more")
        
        print(f"\n  Unexpected keys: {len(unexpected)}")
        if unexpected:
            for u in unexpected[:5]:  # Show first 5
                print(f"    - {u}")
            if len(unexpected) > 5:
                print(f"    ... and {len(unexpected)-5} more")
    
    model.to(device)
    model.eval()
    return model, checkpoint

def diagnose_model(model, checkpoint):
    print("\n🔍 MODEL DIAGNOSTICS 🔍")
    
    # 1. Check model structure
    print(f"Country head shape: {model.country_head.weight.shape}")
    print(f"Coordinate head shape: {model.coordinate_head.weight.shape}")
    
    # 2. Check coordinate head outputs
    test_input = torch.randn(1, 2048).to(DEVICE)  # Random features
    coords = model.coordinate_head(test_input)
    print(f"\nRandom feature -> Coordinates: {coords}")
    
    # 3. Check if tanh is applied correctly
    print(f"After tanh: {torch.tanh(coords)}")
    
    # 4. What happens with all-zero features?
    zero_input = torch.zeros(1, 2048).to(DEVICE)
    zero_coords = model.coordinate_head(zero_input)
    print(f"\nZero features -> Coordinates: {zero_coords}")
    print(f"Zero features -> After tanh: {torch.tanh(zero_coords)}")
    print(f"Denormalized: Longitude={torch.tanh(zero_coords)[0,0].item()*180:.1f}°, "
          f"Latitude={torch.tanh(zero_coords)[0,1].item()*90:.1f}°")
    
    # 5. Check checkpoint info
    if 'val_acc' in checkpoint:
        print(f"\nCheckpoint validation accuracy: {checkpoint['val_acc']*100:.2f}%")
    if 'val_coord_loss' in checkpoint:
        print(f"Checkpoint coordinate loss: {checkpoint['val_coord_loss']:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not IS_PRETTY:
        print("Using device:", device)

    try:
        ckpt_path = os.getenv("CKPT")
        model, ckpt = load_model_checkpoint(path=ckpt_path, device=device, num_classes=56, is_reg=False)
        load_main = True
    except:
        print("⚠️ CKPT environment variable is not configured or broken.")
        load_main = False
  
    try:
        ckpt_reg_path = os.getenv("CKPT_REG")
        model_reg, ckpt_reg = load_model_checkpoint(path=ckpt_reg_path, device=device, num_classes=7, is_reg=True)
        load_reg = True
    except:
        print("⚠️ CKPT_REG environment variable is not configured or broken.")
        load_reg = False
        

    sample_imgs = []
    sample_imgs_reg = []

    for i in enumerate(IMGS, 0):
        img_crop = crop_resize(Image.open(i[1]).convert("RGB"))
        img_stretch = stretch_resize(Image.open(i[1]).convert("RGB"))
        sample_imgs.append((img_crop, i[1]))
        sample_imgs.append((img_stretch, i[1]))

    for i in enumerate(IMGS, 0):
        sample_imgs_reg.append((img_crop, f"output/{i[1].split('/')[1]}"))
        sample_imgs_reg.append((img_stretch, f"output/{i[1].split('/')[1]}"))

        # img_stretch.show()
        # img_crop.show()

    # for i in enumerate(imgs, 0):
    #     img = resize(i[1]).convert("RGB")
    #     sample_imgs.append(img)
    if load_main and load_reg:
        predict_country(samples=sample_imgs, model=model, IS_PRETTY=IS_PRETTY, show_pictures=True)
        predict_region(samples=sample_imgs, model=model_reg, IS_PRETTY=IS_PRETTY, show_pictures=True)
    elif load_main:
        predict_country(samples=sample_imgs, model=model, IS_PRETTY=IS_PRETTY, show_pictures=True)
    elif load_reg:
        predict_region(samples=sample_imgs, model=model_reg, IS_PRETTY=IS_PRETTY, show_pictures=True)

    # diagnose_model(model, ckpt)

if __name__ == 'main':
    main()