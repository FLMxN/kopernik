# safe_resnet50_trainer.py
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer, ResNetConfig
from sklearn.model_selection import train_test_split
import torch, torchvision
from torchvision import transforms
from torch import nn
from pathlib import Path
import gc
import numpy as np
import sys

# ---------------- CONFIG ----------------
DATASET_NAME = "stochastic/random_streetview_images_pano_v0.0.2"
MODEL_NAME = "microsoft/resnet-50"
OUTPUT_DIR = Path("D:/resnet50-finetuned_raw")
BATCH_SIZE = 4       # start tiny to fit 4GB VRAM
NUM_EPOCHS = 3
FP16 = True          # mixed precision
IMG_SIZE = 561       # smaller image → less VRAM
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------

torch.cuda.empty_cache()
gc.collect()

sys.setrecursionlimit(10000)

np.random.seed(42)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- 1️⃣ Load dataset (train-only)
full_dataset = load_dataset(DATASET_NAME)["train"]

# --- 2️⃣ Train/validation split
train_idx, val_idx = train_test_split(list(range(len(full_dataset))),
                                      test_size=0.1,
                                      shuffle=True,
                                      random_state=SEED)

train_dataset = full_dataset.select(train_idx)
val_dataset = full_dataset.select(val_idx)

# --- 3️⃣ Determine number of classes FIRST
labels = list(set(full_dataset["country_iso_alpha2"]))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
num_labels = len(labels)

print(f"Number of classes: {num_labels}")

# --- 4️⃣ Load model with correct configuration
# model = ResNetForImageClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=num_labels,
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True
# )

config = ResNetConfig(
    num_labels=num_labels,           
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True)

model = ResNetForImageClassification(config)
model = model.to(DEVICE)

# --- 1️⃣ Define same preprocessing as in transforms snippet
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 2️⃣ Preprocessing function
def preprocess_images(examples):
    images = []
    labels_int = []

    for img, label in zip(examples["image"], examples["country_iso_alpha2"]):
        img_cropped = img.crop((1017, 0, 2033, 561)).convert("RGB")
        img_tensor = preprocess(img_cropped)
        images.append(img_tensor)
        labels_int.append(label2id[label])
    
    return {
        "pixel_values": torch.stack(images),  # batch tensor
        "labels": torch.tensor(labels_int)
    }


# --- 7️⃣ Apply preprocessing
train_dataset.set_transform(preprocess_images)
val_dataset.set_transform(preprocess_images)

# --- 8️⃣ Training arguments
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR.absolute()),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    logging_dir="./logs",
    remove_unused_columns=False,
    fp16=FP16,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,  # Reduce memory usage
)

# --- 9️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# --- 🔟 Train
print("Starting training...")
trainer.train()

# --- 1️⃣1️⃣ Save model
trainer.save_model(str(OUTPUT_DIR.absolute()))
print(f"Model saved to: {OUTPUT_DIR.absolute()}")

processor.save_pretrained("D:/resnet50-finetuned_raw")
print("Processor saved with the model!")