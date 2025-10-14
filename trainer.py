# safe_resnet50_trainer.py
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch, torchvision
from torch import nn
from pathlib import Path
import gc

# ---------------- CONFIG ----------------
DATASET_NAME = "stochastic/random_streetview_images_pano_v0.0.2"
MODEL_NAME = "microsoft/resnet-50"
OUTPUT_DIR = Path("D:/resnet50-finetuned")
BATCH_SIZE = 4       # start tiny to fit 4GB VRAM
NUM_EPOCHS = 3
FP16 = True          # mixed precision
IMG_SIZE = 512       # smaller image → less VRAM
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------

torch.cuda.empty_cache()
gc.collect()
torch.manual_seed(SEED)

# --- 1️⃣ Load dataset (train-only)
full_dataset = load_dataset(DATASET_NAME)["train"]

# --- 2️⃣ Train/validation split
train_idx, val_idx = train_test_split(list(range(len(full_dataset))),
                                      test_size=0.2,
                                      shuffle=True,
                                      random_state=SEED)

train_dataset = full_dataset.select(train_idx)
val_dataset = full_dataset.select(val_idx)

# --- 3️⃣ Determine number of classes FIRST
# using 'country_iso_alpha2' as example label
labels = list(set(full_dataset["country_iso_alpha2"]))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}
num_labels = len(labels)  # Use actual count, not hardcoded 56

print(f"Number of classes: {num_labels}")

# --- 4️⃣ Load model with correct configuration
model = ResNetForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # This allows classifier size mismatch
)

# model = torchvision.models.resnet50(pretrained=False)

model = model.to(DEVICE)

# --- 5️⃣ Load processor
processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME, 
    do_resize=True, 
    size=IMG_SIZE,
    do_center_crop=False
)

# --- 6️⃣ FIXED preprocessing function
def preprocess_images(examples):
    # Process images
    processed = processor(
        [img.convert("RGB") for img in examples["image"]],
        return_tensors="pt"
    )
    
    labels_int = [label2id[l] for l in examples["country_iso_alpha2"]]
    return {
        "pixel_values": processed["pixel_values"],  # Keep as batch tensor
        "labels": torch.tensor(labels_int)  # Use "labels" key (required by Trainer)
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
    # Remove processing_class as it's not needed with set_transform
)

# --- 🔟 Train
print("Starting training...")
trainer.train()

# --- 1️⃣1️⃣ Save model
trainer.save_model(str(OUTPUT_DIR.absolute()))
print(f"Model saved to: {OUTPUT_DIR.absolute()}")

processor.save_pretrained("D:/resnet50-finetuned-process")
print("Processor saved with the model!")