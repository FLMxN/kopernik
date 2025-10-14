from transformers import AutoConfig
import json

model_path = "D:/resnet50-finetuned"

print("🔍 Checking model configuration...")
config = AutoConfig.from_pretrained(model_path)

print(f"Number of labels: {config.num_labels}")
print(f"id2label mapping: {config.id2label}")
print(f"label2id mapping: {config.label2id}")

# Check if config file exists and what's in it
import os
config_file = os.path.join(model_path, "config.json")
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    print(f"\n📋 Raw config.json content:")
    print(json.dumps(config_data, indent=2))
else:
    print("❌ config.json not found!")

from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained(model_path)
print("Size config:", image_processor.size)
print("Do center crop:", getattr(image_processor, 'do_center_crop', False))
print("Crop size:", getattr(image_processor, 'crop_size', None))

# from datasets import load_dataset

# # Load your dataset
# dataset = load_dataset("stochastic/random_streetview_images_pano_v0.0.2")

# # Count rows with specific label
# label_to_count = "KR"  # replace with your label
# count = len(dataset["train"].filter(lambda example: example["country_iso_alpha2"] == label_to_count))
# print(f"Rows with label '{label_to_count}': {count}")