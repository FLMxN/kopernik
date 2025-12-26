import torch
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoConfig
from PIL import Image
import PIL
import os
import torch.nn.functional as F
from torchvision import transforms

HEIGHT = 561
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(model, resized_img: Image.Image, processor=None, top_k=5, device=DEVICE):
        try:
            inputs = processor(resized_img, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print('HuggingFace AutoModel found')
        except:
            inputs = preprocess(resized_img).unsqueeze(0) #FIX THIS!!!!!!!!!!!!
            inputs = inputs.to(device)
            print('PyTorch model from checkpoint found')

        with torch.no_grad():
            try:
                logits = model(**inputs)
            except:
                logits = model(inputs)
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        if hasattr(model, 'config'):
            print("Using HuggingFace AutoConfig labels")
        else:
            print("Using PyTorch raw labels")

        print(f"\nPredictions via preprocessor:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            try:
                label = model.config.id2label[idx]
            except:
                label = model.id2label[idx]
            confidence = prob * 100
            print(f"  {i+1}. {label}: {prob:.4f} ({confidence:.2f}%)")
        
        try:
            top_label = model.config.id2label[top_indices[0]]
        except:
            top_label = model.id2label[top_indices[0]]
        top_confidence = top_probs[0]
        
        return top_label, top_confidence