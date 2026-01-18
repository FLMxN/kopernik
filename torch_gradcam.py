import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class GradCAM:
    """
    Grad-CAM implementation for visualizing model predictions.
    Shows which regions of an image are important for a specific prediction.
    """
    
    def __init__(self, model, target_layer_name, device='cuda'):
        """
        Args:
            model: PyTorch model (must be eval mode)
            target_layer_name: name of layer to extract activations from (e.g., 'layer4')
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.device = device
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self._register_hooks(target_layer_name)
    
    def _register_hooks(self, target_layer_name):
        """Register forward and backward hooks on target layer"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find the target layer in ResNet
        if hasattr(self.model, 'resnet'):
            target_layer = getattr(self.model.resnet, target_layer_name)
        else:
            target_layer = getattr(self.model, target_layer_name)
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        self.target_layer = target_layer
    
    def generate(self, input_image, class_idx, eigen_smooth=False):
        """
        Generate Grad-CAM heatmap for a specific class.
        
        Args:
            input_image: tensor of shape (1, 3, H, W)
            class_idx: target class index for which to generate CAM
            eigen_smooth: whether to smooth the heatmap
        
        Returns:
            heatmap: numpy array of shape (H, W) with values in [0, 1]
        """
        
        # Forward pass
        if hasattr(self.model, 'resnet'):
            # For ResNet50Country or ResNet50Region
            logits = self.model(input_image)
            if isinstance(logits, tuple):
                logits = logits[0]  # Get country/region logits, not coordinates
        else:
            logits = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target_score = logits[:, class_idx].sum()
        target_score.backward()
        
        # Compute Grad-CAM
        # Shape: (batch, channels, height, width)
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Compute weights: average gradient across spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted activation map
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def visualize(self, original_image, heatmap, alpha=0.42, colormap='jet'):
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            original_image: PIL Image or numpy array (H, W, 3)
            heatmap: numpy array (H, W) with values in [0, 1]
            alpha: blending factor (0=original, 1=heatmap only)
            colormap: colormap name ('jet', 'hot', 'cool', etc.)
        
        Returns:
            result_image: PIL Image with heatmap overlay
        """
        
        # Convert original image to numpy if needed
        if isinstance(original_image, Image.Image):
            original_np = np.array(original_image)
        else:
            original_np = original_image
        
        # Resize heatmap to match original image dimensions
        h, w = original_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), 
                                            getattr(cv2, f'COLORMAP_{colormap.upper()}'))
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        result = (alpha * heatmap_colored + (1 - alpha) * original_np).astype(np.uint8)
        
        return Image.fromarray(result)


def apply_gradcam(model, image_path, class_idx, device='cuda', 
                  alpha=0.42, colormap='jet', target_layer='layer4'):
    """
    Convenience function to apply Grad-CAM to an image.
    Preserves original image resolution in output.
    
    Args:
        model: PyTorch model
        image_path: path to image file
        class_idx: target class index
        id2label_map: dictionary mapping class index to label
        device: 'cuda' or 'cpu'
        alpha: blending factor for visualization (0=original, 1=heatmap only)
        target_layer: which ResNet layer to use ('layer1', 'layer2', 'layer3', 'layer4')
        colormap: OpenCV colormap name. Options:
            'jet' (default, blue-green-red), 'hot' (black-red-yellow-white),
            'cool' (cyan-magenta), 'spring' (magenta-yellow), 'summer' (green-yellow),
            'autumn' (red-yellow), 'winter' (blue-green), 'viridis' (blue-green-yellow),
            'plasma' (purple-orange), 'inferno' (black-purple-orange),
            'magma' (black-purple-white), 'cividis' (blue-yellow)
    
    Returns:
        original: PIL Image at original resolution
        heatmap: numpy array at original resolution
        visualization: PIL Image with overlay at original resolution
        label: class label
        confidence: confidence score
    """
    
    # Load original image at full resolution
    original_image = Image.open(image_path).convert('RGB')
    original_dims = original_image.size  # (width, height)
    
    # Preprocess for model (resize to 224x224)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)
    
    # Ensure model is in eval mode
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'resnet'):
            logits = model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
        else:
            logits = model(input_tensor)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer, device=device)
    
    # Generate heatmap (at 224x224)
    heatmap_224 = gradcam.generate(input_tensor, class_idx)
    
    # Resize heatmap back to original image dimensions
    heatmap_original = cv2.resize(heatmap_224, original_dims)
    
    # Create visualization with original resolution and specified colormap
    visualization = gradcam.visualize(original_image, heatmap_original, alpha=alpha, colormap=colormap)
    
    # Get prediction details
    probs = F.softmax(logits, dim=1)
    confidence = probs[0, class_idx].item()
    
    return {
        'original': original_image,
        'heatmap': heatmap_original,
        'visualization': visualization,
        'confidence': confidence
    }

def apply_gradcam_multi(model, image_path, class_indices, device='cuda',
                        alpha=0.42, target_layer='layer4', colormap='jet', combine=True):
    """
    Apply Grad-CAM for multiple class labels at once.
    
    Args:
        model: PyTorch model
        image_path: path to image file
        class_indices: list of class indices to visualize
        id2label_map: dictionary mapping class index to label
        device: 'cuda' or 'cpu'
        alpha: blending factor (0=original, 1=heatmap only)
        target_layer: which ResNet layer to use
        colormap: single colormap name applied to all heatmaps.
                 Options: 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
                          'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        combine: if True, blend all heatmaps into single image; if False, return separate images
    
    Returns:
        If combine=False:
            List of dicts, each with keys: 'visualization', 'heatmap', 'label', 'confidence'
        If combine=True:
            Single dict with: 'visualization' (blended), 'heatmaps' (list), 'labels' (list), 'confidences' (list)
    """
    
    # Load original image at full resolution
    original_image = Image.open(image_path).convert('RGB')
    original_dims = original_image.size
    original_np = np.array(original_image)
    
    # Preprocess for model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'resnet'):
            logits = model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
        else:
            logits = model(input_tensor)
    
    probs = F.softmax(logits, dim=1)
    
    # Initialize Grad-CAM once
    gradcam = GradCAM(model, target_layer, device=device)
    
    results = []
    heatmaps_list = []
    
    # Generate heatmap for each class
    for class_idx in class_indices:
        heatmap_224 = gradcam.generate(input_tensor, class_idx)
        heatmap_original = cv2.resize(heatmap_224, original_dims)
        
        visualization = gradcam.visualize(original_image, heatmap_original, alpha=alpha, colormap=colormap)
        
        confidence = probs[0, class_idx].item()
        
        results.append({
            'visualization': visualization,
            'heatmap': heatmap_original,
            'confidence': confidence
        })
        
        heatmaps_list.append(heatmap_original)
    
    if combine:
        # Blend all heatmaps onto original image with same colormap
        blended = original_np.astype(np.float32)
        
        for heatmap in heatmaps_list:
            # Apply colormap
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8),
                                               getattr(cv2, f'COLORMAP_{colormap.upper()}'))
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend with reduced alpha to see all overlays
            blend_alpha = alpha / len(class_indices)
            blended = blended * (1 - blend_alpha) + heatmap_colored * blend_alpha
        
        combined_image = Image.fromarray(blended.astype(np.uint8))
        
        return {
            'visualization': combined_image,
            'heatmaps': heatmaps_list,
            'confidences': [probs[0, idx].item() for idx in class_indices]
        }
    else:
        return results


def gradcam(model, image_path, class_idx, alpha, device, colormap, task):
    # Convert single index to list if needed
    if isinstance(class_idx, int):
        class_indices = [class_idx]
    else:
        class_indices = class_idx
    
    result = apply_gradcam_multi(model=model, image_path=image_path, class_indices=class_indices, alpha=alpha, device=device, colormap=colormap)
        
    # Save visualizations
    # result['visualization'].save(f'output/{task}_{image_path.split("/")[1]}')
    return result['visualization']

if __name__ == "__main__":
    # Example usage
    from torch_main import load_model_checkpoint
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.getenv("CKPT")
    
    # Load model
    model, _ = load_model_checkpoint(False, ckpt_path, device=device, num_classes=56)
    model.eval()
    
    # Apply Grad-CAM to an image
    image_path = "pics/image.png"
    class_idx = [52, 35]  # Example: visualize for Russia and Japan

    gradcam(model, image_path, class_idx, alpha=0.42, device=device, colormap='jet')