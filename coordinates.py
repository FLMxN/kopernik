import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ResNetForImageClassification
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from PIL import Image

class CoordinatePredictor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load your existing trained model
        self.model = ResNetForImageClassification.from_pretrained(model_path)
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        
        # Remove the classification head to use as feature extractor
        self.feature_extractor = torch.nn.Sequential(
            *list(self.model.resnet.children())[:-1]  # Remove the final classifier
        )
        
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # This will be our coordinate predictor
        self.coord_predictor = None
    
    def extract_features(self, image_paths):
        """Extract features from images using your trained ResNet"""
        features = []
        
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Get features before classification
                feature_maps = self.feature_extractor(inputs.pixel_values)
                # Global average pooling
                features_vec = torch.mean(feature_maps, dim=[2, 3]).cpu().numpy()
                features.append(features_vec[0])
        
        return np.array(features)
    
    def train_coord_predictor(self, image_paths, coordinates):
        """Train a simple model to predict coordinates from features"""
        print("📊 Extracting features from training images...")
        features = self.extract_features(image_paths)
        
        # Train a simple regressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, coordinates, test_size=0.2, random_state=42
        )
        
        # Train Random Forest (you can use any regressor)
        self.coord_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.coord_predictor.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.coord_predictor.score(X_train, y_train)
        test_score = self.coord_predictor.score(X_test, y_test)
        
        print(f"✅ Coordinate predictor trained!")
        print(f"   Train R²: {train_score:.3f}")
        print(f"   Test R²: {test_score:.3f}")
        
        return test_score
    
    def predict_coordinates(self, image_path):
        if self.coord_predictor is None:
            print("❌ Coordinate predictor not trained yet!")
            return None
        
        features = self.extract_features([image_path])
        coords = self.coord_predictor.predict(features)[0]
        
        return coords
    
if __name__ == "__main__":
    coord_predictor = CoordinatePredictor("D:/resnet50-finetuned")
    coord_predictor.train_coord_predictor(image_paths=['pics/image.png'], coordinates=coord_predictor.extract_features(image_paths=['pics/image.png']))
    coord_predictor.predict_coordinates(image_path='pics/image.jpg')