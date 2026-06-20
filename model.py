import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 1. Define a lightweight CNN Architecture
class BloodCellCNN(nn.Module):
    def __init__(self, num_classes=2): # Defaulting to 2 (Normal vs Anomaly)
        super(BloodCellCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128x128 -> 64x64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

# 2. Image Preprocessing Transform
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 3. Inference Function
def predict_image(image_path, model, class_names=['Normal', 'Anomaly']):
    """Predicts if an image is Normal or Anomalous"""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    return class_names[predicted_idx.item()], confidence.item()