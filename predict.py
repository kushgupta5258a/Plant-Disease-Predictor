import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import sys

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "model/model.pkl"
DATA_DIR = "data/dataset/PlantVillage"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# LOAD CLASS NAMES
# -----------------------
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Total classes: {num_classes}\n")

# -----------------------
# LOAD MODEL
# -----------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Replace classifier to match training
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!\n")

# -----------------------
# TRANSFORM (same as validation)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict(image_path):
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded: {image_path}")
        
        # Preprocess
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, 1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Display results
        print(f"\nPredicted Class: {predicted_class}")
        print(f"Confidence: {confidence_score:.4f} ({confidence_score*100:.2f}%)\n")
        
        # Show top 3 predictions
        top_3_probs, top_3_indices = torch.topk(probabilities, 3, 1)
        print("Top 3 Predictions:")
        for i, (prob, idx) in enumerate(zip(top_3_probs[0], top_3_indices[0])):
            print(f"  {i+1}. {class_names[idx.item()]} - {prob.item():.4f}")
        
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
    except Exception as e:
        print(f"Error: {e}")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image (change this path)
        image_path = input("Enter image path: ")
    
    predict(image_path)
