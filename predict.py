import torch
from torchvision import transforms
from PIL import Image
from model import ASLModel
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = sorted(os.listdir("asl_data/asl_alphabet_train"))

model = ASLModel(num_classes=len(classes))
model.load_state_dict(torch.load("saved_model/asl_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Prediction:", classes[predicted.item()])