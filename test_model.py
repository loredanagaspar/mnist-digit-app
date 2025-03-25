# model/test_model.py

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from train import DigitCNN

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
image, label = test_data[0]

# Load model
model = DigitCNN()
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()

# Inference
image = image.unsqueeze(0)  # Add batch dimension
output = model(image)
predicted = torch.argmax(output, dim=1).item()
confidence = torch.max(F.softmax(output, dim=1)).item()

print(f"✅ True label: {label}")
print(f"🎯 Predicted: {predicted}")
print(f"📈 Confidence: {confidence:.2f}")
