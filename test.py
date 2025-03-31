# test.py
import sys, os
sys.path.append(os.path.abspath("."))  

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from model.train import DigitCNN 

# === Load config ===
MODEL_PATH = "model/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load MNIST test set ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# === Load trained model ===
model = DigitCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Manual test ===
with torch.no_grad():
    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        conf = torch.max(F.softmax(output, dim=1)).item()

        print(f"[{i+1}] Prediction: {pred} | Confidence: {conf:.2%} | True label: {label.item()}")

        if i >= 9:  # Show 10 predictions max
            break
