# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.cnn import DigitCNN

# Augmented MNIST
transform = transforms.Compose([
    transforms.RandomAffine(15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
val_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1000)

# Init model
model = DigitCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
model.train()
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    print(f"✅ Epoch {epoch + 1} complete")

# Save model
torch.save(model.state_dict(), "model/model.pt")
print("✅ Saved to model/model.pt")

# Evaluate
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

print(f"🎯 Accuracy: {correct / total:.2%}")
