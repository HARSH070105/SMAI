import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# -----------------------------
# DATASET (make it harder → forces overfitting)
# -----------------------------
X, y = make_moons(n_samples=300, noise=0.35)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# -----------------------------
# SAME MODEL (only difference = dropout)
# -----------------------------
class Net(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),

            nn.Dropout(0.3) if use_dropout else nn.Identity(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Dropout(0.3) if use_dropout else nn.Identity(),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model(model, epochs=150):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = criterion(test_output, y_test)
            test_losses.append(test_loss.item())

    return train_losses, test_losses


def accuracy(model):
    model.eval()
    with torch.no_grad():
        preds_train = model(X_train).argmax(dim=1)
        preds_test = model(X_test).argmax(dim=1)

        train_acc = (preds_train == y_train).float().mean().item()
        test_acc = (preds_test == y_test).float().mean().item()

    return train_acc, test_acc


# -----------------------------
# TRAIN BOTH MODELS
# -----------------------------
model_no_dropout = Net(use_dropout=False)
model_dropout = Net(use_dropout=True)

print("Training WITHOUT dropout...")
train_no, test_no = train_model(model_no_dropout)

print("Training WITH dropout...")
train_do, test_do = train_model(model_dropout)

acc_train_no, acc_test_no = accuracy(model_no_dropout)
acc_train_do, acc_test_do = accuracy(model_dropout)

# -----------------------------
# RESULTS
# -----------------------------
print("\n=== RESULTS ===")
print(f"No Dropout  -> Train: {acc_train_no:.4f}, Test: {acc_test_no:.4f}")
print(f"Dropout     -> Train: {acc_train_do:.4f}, Test: {acc_test_do:.4f}")

# -----------------------------
# PLOT
# -----------------------------
plt.figure()

plt.plot(test_no, label="Test Loss (No Dropout)")
plt.plot(test_do, label="Test Loss (Dropout)")

plt.plot(train_no, linestyle="--", label="Train Loss (No Dropout)")
plt.plot(train_do, linestyle="--", label="Train Loss (Dropout)")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Same Network: Dropout vs No Dropout")
plt.legend()

plt.savefig("dropout_final.png")
plt.show()