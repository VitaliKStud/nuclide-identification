import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import src.peaks.api as ppi
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score

dates = ppi.unique_dates()
dataset = ppi.measurement(dates[0:1000])


def normalize_spectrum(df):
    scaler_energy = MinMaxScaler()
    scaler_count = MinMaxScaler()

    # Reshape to (1024, 1) for sklearn
    energy_scaled = scaler_energy.fit_transform(
        df["energy"].values.reshape(-1, 1)
    ).flatten()
    count_scaled = scaler_count.fit_transform(
        df["count"].values.reshape(-1, 1)
    ).flatten()

    return np.stack([energy_scaled, count_scaled])


# Suppose you have a DataFrame with all measurements, grouped by spectra
spectra = dataset.groupby("datetime").apply(normalize_spectrum)

# Group all identified isotopes per spectrum (datetime)
grouped_labels = dataset.groupby("datetime")["identified_isotope"].apply(
    lambda x: list(set(i for i in x if pd.notnull(i) and i != ""))
)


# Apply MultiLabelBinarizer to convert to binary matrix
mlb = MultiLabelBinarizer()
y_labels = mlb.fit_transform(grouped_labels)
print("Classes:", mlb.classes_)


# For multi-class, no reshaping to (N, 8160) is necessary
# If each sample corresponds to a single class, keep y_encoded as a 1D array.


X = np.stack(spectra.to_list())


# plt.plot(dataset[0:8160]["energy"], dataset[0:8160]["count"])
# plt.show()
# plt.plot(X[0][0], X[0][1])
# plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.2, random_state=10
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)  # (N, 1, 1024)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(
    y_train, dtype=torch.float32
)  # For multi-class, labels should be integer
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define Dataset class
class GammaDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = GammaDataset(X_train, y_train)
test_dataset = GammaDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# Define the CNN model for multi-class classification
class NuclideCNN(nn.Module):
    def __init__(self, num_classes):
        super(NuclideCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(256 * 1016, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)  # Multi-class output

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # No sigmoid here (CrossEntropyLoss handles that)
        return x


# Model setup
num_classes = y_labels.shape[1]  # Number of unique classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NuclideCNN(num_classes=num_classes).to(device)

pos_weight = (y_train == 0).sum(dim=0) / (y_train == 1).sum(dim=0)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(500):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(
            outputs, y_batch
        )  # CrossEntropyLoss automatically applies softmax
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

# Move tensors to CPU and convert to numpy
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

# Concatenate all batches
y_pred = np.vstack(all_preds)
y_true = np.vstack(all_targets)

# Compute macro and micro F1 scores
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

# Also optional:
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

print(f"F1 (Macro): {f1_macro:.4f} | F1 (Micro): {f1_micro:.4f}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")

# fig, axes = plt.subplots(20, 2, figsize=(6, 10 * 2))
#
# for idx in range(20):
#     axes[idx, 0].bar(range(len(y_true[idx])), y_true[idx], color='blue')
#     axes[idx, 0].set_title(f"True Labels - Sample {idx}")
#     axes[idx, 0].set_ylim(0, 1.2)
#
#     axes[idx, 1].bar(range(len(y_pred[idx])), y_pred[idx], color='green')
#     axes[idx, 1].set_title(f"Predicted Labels - Sample {idx}")
#     axes[idx, 1].set_ylim(0, 1.2)
#
# plt.tight_layout()
# plt.show()
