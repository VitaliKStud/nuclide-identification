from src.cnn.dataset import (
    MeasurementTraining,
    MeasurementValidation,
    IterableMeasurementTraining,
)
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.cnn.cnn import CNN
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import src.measurements.api as mpi
import src.peaks.api as ppi
import src.statistics.api as spi
from sklearn.preprocessing import MultiLabelBinarizer
import src.vae.api as vpi

model = CNN(num_classes=7).to("cuda")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

splited_keys = mpi.API().splitted_keys()
datetime = (
    splited_keys.loc[splited_keys["type"] == "cnn_training"]
    .reset_index(drop=True)["datetime"]
    .tolist()
)

# datetime = vpi.API().unique_dates()
# dataset = vpi.API().synthetic(datetime)
dataset = ppi.API().measurement(dates=datetime)

isotopes = spi.API().view_identified_isotopes()
isos = isotopes.loc[
    (isotopes["identified_isotopes"] != "")
    & (isotopes["source_table"] == "processed_measurements")
]["identified_isotopes"].tolist()
unique_isos = list(set(isos))
base_classes = [s.split(",") for s in isos]
mlb = MultiLabelBinarizer()
mlb.fit(base_classes)
test_set = MeasurementTraining(
    dataset=dataset, datetimes=datetime, isotopes=isotopes, fitted_mlb=mlb
)
validation_set = MeasurementValidation()
loader = DataLoader(
    test_set, batch_size=128
)  # NO NEED TO SHUFFLE; NEEDS TO BE DONE MANUALLY VIA KEYS (FASTER)
validation_loader = DataLoader(validation_set, batch_size=128, shuffle=True)

for epoch in range(20):
    model.train()
    epoch_loss = 0
    probs_all, preds_all, y_train_all = [], [], []
    print(epoch)

    for i, (x_train, keys, y_train) in tqdm(enumerate(loader), total=len(loader)):
        x_train = x_train.float().unsqueeze(1).to("cuda")
        y_train = y_train.float().squeeze(1).to("cuda")

        optimizer.zero_grad()
        outputs = model(x_train)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        probs_all.extend(probs.detach().cpu().numpy())
        preds_all.extend(preds.detach().cpu().numpy())
        y_train_all.extend(y_train.detach().cpu().numpy())


probs_all_validation = []
preds_all_validation = []
y_train_all_validation = []
loop_validation = tqdm(enumerate(validation_loader))
for i, (validaition_lead) in loop_validation:
    for idx_lead, x_valid in enumerate(validaition_lead[0]):
        model.eval()
        keys = validaition_lead[1][idx_lead]
        y_valid = validaition_lead[2][idx_lead].to("cuda")
        x_valid = x_valid.float().to("cuda").unsqueeze(0).unsqueeze(0)
        print(y_valid)
        outputs = model(x_valid)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        probs_all_validation.append(probs.to("cpu").detach().numpy()[0])
        preds_all_validation.append(preds.to("cpu").detach().numpy()[0])
        y_train_all_validation.append(y_valid.to("cpu").detach().numpy()[0])


import numpy as np
import matplotlib.pyplot as plt

probs_all_numpy = np.array(probs_all_validation)
y_train_all_numpy = np.array(y_train_all_validation)
macro_roc_auc_ovr = roc_auc_score(
    y_train_all_numpy,
    probs_all_numpy,
    average=None,
)
class_labels = [f"Class {i}" for i in range(len(macro_roc_auc_ovr))]

plt.figure(figsize=(10, 6))
bars = plt.bar(
    class_labels,
    macro_roc_auc_ovr,
    color=["red" if np.isnan(r) else "skyblue" for r in macro_roc_auc_ovr],
)

# Add value labels
for bar, score in zip(bars, macro_roc_auc_ovr):
    height = bar.get_height()
    label = "NaN" if np.isnan(score) else f"{score:.2f}"
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height if not np.isnan(score) else 0.01,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.title("ROC AUC per Class")
plt.ylabel("ROC AUC Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Assuming you have these as numpy arrays
# probs_all_numpy: predicted probabilities (N samples x C classes)
# y_train_all_numpy: true binary labels (N samples x C classes)

classes = test_set.__getmlbclasses__()
n_classes = y_train_all_numpy.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))

for i in range(n_classes):
    if np.sum(y_train_all_numpy[:, i]) == 0:
        # Skip classes with no positive examples
        continue

    fpr[i], tpr[i], _ = roc_curve(y_train_all_numpy[:, i], probs_all_numpy[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

# Plot baseline
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve per Class")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

model.eval()
model.zero_grad()
energy_max = 0.34507313512321336 * 8160
energy_axis = np.arange(0, energy_max, 0.34507313512321336)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

for i in range(len(validaition_lead[0])):
    keys = validaition_lead[1][i]
    y_valid = validaition_lead[2][i].to("cuda")
    x_valid = validaition_lead[0][i].float().to("cuda").unsqueeze(0).unsqueeze(0)
    x_sample = x_valid.squeeze().detach().clone().to("cuda")
    x_sample.requires_grad = True
    x_input = x_sample.unsqueeze(0).unsqueeze(0)

    output = model(x_input)
    probs = torch.sigmoid(output)
    preds = (probs > 0.50).int()
    print(preds)

    plt.figure(figsize=(14, 5))
    color_map = plt.cm.get_cmap(
        "tab10", len(classes)
    )  # Use tab10 for up to 10 distinct colors

    for classe in range(len(classes)):
        score = output[0, classe]
        model.zero_grad()
        if x_sample.grad is not None:
            x_sample.grad.zero_()
        score.backward(retain_graph=True)

        if x_sample.grad is not None:
            saliency = (
                x_sample.grad.abs().cpu().numpy() * x_sample.detach().cpu().numpy()
            )
            # saliency = x_sample.grad.abs().cpu().numpy()
            plt.plot(
                energy_axis,
                saliency,
                label=f"Class {classe}",
                linewidth=1.5,
                color=color_map(classe),
            )
        else:
            print(f"No gradient for class {classe} at sample {i}")

    # Add the original sample spectrum
    plt.plot(
        energy_axis,
        x_sample.detach().cpu().numpy(),
        label="Input Spectrum",
        color="black",
        linestyle="--",
        zorder=0,
    )
    # plt.yscale('log')
    plt.ylim(0, 0.025)
    plt.title(
        f"Saliency Map Overlay | Sample: {keys}\nOutput: {output}\nPreds: {preds}\nTarget: {y_valid} \n {probs}"
    )
    plt.xlabel("Energy [keV or MeV]")
    plt.ylabel("Gradient Magnitude × Signal")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"tmp/overlay_{i}.png")
    plt.close()
