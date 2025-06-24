from src.cnn.dataset import MeasurementTraining, MeasurementValidation
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.cnn.cnn import CNN
import torch
import torch.nn as nn
import src.measurements.api as mpi
import mlflow
import src.vae.api as vpi
import src.statistics.api as spi
import src.peaks.api as ppi
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from config.loader import load_config
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Training:
    def __init__(
        self,
        use_processed_measuremnets=True,
        use_processed_synthetics=False,
        chunk_ps=2000,
    ):
        self.use_processed_measuremnets = use_processed_measuremnets
        self.use_processed_synthetics = use_processed_synthetics
        self.configs = load_config()

        self.mlb = self.__get_mlb_fitter()
        self.model = CNN(num_classes=len(self.mlb.classes_)).to("cuda")
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.chunk_ps = chunk_ps

        self.splitted_keys = mpi.API().splitted_keys()

        if self.use_processed_synthetics:
            self.synthetic_keys = vpi.API().unique_dates()[0:1001]
            self.len_synthetics = len(self.synthetic_keys)
        self.keys_cnn_training, self.keys_cnn_validation = self.__get_processed_keys()

        if self.use_processed_measuremnets:
            self.training_cnn_pm_dataset = ppi.API().measurement(self.keys_cnn_training)
        self.validation_cnn_pm_dataset = ppi.API().measurement(self.keys_cnn_validation)

        self.validation_cnn_pm = MeasurementTraining(
            self.validation_cnn_pm_dataset, self.keys_cnn_validation, self.mlb
        )
        self.validation_cnn_pm_loader = DataLoader(
            self.validation_cnn_pm, batch_size=128, shuffle=True
        )
        self.best_model = None
        self.best_validation_loss = 0
        self.used_keys = []

        self.training_loss_history = []
        self.training_entropy_loss_history = []
        self.validation_loss_history = []
        self.validation_entropy_loss_history = []

    def __get_processed_keys(self):
        keys_training = (
            self.splitted_keys.loc[self.splitted_keys["type"] == "cnn_training"]
            .reset_index(drop=True)["datetime"]
            .tolist()
        )
        keys_validation = (
            self.splitted_keys.loc[self.splitted_keys["type"] == "cnn_validation"]
            .reset_index(drop=True)["datetime"]
            .tolist()
        )

        return keys_training, keys_validation

    def __yield_loader(self):
        if self.use_processed_measuremnets:
            if self.use_processed_synthetics:
                pass
            else:
                self.training_cnn_pm = MeasurementTraining(
                    self.training_cnn_pm_dataset, self.keys_cnn_training, self.mlb
                )
                yield DataLoader(self.training_cnn_pm, batch_size=128, shuffle=True)
        if self.use_processed_synthetics:
            for chunk in range(0, self.len_synthetics, self.chunk_ps):
                end = min(chunk + self.chunk_ps, self.len_synthetics)
                selected_synthetics_keys = self.synthetic_keys[chunk:end]
                dataset = vpi.API().synthetic(selected_synthetics_keys)
                if chunk == 0 and self.use_processed_measuremnets is True:
                    dataset = pd.concat(
                        [dataset, self.training_cnn_pm_dataset],
                        ignore_index=True,
                        axis=0,
                    )
                    training_cnn_ps = MeasurementTraining(
                        dataset,
                        selected_synthetics_keys + self.keys_cnn_training,
                        self.mlb,
                    )
                else:
                    training_cnn_ps = MeasurementTraining(
                        dataset, selected_synthetics_keys, self.mlb
                    )
                yield DataLoader(training_cnn_ps, batch_size=128, shuffle=True)

    def __get_mlb_fitter(self):
        isotopes = spi.API().view_identified_isotopes()
        isos = isotopes.loc[
            (isotopes["identified_isotopes"] != "")
            & (isotopes["source_table"] == "processed_measurements")
        ]["identified_isotopes"].tolist()
        unique_isos = list(set(isos))
        base_classes = [s.split(",") for s in unique_isos]
        mlb = MultiLabelBinarizer()
        mlb.fit(base_classes)
        logging.warning(mlb.classes_)
        return mlb

    def __save_model(self):
        os.environ["AWS_ACCESS_KEY_ID"] = load_config()["minio"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = load_config()["minio"][
            "AWS_SECRET_ACCESS_KEY"
        ]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = load_config()["minio"][
            "MLFLOW_S3_ENDPOINT_URL"
        ]
        mlflow.set_tracking_uri(uri=load_config()["mlflow"]["uri"])
        mlflow.set_registry_uri(uri=load_config()["mlflow"]["uri"])
        mlflow.set_experiment("NuclideCNN")

        with mlflow.start_run(run_name="CNN"):
            mlflow.log_param("mlb_classes", ",".join(self.mlb.classes_))
            mlflow.log_param("scaler", self.validation_cnn_pm.__get_scaler__())
            mlflow.log_param("used_synthetics", self.use_processed_synthetics)
            mlflow.log_param("used_measurements", self.use_processed_synthetics)
            mlflow.log_param("best_validation_loss", self.best_validation_loss)

            best_model = CNN(num_classes=len(self.mlb.classes_)).to(
                self.configs["cnn"]["device"]
            )
            best_model.load_state_dict(self.best_model)
            best_model.eval()
            mlflow.pytorch.log_model(best_model.to("cuda"), "model_cuda")
            mlflow.pytorch.log_model(best_model.to("cpu"), "model_cpu")
            mlflow.log_dict(
                {"used_keys": [str(i) for i in self.used_keys]}, "artifacts.json"
            )

            for training_loss in self.training_loss_history:
                mlflow.log_metric("training_loss", training_loss)
            for validation_loss in self.validation_loss_history:
                mlflow.log_metric("validation_loss", validation_loss)
            for training_entropy_loss in self.training_entropy_loss_history:
                mlflow.log_metric("training_entropy_loss", training_entropy_loss)
            for validation_entropy_loss in self.validation_entropy_loss_history:
                mlflow.log_metric("validation_entropy_loss", validation_entropy_loss)

    def cnn_training(self):
        for epoch in range(20):
            probs_all, y_train_all = [], []
            epoch_loss = 0
            len_loader = 0
            print(epoch)
            for loader in self.__yield_loader():
                self.model.train()
                for i, (x_train, keys, y_train) in tqdm(
                    enumerate(loader), total=len(loader)
                ):
                    self.used_keys.append(keys)
                    x_train = x_train.float().unsqueeze(1).to("cuda")
                    y_train = y_train.float().squeeze(1).to("cuda")

                    self.optimizer.zero_grad()
                    outputs = self.model(x_train)
                    probs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, y_train)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    probs_all.extend(probs.detach().cpu().numpy())
                    y_train_all.extend(y_train.detach().cpu().numpy())
                    len_loader += len(x_train)

            avg_epoch_loss = epoch_loss / len_loader
            self.training_entropy_loss_history.append(avg_epoch_loss)
            probs_all_numpy = np.array(probs_all)
            y_train_all_numpy = np.array(y_train_all)
            macro_roc_auc_ovr = roc_auc_score(
                y_train_all_numpy,
                probs_all_numpy,
                average=None,
            )
            mean = np.nanmean(macro_roc_auc_ovr)
            self.training_loss_history.append(mean)
            self.__cnn_validation()
        self.__save_model()

    def __cnn_validation(self):
        probs_all_validation = []
        y_train_all_validation = []
        epoch_loss = 0
        loop_validation = tqdm(enumerate(self.validation_cnn_pm_loader))
        for i, (validaition_lead) in loop_validation:
            for idx_lead, x_valid in enumerate(validaition_lead[0]):
                self.model.eval()
                keys = validaition_lead[1][idx_lead]
                y_valid = validaition_lead[2][idx_lead].to("cuda")
                y_valid = y_valid.float().squeeze(1).to("cuda")
                x_valid = x_valid.float().to("cuda").unsqueeze(0).unsqueeze(0)
                outputs = self.model(x_valid)
                probs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, y_valid)
                epoch_loss += loss.item()
                probs_all_validation.append(probs.to("cpu").detach().numpy()[0])
                y_train_all_validation.append(y_valid.to("cpu").detach().numpy()[0])
        avg_epoch_loss = epoch_loss / self.validation_cnn_pm_loader.__len__()
        self.validation_entropy_loss_history.append(avg_epoch_loss)
        probs_all_numpy = np.array(probs_all_validation)
        y_train_all_numpy = np.array(y_train_all_validation)
        macro_roc_auc_ovr = roc_auc_score(
            y_train_all_numpy,
            probs_all_numpy,
            average=None,
        )
        mean = np.nanmean(macro_roc_auc_ovr)
        self.validation_loss_history.append(mean)
        if mean > self.best_validation_loss:
            self.best_validation_loss = mean
            self.best_model = self.model.state_dict()
