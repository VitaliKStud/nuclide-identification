from src.cnn.dataset import MeasurementTraining
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
from sklearn.metrics import roc_auc_score
from config.loader import load_config
import logging
import os
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, auc


class Training:
    def __init__(
            self,
            use_processed_measuremnets=True,
            use_processed_synthetics=False,
            use_re_processed_data=False,
            chunk_ps=500,
    ):
        """
        The Training class of the cnn.

        :param use_processed_measuremnets: Will use labeled and transformed measurements for training
        :type use_processed_measuremnets: bool
        :param use_processed_synthetics: Will use labeled and transformed synthetics for training
        :type use_processed_synthetics: bool
        :param use_re_processed_data: Will use data of second labeling-step for training
        :type use_re_processed_data: bool
        :param chunk_ps: Depends on your local RAM, will train and load data in the size of the chunks
        :type chunk_ps: int
        """
        self.use_processed_measuremnets = use_processed_measuremnets
        self.use_processed_synthetics = use_processed_synthetics
        self.use_re_processed_data = use_re_processed_data
        self.configs = load_config()

        self.mlb = self.__get_mlb_fitter()
        self.model = CNN(num_classes=len(self.mlb.classes_)).to("cuda")
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.chunk_ps = chunk_ps

        if self.use_re_processed_data is True:
            logging.warning("LOADING RESPLITTED KEYS")
            self.splitted_keys = mpi.API().re_splitted_keys()
        else:
            logging.warning("LOADING SPLITTED KEYS")
            self.splitted_keys = mpi.API().splitted_keys()

        if self.use_processed_synthetics:
            if self.use_re_processed_data is True:
                logging.warning("LOADING: RELABLED SYNTHETIC KEYS")
                rng = np.random.default_rng(seed=42)
                loaded_keys = vpi.API().re_unique_dates()
                self.synthetic_keys = rng.permutation(np.array(loaded_keys)).tolist()[0:844]
            else:
                logging.warning("LOADING: SYNTHETIC KEYS")
                self.synthetic_keys = random.shuffle(vpi.API().unique_dates())[0:844]
            self.len_synthetics = len(self.synthetic_keys)
        self.keys_cnn_training, self.keys_cnn_validation = self.__get_processed_keys()

        if self.use_processed_measuremnets:
            if self.use_re_processed_data is True:
                logging.warning("LOADING: RELABLED MEASUREMENTS")
                self.training_cnn_pm_dataset = ppi.API().re_measurement(self.keys_cnn_training)
            else:
                logging.warning("LOADING: MEASUREMENTS")
                self.training_cnn_pm_dataset = ppi.API().measurement(self.keys_cnn_training)

        if self.use_re_processed_data is True:
            logging.warning("LOADING: RELABLED VALIDATION DATA")
            self.validation_cnn_pm_dataset = ppi.API().re_measurement(self.keys_cnn_validation)
        else:
            self.validation_cnn_pm_dataset = ppi.API().measurement(self.keys_cnn_validation)
            logging.warning("LOADING: VALIDATION DATA")

        self.validation_cnn_pm = MeasurementTraining(
            self.validation_cnn_pm_dataset, self.keys_cnn_validation, self.mlb
        )
        self.validation_cnn_pm_loader = DataLoader(
            self.validation_cnn_pm, batch_size=128, shuffle=True,
        )
        self.best_model = None
        self.best_validation_loss = 0
        self.used_keys = []

        self.training_loss_macro = []
        self.training_loss_micro = []
        self.training_entropy_loss_history = []
        self.training_tpr_history = []
        self.training_fpr_history = []
        self.training_auc_history = []

        self.validation_loss_macro = []
        self.validation_loss_micro = []
        self.validation_entropy_loss_history = []
        self.validation_tpr_history = []
        self.validation_fpr_history = []
        self.validation_auc_history = []

    def __get_processed_keys(self):
        """
        Loading Keys for Training and Validation
        """
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
        """
        Yielding the data-set chunk-way. Depends on which data-set has been selected through the
        initialization of this class.
        """
        if self.use_processed_measuremnets:
            if self.use_processed_synthetics is True:
                pass
            else:
                self.training_cnn_pm = MeasurementTraining(
                    self.training_cnn_pm_dataset, self.keys_cnn_training, self.mlb
                )
                yield DataLoader(self.training_cnn_pm, batch_size=64, shuffle=True)
        if self.use_processed_synthetics is True:
            for chunk in range(0, self.len_synthetics, self.chunk_ps):
                end = min(chunk + self.chunk_ps, self.len_synthetics)
                selected_synthetics_keys = self.synthetic_keys[chunk:end]
                if self.use_re_processed_data:
                    logging.warning("LOADING: RELABLED SYNTHETIC DATA FOR TRAINING...")
                    dataset = vpi.API().re_synhtetics(selected_synthetics_keys)
                    logging.warning(f"LEN SYNTHETIC RELABLED DATA: {len(dataset)}")
                else:
                    logging.warning("LOADING: SYNTHETIC DATA FOR TRAINING...")
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
                yield DataLoader(training_cnn_ps, batch_size=64, shuffle=True)

    def __get_mlb_fitter(self):
        """
        Fitting and creating a MultiLabelBinarizer for all nuclides.
        """
        if self.use_re_processed_data is True:
            logging.warning("LOADING: RELABLED IDENTIFIED ISOTOPES IN DATA")
            isotopes = spi.API().view_re_identified_isotopes()
        else:
            logging.warning("LOADING: IDENTIFIED ISOTOPES IN DATA")
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
        """
        Saving the trained model based on the validation-data-set. Saving artifacts and metrics into MLFlow
        """
        os.environ["AWS_ACCESS_KEY_ID"] = self.configs["minio"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.configs["minio"][
            "AWS_SECRET_ACCESS_KEY"
        ]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.configs["minio"][
            "MLFLOW_S3_ENDPOINT_URL"
        ]
        mlflow.set_tracking_uri(uri=self.configs["mlflow"]["uri"])
        mlflow.set_registry_uri(uri=self.configs["mlflow"]["uri"])
        mlflow.set_experiment("NuclideCNN")

        with mlflow.start_run(run_name="CNN"):
            mlflow.log_param("mlb_classes", ",".join(self.mlb.classes_))
            mlflow.log_param("scaler", self.validation_cnn_pm.__get_scaler__())
            mlflow.log_param("used_synthetics", self.use_processed_synthetics)
            mlflow.log_param("used_measurements", self.use_processed_measuremnets)
            mlflow.log_param("best_validation_loss", self.best_validation_loss)

            best_model = CNN(num_classes=len(self.mlb.classes_)).to(
                self.configs["cnn"]["device"]
            )
            best_model.load_state_dict(self.best_model)
            best_model.eval()
            mlflow.pytorch.log_model(best_model.to("cuda"), "model_cuda")
            mlflow.pytorch.log_model(best_model.to("cpu"), "model_cpu")
            mlflow.log_dict(
                {
                    "used_keys": [str(i) for i in self.used_keys],
                    "training_tpr": self.training_tpr_history,
                    "training_fpr": self.training_fpr_history,
                    "training_auc": self.training_auc_history,
                    "validation_tpr": self.validation_tpr_history,
                    "validation_fpr": self.validation_fpr_history,
                    "validation_auc": self.validation_auc_history,
                }, "artifacts.json"
            )

            for training_loss in self.training_loss_macro:
                mlflow.log_metric("training_macro_loss", training_loss)
            for training_micro_loss in self.training_loss_micro:
                mlflow.log_metric("training_micro_loss", training_micro_loss)
            for validation_loss in self.validation_loss_macro:
                mlflow.log_metric("validation_macro_loss", validation_loss)
            for training_entropy_loss in self.training_entropy_loss_history:
                mlflow.log_metric("training_entropy_loss", training_entropy_loss)
            for validation_entropy_loss in self.validation_entropy_loss_history:
                mlflow.log_metric("validation_entropy_loss", validation_entropy_loss)

    def __track_roc_curve_values(self, y_train_all_numpy, probs_all_numpy, for_training=False):
        """
        Calculation of the ROC curve for all classes for every epoch (validation and training)
        """
        classes = self.mlb.classes_
        n_classes = y_train_all_numpy.shape[1]
        if n_classes != len(classes):
            logging.warning("Number of classes does not match number of classes")
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for idx, i in enumerate(classes):
            fpr_r, tpr_r, _ = roc_curve(y_train_all_numpy[:, idx], probs_all_numpy[:, idx])

            fpr[i] = fpr_r.tolist()
            tpr[i] = tpr_r.tolist()
            roc_auc[i] = auc(fpr_r, tpr_r)
        if for_training is True:
            self.training_fpr_history.append(fpr)
            self.training_tpr_history.append(tpr)
            self.training_auc_history.append(roc_auc)
        else:
            self.validation_fpr_history.append(fpr)
            self.validation_tpr_history.append(tpr)
            self.validation_auc_history.append(roc_auc)

    def cnn_training(self):
        """
        Training through 100 Epoch (if change is needed, could be set to other value)
        """
        for epoch in range(100): # NUMBER OF EPOCHS
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
            training_loss_micro = roc_auc_score(
                y_train_all_numpy,
                probs_all_numpy,
                average="micro",
            )
            self.training_loss_micro.append(training_loss_micro)
            logging.warning(f"TRAINING LOSS MICRO: {training_loss_micro}, TRAINING LOSS MACRO: {mean}")
            self.training_loss_macro.append(mean)
            self.__track_roc_curve_values(y_train_all_numpy=y_train_all_numpy, probs_all_numpy=probs_all_numpy,
                                          for_training=True)
            self.__cnn_validation()
        self.__save_model()

    def __cnn_validation(self):
        """
        Validating results for the CNN for every epoch and tracking via MLFlow
        """
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
        validation_loss_micro = roc_auc_score(
            y_train_all_numpy,
            probs_all_numpy,
            average="micro",
        )
        self.validation_loss_micro.append(validation_loss_micro)
        logging.warning(f"VALIDATION LOSS MICRO: {validation_loss_micro}, VALIDATION LOSS MACRO: {mean}")
        self.validation_loss_macro.append(mean)
        self.__track_roc_curve_values(y_train_all_numpy=y_train_all_numpy, probs_all_numpy=probs_all_numpy,
                                      for_training=False)
        if mean > self.best_validation_loss:
            self.best_validation_loss = mean
            self.best_model = self.model.state_dict()
