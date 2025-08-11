import src.measurements.api as mpi
import src.peaks.api as ppi
import src.vae.api as vpi
import src.statistics.api as spi
import random
from sklearn.preprocessing import MultiLabelBinarizer
from src.cnn.training import Training
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import mlflow
import os
from config.loader import load_config
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, roc_auc_score
import logging

class RF:

    def __init__(self):
        self.splitted_keys = mpi.API().re_splitted_keys()
        self.rem_unique_keys = ppi.API().re_unique_dates()
        self.configs = load_config()
        self.use_meas = self.configs["randomforest"]["use_processed_measurements"]
        self.use_syn = self.configs["randomforest"]["use_processed_synthetics"]
        self.use_yielder = self.configs["randomforest"]["use_yielder"]

        self.rem_validation_keys = self.splitted_keys.loc[
            self.splitted_keys["type"] == "cnn_validation"
            ].reset_index(drop=True)[
            "datetime"].tolist()
        self.rem_training_keys = self.splitted_keys.loc[
            self.splitted_keys["type"] == "cnn_training"
            ].reset_index(drop=True)[
            "datetime"].tolist()
        self.rng = np.random.default_rng(seed=42)

        self.re_processed_measurements_validation = ppi.API().re_measurement(self.rem_validation_keys)

        if self.use_syn is True and self.use_meas is False:
            logging.warning("LOADING ONLY SYNTHETIC DATA...")
            self.res_unique_keys = vpi.API().re_unique_dates()
            shuffled_keys = self.rng.permutation(np.array(self.res_unique_keys)).tolist()[0:844]
            data = vpi.API().re_synhtetics(shuffled_keys)[
                ["datetime", "energy", "count", "identified_isotope"]
            ].reset_index(drop=True)
            self.label_df_validation, self.label_df_training = self.get_label_matrix(data)
            self.data_validation, self.data_training = self.transform_data_fro_training(data)

        elif self.use_meas is True and self.use_syn is False:
            logging.warning("LOADING ONLY MEASURED DATA...")
            data = ppi.API().re_measurement(self.rem_training_keys)[
                ["datetime", "energy", "count", "identified_isotope"]
            ].reset_index(drop=True)
            self.label_df_validation, self.label_df_training = self.get_label_matrix(data)
            self.data_validation, self.data_training = self.transform_data_fro_training(data)

        elif self.use_meas is True and self.use_syn is True:
            if self.use_yielder is True:
                pass
            else:
                data = self.static_loader()
                self.label_df_validation, self.label_df_training = self.get_label_matrix(data)
                self.data_validation, self.data_training = self.transform_data_fro_training(data)

        self.mlb = self.get_mlb()
        self.label_df_validation, _ = self.get_label_matrix("PASS")
        self.data_validation, _ = self.transform_data_fro_training("PASS")


        self.used_keys = []
        self.training_fpr_history = []
        self.training_tpr_history = []
        self.training_auc_history = []
        self.validation_fpr_history = []
        self.validation_tpr_history = []
        self.validation_auc_history = []

    def static_loader(self):
        logging.warning("LOADING MEASURED AND SYNTHETIC DATA...")
        self.res_unique_keys = vpi.API().re_unique_dates()
        shuffled_keys = self.rng.permutation(np.array(self.res_unique_keys)).tolist()
        data_syn = vpi.API().re_synhtetics(shuffled_keys)[
            ["datetime", "energy", "count", "identified_isotope"]
        ].reset_index(drop=True)
        data_syn["energy"] = data_syn["energy"].round(2)
        data_syn = (
            data_syn.groupby("datetime", group_keys=False)
            .apply(self.reset_energy)
            .reset_index(drop=True)
        )

        data_meas = ppi.API().re_measurement(self.rem_training_keys)[
            ["datetime", "energy", "count", "identified_isotope"]
        ].reset_index(drop=True)
        data_meas = (
            data_meas.groupby("datetime", group_keys=False)
            .apply(self.reset_energy)
            .reset_index(drop=True)
        )
        data = pd.concat([data_syn, data_meas], axis=0)
        return data

    def yield_data(self):
        logging.warning("LOADING MEASURED AND SYNTHETIC DATA...")
        self.res_unique_keys = vpi.API().re_unique_dates()
        shuffled_keys = self.rng.permutation(np.array(self.res_unique_keys)).tolist()
        for chunk in range(0, len(shuffled_keys), 1000):
            end = min(chunk + 1000, len(shuffled_keys))
            logging.warning(f"LOADING CHUNK SYNTHETIC DATA {chunk} END {end}...")
            selected_synthetics_keys = shuffled_keys[chunk:end]
            data_syn = vpi.API().re_synhtetics(selected_synthetics_keys)[
                ["datetime", "energy", "count", "identified_isotope"]
            ].reset_index(drop=True)
            data_syn["energy"] = data_syn["energy"].round(2)
            data_syn = (
                data_syn.groupby("datetime", group_keys=False)
                .apply(self.reset_energy)
                .reset_index(drop=True)
            )
            if chunk == 0:
                logging.warning("LOADING MEAS DATA ONCE...")
                data_meas = ppi.API().re_measurement(self.rem_training_keys)[
                    ["datetime", "energy", "count", "identified_isotope"]
                ].reset_index(drop=True)
                data_meas = (
                    data_meas.groupby("datetime", group_keys=False)
                    .apply(self.reset_energy)
                    .reset_index(drop=True)
                )
                data = pd.concat([data_syn, data_meas], axis=0)
            else:
                data = data_syn.copy()

            label_df_validation, label_df_training = self.get_label_matrix(data)
            data_validation, data_training = self.transform_data_fro_training(data)
            common_index = label_df_training.index.intersection(data_training.index)
            label_df_training = label_df_training.loc[common_index]
            data_training = data_training.loc[common_index]
            yield label_df_training, data_training

    def reset_energy(self, group):
        n = len(group)
        group = group.copy()
        group["energy"] = np.linspace(0, 8160, n)
        return group

    def __save_model(self):
        os.environ["AWS_ACCESS_KEY_ID"] = self.configs["minio"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.configs["minio"][
            "AWS_SECRET_ACCESS_KEY"
        ]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.configs["minio"][
            "MLFLOW_S3_ENDPOINT_URL"
        ]
        mlflow.set_tracking_uri(uri=self.configs["mlflow"]["uri"])
        mlflow.set_registry_uri(uri=self.configs["mlflow"]["uri"])
        mlflow.set_experiment("RandomForest")

        with mlflow.start_run(run_name="RF"):
            mlflow.log_param("mlb_classes", ",".join(self.mlb.classes_))
            mlflow.log_param("used_synthetics", self.use_syn)
            mlflow.log_param("used_measurements", self.use_meas)
            mlflow.sklearn.log_model(self.multi_rf, "model")
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

    def calculate_results(self, data, label_df, for_training=False):
        y_proba = self.multi_rf.predict_proba(data)
        y_proba_matrix = np.column_stack([prob[:, 1] for prob in y_proba])
        auc_macro = roc_auc_score(label_df, y_proba_matrix, average="macro") # self.data_validation
        print(f"Makro AUC: {auc_macro:.3f}")
        classes = self.mlb.classes_
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for idx, i in enumerate(classes):
            y_true = label_df.iloc[:, idx].values
            y_score = y_proba_matrix[:, idx]

            fpr_r, tpr_r, _ = roc_curve(y_true, y_score)

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

    def training(self):
        if self.use_yielder is False:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.multi_rf = MultiOutputClassifier(rf)
            self.used_keys.extend(self.data_training.index)
            logging.warning("TRAINING...")
            logging.warning(f"LEN OF TRAINING DATA {len(self.data_training)}")
            self.multi_rf.fit(self.data_training, self.label_df_training)
            self.calculate_results(self.data_training, label_df=self.label_df_training, for_training=True)
            self.calculate_results(self.data_validation, label_df=self.label_df_validation, for_training=False)
            self.__save_model()
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)
            self.multi_rf = MultiOutputClassifier(rf)
            logging.warning("LOADING CHUNKS...")
            for i, (label_df_training, data_training) in enumerate(self.yield_data()):
                if i > 0:
                    self.multi_rf.estimator.n_estimators += 100
                self.multi_rf.fit(data_training, label_df_training)
            self.calculate_results(data_training, label_df=label_df_training, for_training=True)
            self.calculate_results(self.data_validation, label_df=self.label_df_validation, for_training=False)
            self.__save_model()

    def transform_data_fro_training(self, data):
        data_validation = self.re_processed_measurements_validation.pivot_table(
            index="datetime", columns="energy", values="count",
            fill_value=0)
        try:
            data_training = data.pivot_table(index="datetime", columns="energy", values="count", fill_value=0)
        except:
            data_training = None
        return data_validation, data_training

    def get_label_matrix(self, data):
        df_filtered = self.re_processed_measurements_validation[
            self.re_processed_measurements_validation["identified_isotope"] != ""
        ]
        grouped_isotopes = (
            df_filtered.groupby("datetime")["identified_isotope"]
            .apply(lambda x: list(set(sum([s.split(",") for s in x], []))))
        )
        label_matrix = self.mlb.transform(grouped_isotopes.tolist())
        label_df_validation = pd.DataFrame(label_matrix, index=grouped_isotopes.index)
        label_df = None
        try:
            df_filtered = data[data["identified_isotope"] != ""]
            grouped_isotopes = (
                df_filtered.groupby("datetime")["identified_isotope"]
                .apply(lambda x: list(set(sum([s.split(",") for s in x], []))))
            )
            label_matrix = self.mlb.transform(grouped_isotopes.tolist())
            label_df = pd.DataFrame(label_matrix, index=grouped_isotopes.index)
        except:
            pass

        return label_df_validation, label_df

    def get_mlb(self):
        isotopes = spi.API().view_re_identified_isotopes()
        isos = isotopes.loc[
            (isotopes["identified_isotopes"] != "")
            & (isotopes["source_table"] == "processed_measurements")
            ]["identified_isotopes"].tolist()
        unique_isos = list(set(isos))
        base_classes = [s.split(",") for s in unique_isos]
        mlb = MultiLabelBinarizer()
        mlb.fit(base_classes)
        return mlb