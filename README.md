# Nuclide-Identification

Limitations:

- This works only for a specific files (.txt), spectra with 8160 channels
- Some Parameter are hardcoded for this specific case
- Not fully-automated pipeline

Even there are some limitations this project can be used as a layout and can be intergrated
to already existing project. Concepts of VAE, CNN, RandomForest, MLFlow, Data-Generator, Peak-Finding (prominance-based)
can be reused, by a new reimplementation.

This project provides:

- statisical method of prominence to label nuclides within gamma-spectra
- Variational Autoencoders (VAE) for generating synthetic gamma-spectra
- Random Forest and Convolutional Neural Network (CNN) (1-D) to classify nuclides based on labels
- SHAP-Analysis, THE KEY to identify nuclides withing multiple spectra, USE SHAP-Analysis on generated
latent space, to find out which nuclides are included in measurements. 

Backend:

- MLFlow (Model-Tracking)
- S3-Bucket (collecting mlflow-artifacts)
- PostgreSQL (gamma-spectra, nuclide-metadata, mlflow-backend)

Frontend:
- Plotly Dashboard to explore labaled spectra, raw data and some statistics
- MLFlow Dashboard

Structure:

```
.
├── docker-entrypoint-initdb.d\         # Init PostgreSQL script
├── cache\                              # Some cache data for MLFlow
├── config\                             # Environment Configurations
├── data\                               # Data-Folder
│   ├── measurements\                   # Gamma-Spectra
│   └── nuclides\                       # Scraped Nuclide-Data from API (https://www-nds.iaea.org).
├── plots\                              # Plots-Folder
├── src\                                # Packed Code
│   ├── cnn\                            # Convolutional Neural Network for classify Nuclides
│   ├── dashboard\                      # Exploration Dashboard (Frontend)
│   ├── generator\                      # Data-Generator based on VAE
│   ├── measurements\                   # Preparing Measurements (.csv > PostgreSQL)
│   ├── nuclide\                        # Nuclide Scrapper
│   ├── peaks\                          # Statistical Labeling
│   ├── pipeline\                       # The whole Pipeline
│   ├── randomforest\                   # Random Forest for classify Nuclides
│   ├── statistics\                     # Statistical API for PostgreSQL
│   └── vae\                            # Variational Autoencoder to generate synthetic data
└── tmp\    
```

## Terminology

For understanding the whole pipeline an overview about most important terms:

- measurement: raw gamma-spectra (here with 8160 channels)
- vae: variational autoencoder (here for generating samples out of measured gamma-spectra)
- cnn: convolutional neural network (here for classifying nuclides)
- generator: generating samples out of \sigma and \mu out of latent-space, also uses measurements to generate latent-space
- peaks: here labeling nuclides via statistical methods
- pipeline: measurements -> PostgreSQL -> labeling & interpolating -> processed_measurements 
-> splitting data for training VAE -> VAE training -> generating synthetic spectra -> latent space analysis ->
redo the pipeline (relabeling) based on latent space results (via SHAP)
- tablename **measurements**: raw measurements
- tablename **processed_measurements**: labeled and interpolated **measurements**
- tablename **processed_synthetics**: labeled generated synthetic data out of VAE
- tablename **processed_synthetics_latent_space**: used latent space variables to generate spectra (important for analysis further)
- tablename **splitted_keys_for_training_and_validation_pm**: key-based split for training and validation of VAE, CNN and RandomForest
- There is a prefix or a suffix **re_*** or ***_re**: means it is reprocessed labeling process

## Most important files

**Key Reasearch Conecept**
- Check all files here: src/vae/* 
  - VAE for synthetic spectra
- Check notebooks/vae_analysis.ipynb (HERE IS THE KEY OF SHAP-EXPLATION)

If trying to reconstruct this project check: config/* files
  
## Notebooks

folder notebooks/*

**cnn_analysis**: Plots for CNN, loss-function, AUC-macro, AUC for single nuclides ...

**cnn_validation**: Validation plots (Barplot with AUC-Scores, ROC-Curves), comparing results with 
synthetic-spectra and with measured-spectra...

**data_analysis**: Plots for measured and synthetic spectra. Distribution, Heatmap, raw measurements...

**examples**: Plots for savitzky-golay-filter, z-normalization and min-max-scaler

**randomforest_analysis**: Compare file **cnn_validation**

**vae_analysis**: Latent-Space violin-plots, SHAP-Values for latent-space, loss-function...

**validatriondataset_analysis**: Nuclide-Distribution of labeled validation-dataset 

## Environment Configurations

**Environment Variables**

| Name                       | Value                                                                                                    |
|----------------------------|----------------------------------------------------------------------------------------------------------|
| CONFIG_FILE                | the name of the used config-file for the envirnoment (EXAMPLE: config_local.yaml or config_service.yaml)) |
| config/config_local.yaml   | Check this file (for running on local machine)                                                           |
| config/config_service.yaml | Check this file (for dashboard)                                                                          |

There are some example entries within config_loca.yaml. Some detailed overview:

**vae**: Number of training epochs, nodes and connections for the network can be set up here

**generator**: There will be a generation of synthetic spectra based on min, 
max of the latent-space variabels

**cnn**: Important here, there is a configuration, which data should be used (synthetic or measured).
Also the number of nodes and connections can be configured here.

**randomforest**: Similar to **cnn**, configuration which data should be used (synthetic or measured)

**resplitter**: Splitting the dataset for CNN-Training and CNN-Validation (and RandomForest)

**repeakfinder**: Configurations for the prominence. Filtering out the nuclides, intensity of the nuclide -
Check the file src/peaks/refinder* - There is a method called "__get_prominence()". This method is getting the 
prominence value based on standard deviation, for more precise statistical labeling.

**peakfinder**: Initial nuclide labeling

**measurtements**: Defining the number of channels and interpolation-value for the energy (mean)

**postges_db**, **mlflow**, **minio**, **path** can be configured for services and datalocation


## Starting

1. Install dependencies

- Python version: 3.13.1
- pip: 24.3.1
- poetry as package-manager

```
pip install poetry 
poetry install
```

Install CUDA on your Machine or use CPU for training.

2. Run docker-compose (MlFlow, PostgreSQL, S3-Bucket, Adminer (Frontend for PostgreSQL) and Dashboard)

```
docker-compose -up -d
```

3. Use the file pipeline.py 

```
Pipeline().run(
    download_nuclides=True, 
    prepare_measurements=True, 
    find_measurements_peaks=True, 
)

# Will create tables: measurements.measurements, measurements.processed_measurements  
# nuclide.nuclide and nuclide.ground_state
```

Works only for this kind of .txt files (.txt files should be inserted to data/measurements):
```
number of channels:  8160
Realtime:   113963,06   Livetime:   114063,12
-4,355441E-1+2,904070E-1*channel+5,651753E-8*channel^2+0,000000E+0*channel^3

energy in keV	counts
   -0,44	      27
   -0,15	      82
    0,15	     135
    0,44	     185
    0,73	     269
    1,02	     505
    1,31	    1102
```

This will download nuclide metadata and prepare measurements. If you got different type of 
spectra files you could insert your measurements into a table in PostgreSQL:

DB: nuclide

Schema: measurements

Tablename: measurements

Table-Structure:

| datetime | energy           | count |
|----------|------------------|-------|
| meas1    | 0                | 27
| meas1    | 0.4              | 87
| meas1    | 0.8              | 82


datetime: key for a whole spectra

energy: energy for every channel

count: counts for every channel

4. There is some dependency on views of the data within PostgreSQL. They actually needs to be created
manually.

Here are all views listed with bit of explanation:

---

**TABLENAME: view_basic_statistics**

This 

```
SELECT 'processed_measurements'::text AS source_table,
    count(processed_measurements.peak) FILTER (WHERE (processed_measurements.peak = true)) AS peaks_detected,
    count(DISTINCT processed_measurements.datetime) AS total_measurements
   FROM processed_measurements
UNION ALL
 SELECT 'processed_synthetics'::text AS source_table,
    count(processed_synthetics.peak) FILTER (WHERE (processed_synthetics.peak = true)) AS peaks_detected,
    count(DISTINCT processed_synthetics.datetime) AS total_measurements
   FROM processed_synthetics
UNION ALL
 SELECT 'measurements'::text AS source_table,
    NULL::bigint AS peaks_detected,
    count(DISTINCT measurements.datetime) AS total_measurements
   FROM measurements;
```
---






-- For datetime-based aggregations
CREATE INDEX IF NOT EXISTS idx_processed_measurements_datetime ON processed_measurements(datetime);
CREATE INDEX IF NOT EXISTS idx_processed_synthetics_datetime ON processed_synthetics(datetime);
CREATE INDEX IF NOT EXISTS idx_measurements_datetime ON measurements(datetime);

-- For filtering
CREATE INDEX IF NOT EXISTS idx_processed_measurements_peak ON processed_measurements(peak);
CREATE INDEX IF NOT EXISTS idx_processed_synthetics_peak ON processed_synthetics(peak);

---

SELECT datetime,
    count(*) AS row_count,
    sum(
        CASE
            WHEN peak THEN 1
            ELSE 0
        END) AS peak_sum
   FROM processed_measurements
  GROUP BY datetime
  ORDER BY datetime;



SELECT 
  'processed_measurements' AS source_table,
  COUNT(peak) FILTER (WHERE peak = TRUE) AS peaks_detected,
  COUNT(DISTINCT(datetime)) AS total_measurements
FROM processed_measurements

UNION ALL

SELECT 
  'processed_synthetics' AS source_table,
  COUNT(peak) FILTER (WHERE peak = TRUE) AS peaks_detected,
  COUNT(DISTINCT(datetime)) AS total_measurements
FROM processed_synthetics

UNION ALL

SELECT 
  'measurements' AS source_table,
  NULL AS peaks_detected,
  COUNT(DISTINCT(datetime)) AS total_measurements
FROM measurements


use --no-verify tag

