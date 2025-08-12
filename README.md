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
- processed: labaled data

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

Dashboard won't work for the first time, but if the pipeline is finished.

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

Showing statistic about number of measured spectra, synthetic spectra and number of detected nuclides.

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
**TABLENAME: view_dist_measurements**

Showing the distribution of the counts for measurements, by 40 bins.

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
**TABLENAME: view_dist_processed_measurements**

Showing the distribution of the counts for processed_measurements, by 40 bins.

```
SELECT width_bucket(log((sub.count + (1)::double precision)), (0)::double precision, max_l.max_log, 40) AS log_bin,
    count(*) AS frequency
   FROM ( SELECT processed_measurements.count,
            log((processed_measurements.count + (1)::double precision)) AS log_val
           FROM processed_measurements) sub,
    LATERAL ( SELECT max(log((processed_measurements.count + (1)::double precision))) AS max_log
           FROM processed_measurements) max_l
  GROUP BY (width_bucket(log((sub.count + (1)::double precision)), (0)::double precision, max_l.max_log, 40))
  ORDER BY (width_bucket(log((sub.count + (1)::double precision)), (0)::double precision, max_l.max_log, 40));
```
---
**TABLENAME: view_dist_processed_synthetics**

Showing the distribution of the counts for processed_synthetics, by 40 bins.

```
SELECT width_bucket(log((sub.count + (1)::double precision)), (0)::double precision, max_l.max_log, 40) AS log_bin,
    count(*) AS frequency
   FROM ( SELECT processed_synthetics.count,
            log((processed_synthetics.count + (1)::double precision)) AS log_val
           FROM processed_synthetics) sub,
    LATERAL ( SELECT max(log((processed_synthetics.count + (1)::double precision))) AS max_log
           FROM processed_synthetics) max_l
  GROUP BY (width_bucket(log((sub.count + (1)::double precision)), (0)::double precision, max_l.max_log, 40))
  ORDER BY (width_bucket(log((sub.count + (1)::double precision)), (0)::double precision, max_l.max_log, 40));
```
---
**TABLENAME: view_found_isotopes_statistics**

Showing found nuclides over processed_measurements and processed_synthetics. The number of each nuclide
within this data. Only for initial labeling.

```
SELECT 'processed_measurements'::text AS source_table,
    processed_measurements.identified_isotope AS isotopes,
    count(processed_measurements.peak) FILTER (WHERE (processed_measurements.peak = true)) AS count_processed_measurements
   FROM processed_measurements
  GROUP BY processed_measurements.identified_isotope
UNION ALL
 SELECT 'processed_synthetics'::text AS source_table,
    processed_synthetics.identified_isotope AS isotopes,
    count(processed_synthetics.peak) FILTER (WHERE (processed_synthetics.peak = true)) AS count_processed_measurements
   FROM processed_synthetics
  GROUP BY processed_synthetics.identified_isotope;
```
---
**TABLENAME: view_identified_isotopes**

This table is important for training. It shows found nuclides within the data. Is important for 
MultiLabelClass Fit.

```
SELECT DISTINCT pm.identified_isotope AS identified_isotopes,
    'processed_measurements'::text AS source_table
   FROM processed_measurements pm
UNION ALL
 SELECT DISTINCT ps.identified_isotope AS identified_isotopes,
    'processed_synthetics '::text AS source_table
   FROM processed_synthetics ps;
```
---
**TABLENAME: view_identified_isotopes**

Counting found nuclides with processed_measurements per measurement

```
SELECT datetime,
    count(DISTINCT identified_isotope) AS row_count
   FROM processed_measurements
  GROUP BY datetime
  ORDER BY datetime;
```
---
**TABLENAME: view_isotope_per_ps**

Counting found nuclides with processed_synthetics per measurement

```
SELECT datetime,
    count(DISTINCT identified_isotope) AS row_count
   FROM processed_synthetics
  GROUP BY datetime
  ORDER BY datetime;
```
---
**TABLENAME: view_isotope_per_re_pm**

Counting found nuclides with re_processed_measurements per measurement

```
SELECT datetime,
    count(DISTINCT identified_isotope) AS row_count
   FROM re_processed_measurements
  GROUP BY datetime
  ORDER BY datetime;
```
---
**TABLENAME: view_isotope_per_re_ps**

Counting found nuclides with re_processed_synthetics per measurement

```
SELECT datetime,
    count(DISTINCT identified_isotope) AS row_count
   FROM re_processed_synthetics
  GROUP BY datetime
  ORDER BY datetime;
```
---
**TABLENAME: view_mean_measurement**

Mean value by energy

```
SELECT energy,
    avg(count) AS mean_count
   FROM processed_measurements
  GROUP BY energy;
```
---
**TABLENAME: view_measurements_counter**

Validation of number of channels per measurements

```
SELECT datetime,
    count(*) AS row_count
   FROM measurements
  GROUP BY datetime
  ORDER BY datetime;
```
---
**TABLENAME: view_min_max**

Extracting min-max count-values from different datasets (vae,
cnn_validation, cnn_training and processed_synthetics)

```
SELECT max(pm.count) AS max,
    min(pm.count) AS min,
    'processed_measurements_vae'::text AS source_table
   FROM processed_measurements pm
  WHERE (pm.datetime IN ( SELECT spm.datetime
           FROM splitted_keys_for_training_and_validation_pm spm
          WHERE (spm.type = 'vae'::text)))
UNION ALL
 SELECT max(pm.count) AS max,
    min(pm.count) AS min,
    'processed_measurements_cnn_validation'::text AS source_table
   FROM processed_measurements pm
  WHERE (pm.datetime IN ( SELECT spm.datetime
           FROM splitted_keys_for_training_and_validation_pm spm
          WHERE (spm.type = 'cnn_validation'::text)))
UNION ALL
 SELECT max(pm.count) AS max,
    min(pm.count) AS min,
    'processed_measurements_cnn_training'::text AS source_table
   FROM processed_measurements pm
  WHERE (pm.datetime IN ( SELECT spm.datetime
           FROM splitted_keys_for_training_and_validation_pm spm
          WHERE (spm.type = 'cnn_training'::text)))
UNION ALL
 SELECT max(ps.count) AS max,
    min(ps.count) AS min,
    'processed_synthetics'::text AS source_table
   FROM processed_synthetics ps;
```
---
**TABLENAME: view_pm_isotopes_found**

Counting and showing the labeled nuclides within processed_measurements

```
SELECT datetime,
    identified_isotope,
    count(*) AS row_count
   FROM processed_measurements
  GROUP BY datetime, identified_isotope
  ORDER BY datetime;
```
---
**TABLENAME: view_ps_isotopes_found**

Counting and showing the labeled nuclides within processed_synthetics

```
SELECT datetime,
    identified_isotope,
    count(*) AS row_count
   FROM processed_synthetics
  GROUP BY datetime, identified_isotope
  ORDER BY datetime;
```
---
**TABLENAME: view_re_identified_isotopes**

Counting found nuclides with processed_measurements per measurement (second labeling step)

```
SELECT DISTINCT pm.identified_isotope AS identified_isotopes,
    'processed_measurements'::text AS source_table
   FROM re_processed_measurements pm
UNION ALL
 SELECT DISTINCT ps.identified_isotope AS identified_isotopes,
    'processed_synthetics '::text AS source_table
   FROM re_processed_synthetics ps;
```
---
**TABLENAME: view_re_pm_isotopes_found**

Counting found nuclides with processed_measurements per measurement (second labeling step)

```
SELECT datetime,
    identified_isotope,
    count(*) AS row_count
   FROM re_processed_measurements
  GROUP BY datetime, identified_isotope
  ORDER BY datetime;
```
---
**TABLENAME: view_re_ps_isotopes_found**

Counting found nuclides with processed_synthetics per measurement (second labeling step)

```
SELECT datetime,
    identified_isotope,
    count(*) AS row_count
   FROM re_processed_synthetics
  GROUP BY datetime, identified_isotope
  ORDER BY datetime;
```
---
**TABLENAME: view_std_mean_min_max_statistics**

Basic statistics for measurements, procesed_measurements and processed_synthetics

```
SELECT 'measurements'::text AS source_table,
    stddev(measurements.energy) AS energy_std,
    avg(measurements.energy) AS energy_mean,
    min(measurements.energy) AS energy_min,
    max(measurements.energy) AS energy_max,
    stddev(measurements.count) AS count_std,
    avg(measurements.count) AS count_mean,
    min(measurements.count) AS count_min,
    max(measurements.count) AS count_max
   FROM measurements
UNION ALL
 SELECT 'processed_measurements'::text AS source_table,
    stddev(processed_measurements.energy) AS energy_std,
    avg(processed_measurements.energy) AS energy_mean,
    min(processed_measurements.energy) AS energy_min,
    max(processed_measurements.energy) AS energy_max,
    stddev(processed_measurements.count) AS count_std,
    avg(processed_measurements.count) AS count_mean,
    min(processed_measurements.count) AS count_min,
    max(processed_measurements.count) AS count_max
   FROM processed_measurements
UNION ALL
 SELECT 'processed_synthetics'::text AS source_table,
    stddev(processed_synthetics.energy) AS energy_std,
    avg(processed_synthetics.energy) AS energy_mean,
    min(processed_synthetics.energy) AS energy_min,
    max(processed_synthetics.energy) AS energy_max,
    stddev(processed_synthetics.count) AS count_std,
    avg(processed_synthetics.count) AS count_mean,
    min(processed_synthetics.count) AS count_min,
    max(processed_synthetics.count) AS count_max
   FROM processed_synthetics;
```
---

**TABLENAME: view_processed_synthetics_latent_space**

Joining latent-space to the corresponding synthetic-spectra

```
SELECT grouped.datetime AS psl_datetime,
    grouped.identified_isotopes,
    grouped.row_count,
    psl.datetime,
    psl.datetime_from_measurement,
    psl."0",
    psl."1",
    psl."2",
    psl."3",
    psl."4",
    psl."5",
    psl."6",
    psl."7",
    psl."8",
    psl."9",
    psl."10",
    psl."11",
    psl."12",
    psl."13",
    psl."14",
    psl."15",
    psl."16",
    psl."17",
    psl."18",
    psl."19",
    psl."20",
    psl."21",
    psl."22",
    psl."23"
   FROM (( SELECT ps.datetime,
            count(DISTINCT ps.identified_isotope) AS identified_isotopes,
            string_agg(DISTINCT ps.identified_isotope, ','::text ORDER BY ps.identified_isotope) AS row_count
           FROM processed_synthetics ps
          GROUP BY ps.datetime) grouped
     JOIN processed_synthetics_latent_space psl ON ((grouped.datetime = psl.datetime)))
  ORDER BY grouped.datetime;
```
---


Some indices for faster query:

```
-- For datetime-based aggregations
CREATE INDEX IF NOT EXISTS idx_processed_measurements_datetime ON processed_measurements(datetime);
CREATE INDEX IF NOT EXISTS idx_processed_synthetics_datetime ON processed_synthetics(datetime);
CREATE INDEX IF NOT EXISTS idx_measurements_datetime ON measurements(datetime);

-- For filtering
CREATE INDEX IF NOT EXISTS idx_processed_measurements_peak ON processed_measurements(peak);
CREATE INDEX IF NOT EXISTS idx_processed_synthetics_peak ON processed_synthetics(peak);
```

**The best way is to run docker-compose -> trying to run pipeline.py step by step and 
to check if there are some issues. Configurations (config-files) and views might be the biggest issue
while tring to run the pipeline.**

---