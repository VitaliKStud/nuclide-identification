# Nuclide-Identification

This project provides:

- statisical method of prominence to label nuclides within gamma-spectra
- Variational Autoencoders (VAE) for generating synthetic gamma-spectra
- Random Forest and Convolutional Neural Network (CNN) (1-D) to classify nuclides based on labels

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

2. Setup environment variables:

**Environment Variables**

| Name                       | Value                                                                                                    |
|----------------------------|----------------------------------------------------------------------------------------------------------|
| CONFIG_FILE                | the name of the used config-file for the envirnoment (EXAMPLE: config_local.yaml or config_service.yaml)) |
| config/config_local.yaml   | Check this file (for running on local machine)                                                           |
| config/config_service.yaml | Check this file (for dashboard)                                                                          |




2. Run docker-compose (MlFlow, PostgreSQL, S3-Bucket, Adminer (Frontend for PostgreSQL) and Dashboard)

```
docker-compose -up -d
```



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

TODO: 

- Fill Documentation Environment and Setup section.
- Init folder-structure via poetry.

---

## Environment and Setup



```
pip install poetry
```

## Folder Structure


---

## NOTES

Am-241: 59,5 keV
Co-60: 1173,2 und 1332,5 keV
Cs-137: 661,7 keV
Eu-152: 121,8 keV / 344,3 keV / 778,9 keV / 964 keV / 1085,8 keV / 1112,1 keV  / 1408 keV
Eu-154: 123,1 keV / 1274,4 keV / 723,32 keV / 1004,8 keV / 873,18 keV / 996,29 keV

8160 Channels 

# 2017-07-03_15-51-58_Summenspektrum.txt


- 1246 different Measurements


https://www-nds.iaea.org/

- Cool page with API, need to be explored
- https://de.wikipedia.org/wiki/Internationale_Atomenergie-Organisation
- https://www-nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides=am241&rad_types=g
fields=decay_rads&nuclides=135xe&rad_types=e
- gammas

LOCALLY: CONFIG_FILE=config_local.yaml
DOCKER: CONFIG_FILE=config_service.yaml