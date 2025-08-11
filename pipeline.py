from src.pipeline.pipeline import Pipeline

Pipeline().run(
    download_nuclides=False, # Downloading nuclide meta-data
    prepare_measurements=False, # Converting gamma-spectra from .csv to PostgreSQL
    find_measurements_peaks=False, # Labeling (based on prominence) and interpolating energy (raw measurement)
    measurement_peaks_prefix="", # Any key suffix for processed-measurements
    split_dataset=False, # Splitting raw dataset based on nuclides
    vae_training=False, # Training a VAE
    generate_synthetics=False, # Generating synthetics
    truncate_synhtetics=False, # Cleaning synthetic data-table (use prefix, if u need to keep the old data)
    use_measurements_for_latent=False, # Using processed-measurements to generate latent-space out of it
    synthetic_prefix="v53", # Will be added to the key of synthetics
    relable_synthetics=False, # Second labeling-step for synthetic data
    relable_measurements=False, # Second labeling-step for measurements
    resplit_data=False, # Splitting CNN Training data into validation and training dataset
    cnn_training=False, # Training the CNN for classification
    use_relabled_data=False, # Using second labeling-data
    rf_training=False # Training the RF for classification
)
