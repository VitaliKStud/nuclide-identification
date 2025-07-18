from src.pipeline.pipeline import Pipeline


Pipeline().run(
    # find_measurements_peaks=True,
    # measurement_peaks_prefix="",
    # split_dataset=True,
    # vae_training=True,
    # generate_synthetics=True,
    # truncate_synhtetics=True,
    # use_measurements_for_latent=True,
    # synthetic_prefix="v53",
    # relable_synthetics=False,
    # relable_measurements=False,
    # resplit_data=True,
    cnn_training=True,
    use_relabled_data=True
)


# SEHR GUTES BEISPIEL FÜR SEHR GUTES LABELING: 2018-11-06T08:51:03
# SCHELCHTES BEISPIEL FÜR RELATIV SCHLECHTES LABELING: 2017-10-19T10:03:35
