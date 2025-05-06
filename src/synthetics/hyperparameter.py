import torch


class Hyperparameter:
    """
    Hyperparameter for Variational Autoencoder.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 8160
    HIDDEN_DIM = 512  # 512 GOOD
    SECOND_HIDDEN_DIM = 128  # 128 GOOD
    LATENT_DIM = 24  # 24 GOOD
    BATCH_SIZE = 8160
    LEARNING_RATE = 1e-3  # 1e-3 GOOD
    NUM_EPOCHS = 500
