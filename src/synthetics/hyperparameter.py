import torch

class Hyperparameter:
    """
    Hyperparameter for Variational Autoencoder.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 8160
    HIDDEN_DIM = 1024
    SECOND_HIDDEN_DIM = 128
    LATENT_DIM = 24
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100