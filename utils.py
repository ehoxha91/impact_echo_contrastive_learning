import torch
import numpy as np
from dataloaders.augmentations import *

home_path=""  # you may need to update this.


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")
    return


def save_hybrid_model(embedding_model, classifier_model, path):
    torch.save(embedding_model.state_dict(), path+".pth")
    torch.save(classifier_model.state_dict(), path+f"_classifier.pth")
    print(f"Models saved as {path}.pth.")
    return


def normalize_data__(signal):
    return (signal - np.min(signal))/(np.max(signal)-np.min(signal))


def load_ds1_test_data_into_torch_tensor(device, X_path='data/X_test_860.npy', y_path='data/y_test.npy'):
    # DATASET 1 - Test
    X_test = np.load(X_path)
    y_test = np.load(y_path)

    X_temp = np.array([])
    for sig in X_test:
        sig = normalize_data__(sig)
        X_temp = np.append(X_temp, sig)
    X_test = np.reshape(X_temp, (252, 860))
    del X_temp

    X_test = torch.tensor(X_test, dtype=torch.double)
    X_test = X_test.view(X_test.size(0), 1, X_test.size(1))
    X_test = X_test.to(device)
    X_test = X_test.type(torch.cuda.FloatTensor)
    return X_test, y_test


def load_ds3_overlay_test_data_into_torch_tensor(device):
    X_overlay = np.load('data/X_overlayed_860.npy')[252:252*2]
    y_overlay = np.load('data/y_overlayed.npy')
    X_temp = np.array([])
    for sig in X_overlay:
        sig = normalize_data__(sig[0:860])
        X_temp = np.append(X_temp, sig)
    X_overlay = np.reshape(X_temp, (-1, 860))
    X_overlay = torch.tensor(X_overlay, dtype=torch.double)
    X_overlay = X_overlay.view(X_overlay.size(0), 1, X_overlay.size(1))
    X_overlay = X_overlay.to(device)
    X_overlay = X_overlay.type(torch.cuda.FloatTensor)
    return X_overlay, y_overlay


def load_ccny_sep2022_data_into_torch_tensor(
    device, 
    X_path='data/X_our_slab_size860.npy'
):
    X_test = np.load(X_path)
    X_temp = np.array([])
    for sig in X_test:
        sig = normalize_data__(sig)
        X_temp = np.append(X_temp, sig)
    X_test = np.reshape(X_temp, (1824, 860))
    del X_temp
    X_may = X_test[0:1178]
    X_june = X_test[1178:1824]

    X_may = torch.tensor(X_may, dtype=torch.double)
    X_may = X_may.view(X_may.size(0), 1, X_may.size(1))
    X_may = X_may.to(device)
    X_may = X_may.type(torch.cuda.FloatTensor)

    X_june = torch.tensor(X_june, dtype=torch.double)
    X_june = X_june.view(X_june.size(0), 1, X_june.size(1))
    X_june = X_june.to(device)
    X_june = X_june.type(torch.cuda.FloatTensor)
    return X_may, X_june


def load_ccny_sep2022_data_into_torch_tensor_augmented(device):
    X_test = np.load('data/X_our_slab_size860.npy')
    X_temp = np.array([])
    for sig in X_test:
        sig = normalize_data__(sig)
        X_temp = np.append(X_temp, sig)
    X_test = np.reshape(X_temp, (1824, 860))
    del X_temp
    X_may = X_test[0:1178]
    X_june = X_test[1178:1824]

    X_may = torch.tensor(X_may, dtype=torch.double)
    X_may = X_may.view(X_may.size(0), 1, X_may.size(1))
    X_may = X_may.to(device)
    X_may = X_may.type(torch.cuda.FloatTensor)

    X_june = torch.tensor(X_june, dtype=torch.double)
    X_june = X_june.view(X_june.size(0), 1, X_june.size(1))
    X_june = X_june.to(device)
    X_june = X_june.type(torch.cuda.FloatTensor)
    return X_may, X_june


def load_ccny_nov2023_data_into_torch_tensor2(device, X_path='data/nov2023_non_resampled.npy'):
    X_nov23 = np.load(X_path)
    X_temp = np.array([])
    for sig in X_nov23:
        sig = normalize_data__(sig)
        X_temp = np.append(X_temp, sig)
    X_test = np.reshape(X_temp, (1496, 860))
    X_nov23 = torch.tensor(X_test, dtype=torch.double)
    X_nov23 = X_nov23.view(X_nov23.size(0), 1, X_nov23.size(1))
    X_nov23 = X_nov23.to(device)
    X_nov23 = X_nov23.type(torch.cuda.FloatTensor)
    return X_nov23
