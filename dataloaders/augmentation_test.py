import numpy as np
import sys
from augmentations import *
import torch

home_path="/home/ehoxha/projects2023"
sys.path.insert(1, f'{home_path}/impact_echo_cl/impact_echo_cl/dataloaders')

# X = np.load('data/data1024/xtrain.npy')
# X = np.load()
# X = X[:, 0:1024]

X_path = ['data/data1519/xtrain.npy','data/sdnet_dataset.npy']
X = np.array([])
items = 0
for path in X_path:
    tmp = np.load(path)
    tmp = tmp[:, 0:1519]
    items += tmp.shape[0]
    X = np.append(X, tmp)

X = np.reshape(X, (items, -1))
print(f"Dataset Shape: {X.shape}")
import matplotlib.pyplot as plt

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch_device}")

# x_augmentations = augmet_LowPassFilter(X[0], 200000) # not useful
# x_augmentations = augment_Reverse(X[0], 200000) # not useful
# x_augmentations = augment_RoomSimulator(X[0], 200000) # not useful
# x_augmentations = augment_ResampleAndPadding(X[0], 200000) # useful, but make the signal the same length.

# x_augmentations = augment_GaussianNoise(X[0], 200000)
# x_augmentations = augment_GaussianSNR(X[0], 200000)
# x_augmentations = augment_TimeStretch(X[0], 200000)
# x_augmentations = augment_PitchShift(X[0], 200000)
# x_augmentations = augment_Gain(X[0], 200000)
# x_augmentations = augment_Padding(X[0], 200000)
# x_augmentations = augment_PolarityInversion(X[0], 200000)
# x_augmentations = augment_TimeMask(X[0], 200000)
# x_augmentations = augment_TanhDistortion(X, 200000)
# x_augmentations = augment_ResampleAndPadding_200k(X[100], 102400)
# x_augmentations = augment_ResampleAndPadding_500k(X[100], 200000)
# x_augmentations = augment_impact_echo_data(X[100], 500000)

def normalize_data__(signal):
    return (signal - np.min(signal))/(np.max(signal)-np.min(signal))

X_test = np.load(f'{home_path}/impact_echo_cl/data/upsampled/X_our_slab_size860.npy')
print(X_test.shape)
X_path = []
X_temp = np.array([])
for sig in X_test:
    sig = normalize_data__(sig)
    X_temp = np.append(X_temp, sig)
X_test = np.reshape(X_temp, (1824, 860))

from audiomentations import Compose, AddGaussianNoise, TimeStretch, Shift, AddGaussianSNR, \
                            PitchShift, Gain, PolarityInversion, AddShortNoises, \
                            LowPassFilter, HighPassFilter, BandPassFilter, AirAbsorption, \
                            Padding, Resample, Reverse, RoomSimulator, TanhDistortion, \
                            TimeMask

augment_GaussianNoise= Compose([
    # Add Gaussian noise with a 50% probability
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
    AddGaussianSNR(min_snr_in_db=20.0, max_snr_in_db=40.0, p=1)
])

augment_TimeStretch= Compose([ 
    TimeStretch(min_rate=0.8, max_rate=1.25, p=1)
])

augment_PitchShift= Compose([ 
    PitchShift(min_semitones=-4, max_semitones=4, p=1)
])

augment_TimeMask= Compose([ 
    TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1)
])


augment_all= Compose([
    # Add Gaussian noise with a 50% probability
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.8),
    AddGaussianSNR(min_snr_in_db=20.0, max_snr_in_db=40.0, p=0.5),
    # Time-stretch the sound (slow down/speed up)
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    # Pitch-shift the sound (up or down)
    PitchShift(min_semitones=0, max_semitones=5, p=0.7),
    # Shift the audio forwards or backwards with respect to time
    # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    TimeMask(min_band_part=0.05, max_band_part=0.1, fade=True, p=0.7)
])

x_augmentations1 =  normalize_data__(augment_GaussianNoise(X_test[20], 500000)[0:860])
x_augmentations2 =  normalize_data__(augment_TimeStretch(X_test[20], 500000)[0:860])
x_augmentations3 =  normalize_data__(augment_PitchShift(X_test[20], 500000)[0:860])
x_augmentations4 =  normalize_data__(augment_TimeMask(X_test[20], 500000)[0:860])
# x_augmentations5 =  normalize_data__(augment_FrequencyMask(X_test[20], 500000)[0:860])
x_augmentations6 =  normalize_data__(augment_impact_echo_data(X_test[20], 500000)[0:860])


fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))
ax[0].plot(X_test[20])
ax[0].set_title('Original IE Signal')
ax[1].plot(x_augmentations1)
ax[1].set_title('Augmented IE Signal: Gaussian Noise')
ax[2].plot(x_augmentations2)
ax[2].set_title('Augmented IE Signal: Time Stretch')
ax[3].plot(x_augmentations3)
ax[3].set_title('Augmented IE Signal: Pitch Shift')
ax[4].plot(x_augmentations4)
ax[4].set_title('Augmented IE Signal: Time Mask')
# ax[5].plot(x_augmentations5)
# ax[5].set_title('Augmented IE Signal: Frequency Mask')
ax[5].plot(x_augmentations6)
ax[5].set_title('Augmented IE Signal: Combined')
plt.tight_layout()

plt.show()
plt.savefig('augmentaiton_for_cl.png')
plt.close()

# for i in range(20,24):
#     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
#     ax[0].plot(X_test[i])
#     ax[0].set_title('Original IE Signal')
#     x_augmentations5 =  normalize_data__(augment_impact_echo_data(X_test[i], 500000)[0:860])
#     ax[1].plot(x_augmentations5)
#     ax[1].set_title('Augmented IE Signal 1')
#     x_augmentations5 =  normalize_data__(augment_impact_echo_data(X_test[i], 500000)[0:860])
#     ax[2].plot(x_augmentations5)
#     ax[2].set_title('Augmented IE Signal 2')

#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f'unsupervised_method_augs/augmentaiton_for_cl_unsupervised_{i}.png')
#     plt.close()