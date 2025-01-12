from audiomentations import Compose, AddGaussianNoise, TimeStretch, Shift, AddGaussianSNR, \
                            PitchShift, Gain, PolarityInversion, AddShortNoises, \
                            LowPassFilter, HighPassFilter, BandPassFilter, AirAbsorption, \
                            Padding, Resample, Reverse, RoomSimulator, TanhDistortion, \
                            TimeMask, SpecFrequencyMask


# augment = Compose([
#     AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.2, p=1.0),
#     AddGaussianSNR(min_snr_in_db=5.0, max_snr_in_db=40.0, p=1.0),
#     PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1.0),
#     TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
#     Gain(min_gain_in_db=-10.0, max_gain_in_db=15.0, p=1.0),
#     PolarityInversion(p=1),
#     LowPassFilter(min_cutoff_freq=20.0, max_cutoff_freq=25000.0, p=1)
# ])

# √
augment_GaussianNoise = Compose([
    AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.02, p=1.0)
])

augment_GaussianSNR = Compose([
    AddGaussianSNR(min_snr_in_db=5.0, max_snr_in_db=40.0, p=1.0)
])

augment_TimeStretch = Compose([
    TimeStretch(min_rate=0.5, max_rate=1.5, leave_length_unchanged=True, p=1.0)
])

augment_PitchShift = Compose([
    PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1.0)
])

augment_Gain = Compose([
    Gain(min_gain_in_db=-10.0, max_gain_in_db=15.0, p=1.0)
])

augmet_LowPassFilter = Compose([
    LowPassFilter(min_cutoff_freq=15000.0, max_cutoff_freq=25000.0, zero_phase=True,  p=1)
])

# Maybe this one should be combined with an augmentation to keep the length of the signal the same
augment_ResampleAndPadding_500k = Compose([
    Resample(min_sample_rate=500000, max_sample_rate=500000, p=1),
    Padding(p=0)
])

augment_ResampleAndPadding_200k = Compose([
    Resample(min_sample_rate=200000, max_sample_rate=200000, p=1),
    Padding(p=1)
])

augment_ResampleAndPadding_100k = Compose([
    Resample(min_sample_rate=100000, max_sample_rate=100000, p=1),
    Padding(p=0)
])

augment_ResampleAndPadding_48k = Compose([
    Resample(min_sample_rate=48000, max_sample_rate=48000, p=1)
])

augment_Padding = Compose([
    Padding(p=1)
])

augment_PolarityInversion = Compose([
    PolarityInversion(p=1)
])

augment_TimeMask = Compose([
    TimeMask(
    min_band_part=0.05,
    max_band_part=0.1,
    fade=True,
    p=1.0)
])

augment_FrequencyMask = Compose([
    SpecFrequencyMask(
        fill_mode="constant",
        fill_constant=0.0,
        min_mask_fraction=0.01,
        max_mask_fraction=0.05,
        p=1.0)
])

augment_Reverse = Compose([
    Reverse(p=1.0)
])

augment_RoomSimulator = Compose([
    RoomSimulator(p=1.0)
])

augment_TanhDistortion = Compose([
    TanhDistortion(min_distortion=0.005, max_distortion=0.02, p=1.0)
])

# augment_impact_echo_data = Compose([
#     AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.02, p=0.8),
#     AddGaussianSNR(min_snr_in_db=20.0, max_snr_in_db=40.0, p=0.5),
#     # TimeStretch(min_rate=1.5, max_rate=1.5, leave_length_unchanged=True, p=0.6),
#     PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1.0),
#     Gain(min_gain_in_db=-10.0, max_gain_in_db=15.0, p=0.8),
#     # Resample(min_sample_rate=100000, max_sample_rate=200000, p=1),
#     Padding(p=0.01),
#     PolarityInversion(p=0.01),
#     # TimeMask(min_band_part=0.0005, max_band_part=0.001, fade=True, p=0.1),
#     TanhDistortion(min_distortion=0.05, max_distortion=0.2, p=0.3)
# ])

# augment_impact_echo_data = Compose([
#     # Add Gaussian noise with a 50% probability
#     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#     AddGaussianSNR(min_snr_in_db=20.0, max_snr_in_db=40.0, p=0.5),
#     # Time-stretch the sound (slow down/speed up)
#     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
#     # Pitch-shift the sound (up or down)
#     PitchShift(min_semitones=0, max_semitones=4, p=0.5),
#     # Shift the audio forwards or backwards with respect to time
#     # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
#     TimeMask(min_band_part=0.05, max_band_part=0.10, fade=True, p=0.7)
# ])

augment_impact_echo_data= Compose([
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

# all models are trained with this one:
augment_ccny_ie_data____ = Compose([
    # Add Gaussian noise with a 50% probability
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    AddGaussianSNR(min_snr_in_db=20.0, max_snr_in_db=40.0, p=0.5),
    # Time-stretch the sound (slow down/speed up)
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    # Pitch-shift the sound (up or down)
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    # Shift the audio forwards or backwards with respect to time
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.3)
])

augment_ccny_ie_data_old = Compose([
    AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.02, p=0.8),
    AddGaussianSNR(min_snr_in_db=30.0, max_snr_in_db=50.0, p=0.5),
    TimeStretch(min_rate=0.5, max_rate=1.5, leave_length_unchanged=True, p=0.6),
    # PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.0),
    Gain(min_gain_in_db=-10.0, max_gain_in_db=15.0, p=0.8),
    Resample(min_sample_rate=200000, max_sample_rate=200000, p=1),
    Padding(p=0.1),
    # PolarityInversion(p=0.01),
    # TimeMask(min_band_part=0.0005, max_band_part=0.001, fade=True, p=0.1),
    TanhDistortion(min_distortion=0.005, max_distortion=0.02, p=0.3)
])

augment_ccny_ie_data = Compose([
    # Add Gaussian noise with a 50% probability
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
    AddGaussianSNR(min_snr_in_db=20.0, max_snr_in_db=40.0, p=1),
    # Time-stretch the sound (slow down/speed up)
    TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
    # Pitch-shift the sound (up or down)
    PitchShift(min_semitones=-4, max_semitones=4, p=1),
    # Shift the audio forwards or backwards with respect to time
    # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1)
])

augment_gaussian_noise = Compose([
    # Add Gaussian noise with a 50% probability
    AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.8),
    AddGaussianSNR(min_snr_in_db=20.0, max_snr_in_db=40.0, p=0.8),    
    # Shift the audio forwards or backwards with respect to time
    # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.8),
    # Pitch-shift the sound (up or down)
    PitchShift(min_semitones=-3, max_semitones=3, p=0.8),
    TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.8)
])