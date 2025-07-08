import numpy as np


def map_eta_phi() :
    ieta = np.array([
        55, 55, 55, 55, 55, 54, 54, 54, 54, 54, 53, 53, 53, 53, 53, 52, 52, 52, 52, 52, 51, 51, 51, 51, 51,
        50, 50, 50, 50, 50, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 47, 47, 47, 47, 47, 46, 46, 46, 46, 46,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 47, 47, 47, 47, 47, 46, 46, 46, 46, 46,
        50, 50, 50, 50, 50, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 47, 47, 47, 47, 47, 46, 46, 46, 46, 46,
        55, 55, 55, 55, 55, 54, 54, 54, 54, 54, 53, 53, 53, 53, 53, 52, 52, 52, 52, 52, 51, 51, 51, 51, 51,
        55, 55, 55, 55, 55, 54, 54, 54, 54, 54, 53, 53, 53, 53, 53, 52, 52, 52, 52, 52, 51, 51, 51, 51, 51,
        56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60,
        56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60,
        90, 90, 90, 90, 90, 89, 89, 89, 89, 89, 88, 88, 88, 88, 88, 87, 87, 87, 87, 87, 86, 86, 86, 86, 86
    ])

    iphi = np.array([
        10,  9,  8,  7,  6,  6,  7,  8,  9, 10, 10,  9,  8,  7,  6,  6,  7,  8,  9, 10, 10,  9,  8,  7,  6,
        15, 14, 13, 12, 11, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  5,  4,  3,  2,  1,
        10,  9,  8,  7,  6,  6,  7,  8,  9, 10, 10,  9,  8,  7,  6,  6,  7,  8,  9, 10, 10,  9,  8,  7,  6,
        5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  5,  4,  3,  2,  1,
        15, 14, 13, 12, 11, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11,
        1,  2,  3,  4,  5,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,
        6,  7,  8,  9, 10, 10,  9,  8,  7,  6,  6,  7,  8,  9, 10, 10,  9,  8,  7,  6,  6,  7,  8,  9, 10,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 11, 12, 13, 14, 15,
        5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  5,  4,  3,  2,  1
        ])
    return ieta, iphi


def mask_central_channel(eta_min, phi_min):
    ieta, iphi = map_eta_phi()
    mask_central = (ieta == (eta_min+2)) & (iphi == (phi_min+2))
    return mask_central

def mask_5x5_matrix(eta_min, phi_min, eta_max, phi_max):
    ieta, iphi = map_eta_phi()
    mask_eta_maj = ieta >= eta_min
    mask_eta_min = ieta <= eta_max
    mask_phi_maj = iphi >= phi_min
    mask_phi_min = iphi <= phi_max
    mask_5x5 = mask_eta_maj & mask_eta_min & mask_phi_maj & mask_phi_min
    return mask_5x5

def read_data(waves):
    bit13_mask = 1 << 13 #validity bit
    bit12_mask = 1 << 12 #gain bit
    amp_mask   = 0x0FFF #amplitude mask
    is_valid = (waves & bit13_mask) != 0
    gain_is_1 = (waves & bit12_mask) != 0
    amplitudes = waves & amp_mask
    amplitudes_corr = amplitudes.copy()
    amplitudes_corr[gain_is_1] *= 10
    amplitudes_corr[~is_valid] = 0
    return amplitudes_corr, is_valid, gain_is_1

def mask_amplitudes(amplitudes, central_idx, threshold=150):
    #nevents, nchannels, nsamples = amplitudes.shape
    amplitudes_central = amplitudes[:, central_idx, :]
    mask_sig_amp = (amplitudes_central.max(axis=1) > 150).squeeze()
    return mask_sig_amp

def mask_rms_baseline(amplitudes, central_idx, threshold=20, pre=5, post=10):
    nevents, nchannels, nsamples = amplitudes.shape
    mask_rms_bline = np.ones(nevents, dtype=bool)
    baselines = np.zeros((nevents, nchannels), dtype=amplitudes.dtype)
    #print(f"------- DEBUG -------\n{baselines.shape}")
    window_size = pre + post
    signal_window = []
    for ev in range(nevents):
        #baseline
        waveform_central = amplitudes[ev, central_idx, :]
        max_idx = np.argmax(waveform_central)
        if max_idx >= 15:
            baseline_slice = slice(0, 10)
        elif 8 <= max_idx < 15:
            baseline_slice = slice(max(0, max_idx - 6), max_idx - 1)
        else:
            baseline_slice = slice(-10, None)
        baselines[ev] = np.mean(amplitudes[ev, :, baseline_slice], axis=1)
        baseline_central = waveform_central[baseline_slice]
        rms_baseline_central = np.std(baseline_central)
        if rms_baseline_central > threshold:
            mask_rms_bline[ev] = False
        #signal window
        start = max(max_idx - pre, 0)
        end = min(max_idx + post, nsamples)
        if end - start < window_size:
            if start == 0:
                end = min(window_size, nsamples)
            elif end == nsamples:
                start = max(nsamples - window_size, 0)
        signal_window.append(amplitudes[ev, :, start:end])
    signal_window = np.stack(signal_window)
    return mask_rms_bline, baselines, signal_window
