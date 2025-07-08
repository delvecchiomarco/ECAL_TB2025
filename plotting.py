import numpy as np
import matplotlib.pyplot as plt


def plot_central_waveform(waves, central_idx, output_path="central_waveforms.pdf"):
    central_waveform = waves[:, central_idx, :]
    print(f"Central waveform shape: {central_waveform.shape}")
    #plt.figure(figsize=(10, 6))
    for ev in range(central_waveform.shape[0]):
        plt.plot(range(central_waveform.shape[1]), central_waveform[ev, :].squeeze(), alpha=0.1, color='blue')
    plt.xlabel("samples")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Central waveforms saved to {output_path}")

