import uproot
import ROOT
import numpy as np
import argparse
import os, sys
import matplotlib.pyplot as plt
import time

import gpu_routines
from gpu_routines import get_reco_products
import functions
import plotting


def main(arguments):
    #start time
    time_start = time.time()

    #input parameters
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input", type=str, required=True, help="input ROOT file with unpacked tree")
    parser.add_argument("-r", "--run", type=str, required=True, help="run number")
    parser.add_argument("-s", "--spill", type=str, required=True, help="spill number")
    parser.add_argument("-de", "--eta-min", type=int, required=True, help="eta min")
    parser.add_argument("-dE", "--eta-max", type=int, required=True, help="eta max")
    parser.add_argument("-dp", "--phi-min", type=int, required=True, help="phi min")
    parser.add_argument("-dP", "--phi-max", type=int, required=True, help="phi max")
    args = parser.parse_args(arguments)
    #print(f"------- DEBUG -------\nargs = {args}")
    if args.eta_min >= args.eta_max:
        print("ERROR -> eta min > eta max")
        sys.exit(1)
    if args.phi_min >= args.phi_max:
        print("ERROR -> phi min > phi max")
        sys.exit(1)
    input_file = args.input
    run = args.run
    spill = args.spill
    eta_min = args.eta_min
    eta_max = args.eta_max
    phi_min = args.phi_min
    phi_max = args.phi_max

    #open input file
    file = uproot.open(input_file)
    tree = file["h4"]

    #mapping eta and phi with channels
    ieta, iphi = functions.map_eta_phi()
    #mask for central channel and 5x5 matrix
    mask_central = functions.mask_central_channel(eta_min, phi_min)
    central_idx = np.where(mask_central)[0][0]
    #print(f"------- DEBUG -------\nmask_central: {mask_central}")
    mask_5x5 = functions.mask_5x5_matrix(eta_min, phi_min, eta_max, phi_max)
    #print(f"------- DEBUG -------\nmask_matrix: {mask_5x5}")
    

    #charge_tot
    waves = tree["xtal_sample"].array(library="np")
    #print(f"------- DEBUG -------\nwaves.shape: {waves.shape}")
    #print(f"------- DEBUG -------\nwaves[0, 0, :] {waves[0, 0, :]}")
    amplitudes_corr, is_valid, gain_is_1 = functions.read_data(waves)
    #print(f"------- DEBUG -------\namplitudes_corr.shape: {amplitudes_corr.shape}")
    #print(f"------- DEBUG -------\namplitudes_corr[0, 0, :]: {amplitudes_corr[0, 0, :]}")

    
    #plot of all waves for central channel
    plotting.plot_central_waveform(amplitudes_corr, central_idx, output_path="central_waveforms.pdf")
    

    #mask for signal amplitude above threshold
    mask_sig_amp = functions.mask_amplitudes(amplitudes_corr, central_idx, threshold=150)
    waves_amp_masked = amplitudes_corr[mask_sig_amp, :]
    #print(f"------- DEBUG -------\nwaves_amp_masked.shape: {waves_amp_masked.shape}")

    #mask for baseline rms, baseline subtraction and definition of signal window
    mask_rms_bline, baselines, signal_window = functions.mask_rms_baseline(waves_amp_masked, central_idx, threshold=20, pre=5, post=10)
    #print(f"------- DEBUG -------\nmask_rms_bline.shape: {mask_rms_bline.shape}")
    waves_rms_masked = waves_amp_masked[mask_rms_bline, :]
    signal_window = signal_window[mask_rms_bline, :]
    #print(f"------- DEBUG -------\nwaves_rms_masked.shape: {waves_rms_masked.shape}")
    #print(f"------- DEBUG -------\nsignal_window.shape: {signal_window.shape}")
    nevents, nchannels, nsamples = waves_rms_masked.shape
    waves_rms_masked = waves_rms_masked - np.repeat(baselines[mask_rms_bline, :, np.newaxis], nsamples, axis=2)
    signal_window = signal_window - np.repeat(baselines[mask_rms_bline, :, np.newaxis], signal_window.shape[2], axis=2)
    #print(f"------- DEBUG -------\nwaves_rms_masked.shape: {waves_rms_masked.shape}")
    #print(f"------- DEBUG -------\nsignal_window.shape: {signal_window.shape}")

    
    #plot of all waves for central channel after masking
    plotting.plot_central_waveform(waves_rms_masked, central_idx, output_path="central_waveforms_masked.pdf")
    

    #TH1 of the 5x5 matrix charge
    signal_window5x5 = signal_window[:, mask_5x5, :]
    charge_thr = 100
    charge = signal_window5x5.sum(axis=2)
    #print(f"------- DEBUG -------\n{charge.shape}")
    charge[charge < charge_thr] = 0
    charge_sum = charge.sum(axis=1)
    #print(f"------- DEBUG -------\n{charge_sum.shape}")
    h1 = ROOT.TH1F("h1_charge_sum", "", 10000, 0, 20000)
    for ev in range(charge_sum.shape[0]):
        h1.Fill(charge_sum[ev])
    c1 = ROOT.TCanvas("c1", "", 1000, 600)
    #h1.SetStats(0)
    h1.Draw()
    outfile1 = ROOT.TFile("./Plots/my_algo.root", "RECREATE")
    h1.Write()
    outfile1.Close()
    
    time_end = time.time()
    
    print(f"Time elapsed: {time_end - time_start:.4f} s")
    

if __name__ == '__main__':
    main(sys.argv[1:])

