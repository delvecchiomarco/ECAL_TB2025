import uproot
import ROOT
import argparse
import sys
import numpy as np
import time
from array import array

import reco_functions
import plot_functions


def main(arguments):
    # start time
    time_start = time.time()

    # input parameters
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input", type=str, required=True, help="input ROOT file with unpacked tree")
    parser.add_argument("-r", "--run", type=str, required=True, help="run number")
    parser.add_argument("-s", "--spill", type=str, required=True, help="spill number")
    parser.add_argument("-de", "--eta-min", type=int, required=True, help="eta min")
    parser.add_argument("-dE", "--eta-max", type=int, required=True, help="eta max")
    parser.add_argument("-dp", "--phi-min", type=int, required=True, help="phi min")
    parser.add_argument("-dP", "--phi-max", type=int, required=True, help="phi max")
    parser.add_argument("-o", "--reco-output-dir", type=str, required=True, help="directory for reco output")
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
    reco_dir = args.reco_output_dir

    # open input file
    file = uproot.open(input_file)
    tree = file["h4"]

    # mapping eta and phi with channels
    ieta, iphi = reco_functions.map_eta_phi()
    # mask for central channel and 5x5 matrix
    mask_central = reco_functions.mask_central_channel(eta_min, phi_min)
    central_idx = np.where(mask_central)[0][0]
    # print(f"------- DEBUG -------\nmask_central: {mask_central}")
    mask_5x5 = reco_functions.mask_5x5_matrix(eta_min, phi_min, eta_max, phi_max)
    # print(f"------- DEBUG -------\nmask_5x5: {mask_5x5}")
    ieta_5x5 = ieta[mask_5x5]
    iphi_5x5 = iphi[mask_5x5]
    mask_5x5_central = reco_functions.mask_5x5_central(ieta_5x5, iphi_5x5, eta_min, phi_min)
    # print(f"------- DEBUG -------\nmask_5x5_central: {mask_5x5_central}")
    
    # read branch xtal_sample
    waves = tree["xtal_sample"].array(library="np")
    # print(f"------- DEBUG -------\nwaves.shape: {waves.shape}")
    amplitudes_corr, is_valid, gain_is_1 = reco_functions.read_data(waves)
    # print(f"------- DEBUG -------\namplitudes_corr.shape: {amplitudes_corr.shape}")
    # plot_functions.plot_central_waveform(amplitudes_corr, central_idx, output_path=f"./{reco_dir}/central_waveforms.pdf")  # plot of all waves for central channel

    mask_sig_amp = reco_functions.mask_amplitudes(amplitudes_corr, central_idx, threshold=150)  # mask for signal amplitude above threshold
    waves_amp_masked = amplitudes_corr[mask_sig_amp, :]
    # print(f"------- DEBUG -------\nwaves_amp_masked.shape: {waves_amp_masked.shape}")
    mask_rms_bline, baselines, signal_window = reco_functions.mask_rms_baseline(waves_amp_masked, central_idx, threshold=20, pre=5, post=10)  # mask for baseline rms, baseline subtraction and definition of signal window
    waves_rms_masked = waves_amp_masked[mask_rms_bline, :]
    signal_window = signal_window[mask_rms_bline, :]
    # print(f"------- DEBUG -------\nwaves_rms_masked.shape: {waves_rms_masked.shape}")
    # print(f"------- DEBUG -------\nsignal_window.shape: {signal_window.shape}")
    nevents, nchannels, nsamples = waves_rms_masked.shape
    waves_masked = waves_rms_masked - np.repeat(baselines[mask_rms_bline, :, np.newaxis], nsamples, axis=2)  # baseline subtraction
    signal_window = signal_window - np.repeat(baselines[mask_rms_bline, :, np.newaxis], signal_window.shape[2], axis=2)
    # print(f"------- DEBUG -------\nwaves_rms_masked.shape: {waves_masked.shape}")
    # print(f"------- DEBUG -------\nsignal_window.shape: {signal_window.shape}")
    # plot_functions.plot_central_waveform(waves_masked, central_idx, output_path=f"./{reco_dir}/central_waveforms_masked.pdf")  # plot of all waves for central channel after masking

    # charge_sum in 5x5 matrix
    charge_5x5, charge_sum_5x5, charge_central = reco_functions.charge_5x5(signal_window, mask_5x5, mask_5x5_central, charge_thr=100)
    # index of the maximum sample
    sample_max = np.argmax(waves_masked, axis=2)
    # mean of all values
    values_mean = np.mean(waves_masked, axis=2)
    # std of all values
    values_std = np.std(waves_masked, axis=2)
    # maximum sample
    values_max = np.max(waves_masked, axis=2)
    # amplitude_map of the 5x5 matrix
    amplitude_map_5x5 = charge_5x5 / charge_sum_5x5[:, np.newaxis]

    # data_to_plot_nevents.csv creation
    f = ROOT.TFile(f"./{reco_dir}/{run}_{spill}_reco.root", "RECREATE")
    tree_nevents = ROOT.TTree("h4_reco", "")
    charge_sum_5x5_branch = array('f', [0.0])
    charge_central_branch = array('f', [0.0])
    ieta_branch = array('f', [0.0] * nchannels)
    iphi_branch = array('f', [0.0] * nchannels)
    sample_max_branch = array('f', [0.0] * nchannels)
    values_mean_branch = array('f', [0.0] * nchannels)
    values_std_branch = array('f', [0.0] * nchannels)
    values_max_branch = array('f', [0.0] * nchannels)
    ieta_5x5_branch = array('f', [0.0] * 25)
    iphi_5x5_branch = array('f', [0.0] * 25)
    amplitude_map_5x5_branch = array('f', [0.0] * 25)
    tree_nevents.Branch("charge_sum_5x5", charge_sum_5x5_branch, "charge_sum_5x5/F")
    tree_nevents.Branch("charge_central", charge_central_branch, f"charge_central/F")
    tree_nevents.Branch("ieta", ieta_branch, f"ieta[{nchannels}]/F")
    tree_nevents.Branch("iphi", iphi_branch, f"iphi[{nchannels}]/F")
    tree_nevents.Branch("sample_max", sample_max_branch, f"sample_max[{nchannels}]/F")
    tree_nevents.Branch("values_mean", values_mean_branch, f"values_mean[{nchannels}]/F")
    tree_nevents.Branch("values_std", values_std_branch, f"values_std[{nchannels}]/F")
    tree_nevents.Branch("values_max", values_max_branch, f"values_max[{nchannels}]/F")
    tree_nevents.Branch("ieta_5x5", ieta_5x5_branch, f"ieta_5x5[{25}]/F")
    tree_nevents.Branch("iphi_5x5", iphi_5x5_branch, f"iphi_5x5[{25}]/F")
    tree_nevents.Branch("amplitude_map_5x5", amplitude_map_5x5_branch, f"amplitude_map_5x5[{25}]/F")
    for i in range(nevents):
        charge_sum_5x5_branch[0] = charge_sum_5x5[i]
        for ch in range(nchannels):
            ieta_branch[ch] = ieta[ch]
            iphi_branch[ch] = iphi[ch]
            # print(ieta[ch], iphi[ch])
            sample_max_branch[ch] = sample_max[i][ch]
            values_mean_branch[ch] = values_mean[i][ch]
            values_std_branch[ch] = values_std[i][ch]
            values_max_branch[ch] = values_max[i][ch]
            if ch < 25:
                ieta_5x5_branch[ch] = ieta_5x5[ch]
                iphi_5x5_branch[ch] = iphi_5x5[ch]
                amplitude_map_5x5_branch[ch] = amplitude_map_5x5[i][ch]
        tree_nevents.Fill()
    tree_nevents.Write()
    print(f"Tree h4_reco written in {reco_dir}/{run}_{spill}_reco.root")
    f.Close()
    
    time_end = time.time()
    print(f"Time elapsed for reco: {time_end - time_start:.4f} s")


if __name__ == '__main__':
    main(sys.argv[1:])

