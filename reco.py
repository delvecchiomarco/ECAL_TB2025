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
    # print(f"------- DEBUG -------\nmask_matrix: {mask_5x5}")
    
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
    charge_5x5 = reco_functions.charge_sum_5x5(signal_window, mask_5x5, charge_thr=100)

    # mean of the index of the maximum sample
    # mean_sample_max = reco_functions.mean_sample_max(waves_masked)
    sample_max = np.argmax(waves_masked, axis=2)

    # data_to_plot_nevents.csv creation
    f = ROOT.TFile(f"./{reco_dir}/{run}_{spill}_reco.root", "RECREATE")
    tree_nevents = ROOT.TTree("h4_reco", "")
    charge_branch = array('f', [0.0])
    sample_max_branch = array('f', [0.0] * nchannels)
    ieta_branch = array('f', [0.0] * nchannels)
    iphi_branch = array('f', [0.0] * nchannels)
    tree_nevents.Branch("charge_5x5", charge_branch, "charge_5x5/F")
    tree_nevents.Branch("sample_max", sample_max_branch, f"sample_max[{nchannels}]/F")
    tree_nevents.Branch("ieta", ieta_branch, f"ieta[{nchannels}]/F")
    tree_nevents.Branch("iphi", iphi_branch, f"iphi[{nchannels}]/F")
    for i in range(nevents):
        charge_branch[0] = charge_5x5[i]
        for ch in range(nchannels):
            sample_max_branch[ch] = sample_max[i][ch]
            ieta_branch[ch] = ieta[ch]
            iphi_branch[ch] = iphi[ch]
            # print(ieta[ch], iphi[ch])
        tree_nevents.Fill()
    tree_nevents.Write()
    print(f"Tree h4_reco written in {reco_dir}/{run}_{spill}_reco.root")
    f.Close()

    # # data_to_plot_nchannels.csv creation
    # f = ROOT.TFile(f"./{reco_dir}/{run}_{spill}_reco_nchannels.root", "RECREATE")
    # tree_nchannels = ROOT.TTree("h4_reco", "")
    # eta_branch = array('i', [0])
    # phi_branch = array('i', [0])
    # mean_sample_max_branch = array('f', [0.0]*nevents)
    # tree_nchannels.Branch("ieta", eta_branch, "ieta/I")
    # tree_nchannels.Branch("iphi", phi_branch, "iphi/I")
    # tree_nchannels.Branch("mean_sample_max", mean_sample_max_branch, f"mean_sample_max/F")
    # for i in range(nchannels):
    #     eta_branch[0] = ieta[i]
    #     phi_branch[0] = iphi[i]
    #     mean_sample_max_branch[0] = mean_sample_max[i]
    #     tree_nchannels.Fill()
    # tree_nchannels.Write()
    # print(f"Tree h4_reco written in {reco_dir}/{run}_{spill}_reco_nchannels.root")
    # f.Close()
    
    time_end = time.time()
    print(f"Time elapsed for reco: {time_end - time_start:.4f} s")


if __name__ == '__main__':
    main(sys.argv[1:])

