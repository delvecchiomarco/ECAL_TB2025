This code is used for the reconstruction of unpacked data derived from the EBeTe unpacker used in ECAL Test Beam analysis.

Overview of the code:
  - **reco.py & reco_functions.py**\
    python script for the reconstruction, where all the useful variables for DQM plots are saved in a tree

  - **plotter.py & plot_functions.py**\
    python script for plotting, very general with one single function for the different plots

  - **data_to_plot.csv**\
    csv file with path to the reco tree

  - **plot_list.csv**\
    csv file with the list of the plots and the settings for each one
