import ROOT
import os, sys
import argparse
import time
import pandas as pd

import plot_functions


def main(arguments):
    # start time
    time_start = time.time()

    # input parameters
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data", type=str, required=True, help="csv file with data to plot")
    parser.add_argument("-p", "--plot-list", type=str, required=True, help="csv file with plot list")
    parser.add_argument("-o", "--output-folder", type=str, required=True, help="output folder for plots")
    args = parser.parse_args(arguments)
    v = vars(args)
    vars().update(v)
    dataconf_df = pd.read_csv(v['data'], sep=';')
    plotconf_df = pd.read_csv(v['plot_list'], sep=";")
    outputfolder = v['output_folder']
    # os.system(f"cp index.php {outputfolder}")

    # macro for plot style
    # macro = ["root_logon.C"]
    # for m in macro: ROOT.gROOT.LoadMacro(m)

    dataconf_df.apply(lambda row: plot_functions.process(row, outputfolder, plotconf_df), axis=1)

    time_end = time.time()
    print(f"Time elapsed for plotting: {time_end - time_start:.4f} s")


if __name__ == '__main__':
    main(sys.argv[1:])
