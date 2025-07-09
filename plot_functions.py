import numpy as np
import matplotlib.pyplot as plt
import ROOT
import os


def plot_central_waveform(waves, central_idx, output_path="central_waveforms.pdf"):
    central_waveform = waves[:, central_idx, :]
    # print(f"Central waveform shape: {central_waveform.shape}")
    for ev in range(central_waveform.shape[0]):
        plt.plot(range(central_waveform.shape[1]), central_waveform[ev, :].squeeze(), alpha=0.1, color='blue')
    plt.xlabel("samples")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Central waveforms saved to {output_path}")


def plot(row, chain, outputfolder):
    # print(f"------- DEBUG -------\nrow:\n{row}")
    name = row['name']
    # print(f"------- DEBUG -------\nname: {name}")
    f = ROOT.TFile(f"{outputfolder}/{name}.root", "recreate")
    # print(f"------- DEBUG -------\ncreated file: {outputfolder}/{name}.root")
    f.cd()
    c = ROOT.TCanvas(f"{name}_canvas")
    c.cd()
    if str(row.cuts) == " ": cut = "1"
    else: cut = str(row.cuts)
    print(f"------- DEBUG -------\ncut: {cut}")
    if str(row.y).strip()=="0":
        # print(f"------- DEBUG -------\nempty y")
        h = ROOT.TH1F(f"{name}", f"{row.title}", int(row.binsnx), float(row.binsminx), float(row.binsmaxx))
        chain.Draw(f"{row.x}>>{name}", f"{cut}")
        h.SetLineColor(eval(f"ROOT.{row.color}"))
        binw = (float(row.binsmaxx) - float(row.binsminx))/int(row.binsnx)
        h.GetYaxis().SetTitle(f"entries / {float(f'{binw:.1g}'):g} {row.ylabel}")
    else:
        # print(f"------- DEBUG -------\nNOT empty y")
        h = ROOT.TH2F(f"{name}", f"{row.title}", int(row.binsnx), float(row.binsminx), float(row.binsmaxx), int(row.binsny), float(row.binsminy), float(row.binsmaxy))
        chain.Draw(f"{row.y}:{row.x}>>{name}", f"{cut}", "zcol")
        h.GetYaxis().SetTitle(f"{row.ylabel}")
    h.GetXaxis().SetTitle(f"{row.xlabel}")
    c.SaveAs(f"{outputfolder}/{name}.pdf")
    c.SaveAs(f"{outputfolder}/{name}.png")
    c.SaveAs(f"{outputfolder}/{name}.root")
    c.Write()
    h.Write()
    f.Close()
    c.Close()
    del c
    del h


def process(row, outputfolder, plot_df):
    lst = os.popen(f"/bin/bash -c 'ls -1 {row.filename.strip()}'").read().strip().splitlines()
    # print(f"-------- DEBUG -------\nlst: {lst}")
    chain = ROOT.TChain()
    for file in lst:
        chain.Add(f"{file}/{row.treename.strip()}")
    chain.Print()
    os.system(f"mkdir {outputfolder}/{row.label}")
    print(f"-------- DEBUG -------\nCreated directory: {outputfolder}/{row.label}")
    # os.system(f"cp index.php {outputfolder}/{row.label}")
    plot_df.apply(lambda plotrow: plot(plotrow, chain, f"{outputfolder}/{row.label}"), axis=1)

