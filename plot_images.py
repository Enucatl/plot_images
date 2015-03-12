#!/usr/bin/env python
# encoding: utf-8

"""Nice plot of the three DPC images"""

import os
import h5py
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt


pgf_with_rc_fonts = {
    "image.origin": "lower",
    "font.family": "serif",
    "pgf.rcfonts": False,
    "ytick.major.pad": 5,
    "xtick.major.pad": 5,
    "font.size": 11,
    "linewidth": 1,
    "legend.fontsize": "medium",
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "ytick.labelsize": "medium",
    "xtick.labelsize": "medium",
    "axes.linewidth": 1,
}

mpl.rcParams.update(pgf_with_rc_fonts)


def draw(input_file_name, height,
         absorption_image,
         differential_phase_image,
         dark_field_image,
         language="it", batch=True, magnified_square=[300, 550, 60, 130]):
    """Display the calculated images with matplotlib."""
    if language == "it":
        absorption_image_title = "assorbimento"
        differential_phase_image_title = "fase differenziale"
        dark_field_image_title = "riduzione di visibilit\\`a"
    else:
        absorption_image_title = "absorption"
        differential_phase_image_title = "differential phase"
        dark_field_image_title = "dark field"
    _, ((abs1_plot, abs2_plot),
        (phase1_plot, phase2_plot),
        (df1_plot, df2_plot)) = plt.subplots(
            3, 2, figsize=(6, height), dpi=300)
    plt.subplots_adjust(
        wspace=0.02,
        hspace=0.02)
    min_x, max_x, min_y, max_y = magnified_square
    abs1 = abs1_plot.imshow(absorption_image,
                            cmap=plt.cm.Greys,
                            aspect='auto')
    abs1_plot.add_patch(mpl.patches.Rectangle(
        (min_x, min_y),
        max_x - min_x,
        max_y - min_y,
        fill=False,
        edgecolor="r"))
    abs2 = abs2_plot.imshow(
        absorption_image[min_y:max_y, min_x:max_x],
        cmap=plt.cm.Greys, aspect='auto')
    abs1_plot.set_ylabel(absorption_image_title,
                         size="large")
    abs1_plot.set_frame_on(False)
    abs1_plot.axes.yaxis.set_ticks([])
    abs1_plot.axes.xaxis.set_ticks([])
    abs2_plot.axes.yaxis.set_ticks([])
    abs2_plot.axes.xaxis.set_ticks([])
    for pos in ["top", "bottom", "left", "right"]:
        abs2_plot.spines[pos].set_color("r")
    abs2_plot.axes.xaxis.set_ticks([])
    limits = stats.mstats.mquantiles(
        absorption_image,
        prob=[0.02, 0.98])
    limits = stats.mstats.mquantiles(absorption_image,
                                     prob=[0.02, 0.98])
    abs1.set_clim(*limits)
    abs2.set_clim(*limits)
    print(limits)
    plt.colorbar(abs2,
                 ax=abs2_plot,
                 format="% .2f",
                 ticks=np.arange(0, 1, 0.1).tolist())
    phase1 = phase1_plot.imshow(differential_phase_image, aspect='auto')
    phase1_plot.add_patch(mpl.patches.Rectangle(
        (min_x, min_y),
        max_x - min_x,
        max_y - min_y,
        fill=False,
        edgecolor="r"))
    phase2 = phase2_plot.imshow(
        differential_phase_image[min_y:max_y, min_x:max_x], aspect='auto')
    limits = stats.mstats.mquantiles(differential_phase_image,
                                     prob=[0.02, 0.98])
    #limits = (-3, 3)
    phase1_plot.set_ylabel(differential_phase_image_title,
                           size="large")
    phase1_plot.set_frame_on(False)
    phase1_plot.axes.yaxis.set_ticks([])
    phase1_plot.axes.xaxis.set_ticks([])
    phase2_plot.axes.yaxis.set_ticks([])
    phase2_plot.axes.xaxis.set_ticks([])
    for pos in ["top", "bottom", "left", "right"]:
        phase2_plot.spines[pos].set_color("r")
    phase2_plot.axes.xaxis.set_ticks([])
    plt.colorbar(phase2,
                 ax=phase2_plot,
                 format="% .1f",
                 ticks=np.arange(-0.4, 0.4, 0.1).tolist())
    phase1.set_clim(*limits)
    phase2.set_clim(*limits)
    df1 = df1_plot.imshow(dark_field_image, aspect='auto')
    df1_plot.add_patch(mpl.patches.Rectangle(
        (min_x, min_y),
        max_x - min_x,
        max_y - min_y,
        fill=False,
        edgecolor="r"))
    df2 = df2_plot.imshow(
        dark_field_image[min_y:max_y, min_x:max_x], aspect='auto')
    df1_plot.set_ylabel(dark_field_image_title,
                        size="large")
    df1_plot.set_frame_on(False)
    df1_plot.axes.yaxis.set_ticks([])
    df1_plot.axes.xaxis.set_ticks([])
    df2_plot.axes.yaxis.set_ticks([])
    df2_plot.axes.xaxis.set_ticks([])
    for pos in ["top", "bottom", "left", "right"]:
        df2_plot.spines[pos].set_color("r")
    df2_plot.axes.xaxis.set_ticks([])
    plt.colorbar(df2,
                 ax=df2_plot,
                 format="% .1f",
                 ticks=np.arange(0, 1, 0.1).tolist())
    limits = stats.mstats.mquantiles(dark_field_image,
                                     prob=[0.02, 0.98])
    df1.set_clim(*limits)
    df2.set_clim(*limits)
    plt.savefig('images_{0}.eps'.format(
        os.path.splitext(os.path.basename(input_file_name))[0]),
        bbox_inches="tight", dpi=300)
    if not batch:
        plt.ion()
        plt.show()
        input("Press ENTER to quit...")


if __name__ == '__main__':
    import argparse
    commandline_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    commandline_parser.add_argument(
        "--language",
        default="it",
        choices=["it", "en"],
        help="language for the text")
    commandline_parser.add_argument(
        "file",
        nargs=1,
        help="input file name")
    commandline_parser.add_argument(
        "height",
        nargs="?",
        default=6,
        type=float,
        help="height of the plot")
    commandline_parser.add_argument(
        "--small_crop",
        nargs="*",
        default=[300, 550, 60, 130],
        type=int,
        help="area inside the red rectangle (min_x, max_x, min_y, max_y)")
    commandline_parser.add_argument(
        "--big_crop",
        nargs="*",
        default=[0, -1, 0, -1],
        type=int,
        help="area inside the plot (min_x, max_x, min_y, max_y)")
    commandline_parser.add_argument(
        "--batch",
        action="store_true",
        help="run in batch mode, no picture is shown")
    args = commandline_parser.parse_args()
    input_file_name = args.file[0]
    height = args.height

    if not os.path.exists(input_file_name):
        raise(OSError("{0} not found".format(input_file_name)))

    input_file = h5py.File(input_file_name, "r")
    dataset_name = "postprocessing/dpc_reconstruction"
    min_x, max_x, min_y, max_y = args.big_crop
    dataset = input_file[dataset_name][0, min_y:max_y, min_x:max_x, ...]
    print(dataset.shape)
    absorption_image = dataset[..., 0]
    differential_phase_image = dataset[..., 1]
    visibility_reduction_image = dataset[..., 2]

    draw(input_file_name, height, absorption_image,
         differential_phase_image, visibility_reduction_image,
         args.language, args.batch, args.small_crop)
