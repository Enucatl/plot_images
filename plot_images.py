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
         language="it"):
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
    min_x = 300
    max_x = 550
    min_y = 60
    max_y = 130
    abs1 = abs1_plot.imshow(absorption_image,
                            cmap=plt.cm.Greys,
                            aspect='auto')
    abs1_plot.add_patch(mpl.patches.Rectangle(
        (min_x, min_y),
        max_x - min_x,
        max_y - min_y,
        fill=False,
        edgecolor="r"))
    diff_image = np.diff(absorption_image[min_y:max_y, min_x:max_x])
    abs2 = abs2_plot.imshow(
        diff_image,
        cmap=plt.cm.Greys, aspect='auto')
    abs1_plot.set_ylabel(absorption_image_title + "\n(differential)",
                         size="large")
    abs1_plot.set_frame_on(False)
    abs1_plot.axes.yaxis.set_ticks([])
    abs1_plot.axes.xaxis.set_ticks([])
    abs2_plot.axes.yaxis.set_ticks([])
    abs2_plot.axes.xaxis.set_ticks([])
    for pos in ["top", "bottom", "left", "right"]:
        abs2_plot.spines[pos].set_color("r")
    abs2_plot.axes.xaxis.set_ticks([])
    diff_limits = stats.mstats.mquantiles(diff_image,
                                          prob=[0.02, 0.98])
    abs2.set_clim(*diff_limits)
    plt.colorbar(abs2,
                 ax=abs2_plot,
                 format="% .2f",
                 ticks=np.arange(-0.04, 0.04, 0.01).tolist())
    limits = stats.mstats.mquantiles(absorption_image,
                                     prob=[0.02, 0.98])
    abs1.set_clim(*limits)
    phase1 = phase1_plot.imshow(differential_phase_image)
    phase1_plot.add_patch(mpl.patches.Rectangle(
        (min_x, min_y),
        max_x - min_x,
        max_y - min_y,
        fill=False,
        edgecolor="r"))
    phase2 = phase2_plot.imshow(
        differential_phase_image[min_y:max_y, min_x:max_x])
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
    df1 = df1_plot.imshow(dark_field_image)
    df1_plot.add_patch(mpl.patches.Rectangle(
        (min_x, min_y),
        max_x - min_x,
        max_y - min_y,
        fill=False,
        edgecolor="r"))
    df2 = df2_plot.imshow(
        dark_field_image[min_y:max_y, min_x:max_x])
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

if __name__ == '__main__':
    import argparse
    commandline_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    commandline_parser.add_argument("--language",
                                    default="it",
                                    choices=["it", "en"],
                                    help="language for the text")
    commandline_parser.add_argument("file",
                                    nargs=1,
                                    help="input file name")
    commandline_parser.add_argument("height",
                                    nargs=1,
                                    type=float,
                                    help="height of the plot")
    args = commandline_parser.parse_args()
    input_file_name = args.file[0]
    height = args.height[0]

    if not os.path.exists(input_file_name):
        raise(OSError("{0} not found".format(input_file_name)))

    input_file = h5py.File(input_file_name, "r")
    absorption_image_name = "postprocessing/absorption"
    differential_phase_image_name = "postprocessing/differential_phase"
    visibility_reduction_image_name = "postprocessing/visibility_reduction"

    absorption_image = input_file[absorption_image_name]
    differential_phase_image = input_file[differential_phase_image_name]
    visibility_reduction_image = input_file[visibility_reduction_image_name]

    draw(input_file_name, height, absorption_image,
         differential_phase_image, visibility_reduction_image,
         args.language)
