#!/usr/bin/env python
# encoding: utf-8

"""Nice plot of the three DPC images"""

import os
import h5py
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import click


pgf_with_rc_fonts = {
    "image.origin": "lower",
    "font.family": "serif",
    "pgf.rcfonts": False,
    "ytick.major.pad": 5,
    "xtick.major.pad": 5,
    "font.size": 11,
    "legend.fontsize": "medium",
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "ytick.labelsize": "medium",
    "xtick.labelsize": "medium",
    "axes.linewidth": 1,
}

mpl.rcParams.update(pgf_with_rc_fonts)


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--height", type=float, default=6)
@click.option("--language", type=click.Choice(["it", "en"]), default="it")
@click.option("--batch", is_flag=True)
@click.option("--big_crop", nargs=4, type=int, default=[0, -1, 0, -1])
@click.option("--format", default="eps")
def main(filename, height, language, batch, big_crop, format):
    input_file = h5py.File(filename, "r")
    dataset_name = "postprocessing/dpc_reconstruction"
    min_x, max_x, min_y, max_y = big_crop
    dataset = input_file[dataset_name]
    print("original shape", dataset.shape)
    dataset = dataset[0, min_y:max_y, min_x:max_x, ...]
    print("cropped", dataset.shape)
    absorption_image = dataset[..., 0]
    differential_phase_image = dataset[..., 1]
    visibility_reduction_image = dataset[..., 2]

    draw(filename, height, absorption_image,
         differential_phase_image, visibility_reduction_image,
         language, batch, format)


def draw(input_file_name, height,
         absorption_image,
         differential_phase_image,
         dark_field_image,
         language, batch, format):
    """Display the calculated images with matplotlib."""
    if language == "it":
        absorption_image_title = "assorbimento"
        differential_phase_image_title = "fase differenziale"
        dark_field_image_title = "riduzione di visibilit\\`a"
    else:
        absorption_image_title = "absorption"
        differential_phase_image_title = "differential phase"
        dark_field_image_title = "dark field"
    _, (abs1_plot,
        phase1_plot,
        df1_plot) = plt.subplots(
            3, 1, figsize=(6, height), dpi=300)
    plt.subplots_adjust(
        wspace=0.02,
        hspace=0.02)
    abs1 = abs1_plot.imshow(absorption_image,
                            cmap=plt.cm.Greys,
                            aspect='auto')
    abs1_plot.set_ylabel(absorption_image_title,
                         size="large")
    abs1_plot.set_frame_on(False)
    abs1_plot.axes.yaxis.set_ticks([])
    abs1_plot.axes.xaxis.set_ticks([])
    limits = stats.mstats.mquantiles(
        absorption_image,
        prob=[0.02, 0.98])
    limits = stats.mstats.mquantiles(absorption_image,
                                     prob=[0.02, 0.98])
    abs1.set_clim(*limits)
    print(limits)
    plt.colorbar(abs1,
                 ax=abs1_plot,
                 format="% .2f",
                 ticks=np.arange(0, 1, 0.1).tolist())
    phase1 = phase1_plot.imshow(differential_phase_image, aspect='auto')
    limits = stats.mstats.mquantiles(differential_phase_image,
                                     prob=[0.02, 0.98])
    phase1_plot.set_ylabel(differential_phase_image_title,
                           size="large")
    phase1_plot.set_frame_on(False)
    phase1_plot.axes.yaxis.set_ticks([])
    phase1_plot.axes.xaxis.set_ticks([])
    plt.colorbar(phase1,
                 ax=phase1_plot,
                 format="% .1f",
                 ticks=np.arange(-0.4, 0.4, 0.1).tolist())
    phase1.set_clim(*limits)
    df1 = df1_plot.imshow(dark_field_image, aspect='auto')
    df1_plot.set_ylabel(dark_field_image_title,
                        size="large")
    df1_plot.set_frame_on(False)
    df1_plot.axes.yaxis.set_ticks([])
    df1_plot.axes.xaxis.set_ticks([])
    plt.colorbar(df1,
                 ax=df1_plot,
                 format="% .1f",
                 ticks=np.arange(0, 1, 0.1).tolist())
    limits = stats.mstats.mquantiles(dark_field_image,
                                     prob=[0.02, 0.98])
    df1.set_clim(*limits)
    plt.savefig('images_{0}.{1}'.format(
        os.path.splitext(os.path.basename(input_file_name))[0],
        format),
        bbox_inches="tight", dpi=300)
    if not batch:
        plt.ion()
        plt.show()
        input("Press ENTER to quit...")


if __name__ == '__main__':
    main()
