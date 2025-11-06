### Script containing the routines to make the validation figures for the Poisson probability calculation between a GC sample and a lambda map
### Author: Marta Reina-Campos
### Date: Dec 3rd 2024

# Import modules
import numpy, os
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy import units as u
from astropy.visualization.wcsaxes import add_scalebar

mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.size"] = 18.0
mpl.rcParams["legend.fontsize"] = 16.0


### Lambda map with GCs overlaid
def figure_lensing_model_with_gcs(galaxy_cluster, gcs, lambda_map, out_path):
    # format the figure
    left = 0.1
    right = 0.87
    top = 0.98
    bottom = 0.1
    hspace = 0.0
    wspace = 0.0

    # normalize all panels equally
    cmap = plt.get_cmap("inferno")
    min_value = lambda_map.img[lambda_map.img.value > 0].value.min()
    norm = mpl.colors.LogNorm(vmin=min_value, vmax=lambda_map.img.value.max())

    # create the Figure object with the correct WCS coordinates
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(14, 6.5),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": lambda_map.wcs},
    )
    axs = numpy.atleast_1d(axs)
    axs = axs.ravel()

    axs[0].imshow(
        lambda_map.img.value.T, norm=norm, cmap=cmap, origin="lower", zorder=0
    )
    axs[1].imshow(
        lambda_map.img.value.T, norm=norm, cmap=cmap, origin="lower", zorder=0
    )
    axs[0].annotate(
        lambda_map.name,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=16,
        color="red",
    )

    # GCs from Harris & Reina-Campos 2024
    axs[1].scatter(
        gcs.ra.to("deg"),
        gcs.dec.to("deg"),
        c="white",
        s=5,
        marker=".",
        edgecolor="k",
        linewidth=0.1,
        zorder=10,
        alpha=0.9,
        transform=axs[1].get_transform("fk5"),
    )
    axs[1].annotate(
        gcs.label,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=16,
        color="red",
    )

    # add the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # add an inset axes for the colorbar
    cax = axs[-1].inset_axes([1.001, 0.0, 0.02, 1.0])  # [x0, y0, width, height]
    # create the colorbar object
    cbar = fig.colorbar(sm, cax=cax, ax=axs[0])
    cbar.minorticks_on()  # add minorticks
    cbar.set_label(
        "Unnormalized $\lambda$ [{:s}]".format(lambda_map.img.unit.to_string())
    )  # add label

    # Compute the angle corresponding to 10 pc at the distance of the galactic center
    scalebar_length = 100 * u.kpc
    scalebar_angle = (scalebar_length / galaxy_cluster.distance.to("kpc")).to(
        u.deg, equivalencies=u.dimensionless_angles()
    )
    # Add a scale bar
    add_scalebar(axs[-1], scalebar_angle, label="100 kpc", color="white")

    # format all axes
    for j, ax in enumerate(axs):
        ra = ax.coords[0]
        dec = ax.coords[1]

        if j == 0:
            dec.set_axislabel("Declination (J2000)")
        else:
            dec.set_ticks_visible(False)
            dec.set_ticklabel_visible(False)
            dec.set_axislabel("")
        ra.set_axislabel("Right Ascension (J2000)")
        # set the formatting of the axes
        ra.set_major_formatter("dd:mm:ss")
        dec.set_major_formatter("dd:mm:ss")

        # display minor ticks
        ra.display_minor_ticks(True)
        dec.display_minor_ticks(True)
        ra.set_minor_frequency(10)
        dec.set_minor_frequency(10)

    # format the entire figure
    fig.subplots_adjust(
        left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
    )

    fname = "xy_lambda_{:s}.png".format(lambda_map.name)
    fig.savefig(os.path.join(out_path, fname), bbox_inches="tight")
    plt.close()


### Lambda map with GCs overlaid
def figure_effective_occurrence_rate_with_gcs(
    galaxy_cluster, gcs, lambda_map, out_path, lambda_eff, do_dimension=4
):
    # format the figure
    left = 0.1
    right = 0.87
    top = 0.98
    bottom = 0.1
    hspace = 0.0
    wspace = 0.0

    if do_dimension == 2:
        img = lambda_eff.T
    elif do_dimension == 3:
        img = lambda_eff[:, :, 0].T
    elif do_dimension == 4:
        img = lambda_eff[:, :, 0, 0].T

    # normalize all panels equally
    cmap = plt.get_cmap("inferno")
    min_value = img[img > 0].min()
    norm = mpl.colors.LogNorm(vmin=min_value, vmax=img.max())

    # create the Figure object with the correct WCS coordinates
    fig, axs = plt.subplots(
        1,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": lambda_map.wcs},
    )
    axs = numpy.atleast_1d(axs)
    axs = axs.ravel()

    axs[0].imshow(
        img,
        norm=norm,
        cmap="inferno",
        origin="lower",
        zorder=0,
        transform=axs[0].get_transform(lambda_map.wcs),
    )
    axs[0].annotate(
        lambda_map.name,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=16,
        color="red",
    )

    # GCs from Harris & Reina-Campos 2024
    axs[0].scatter(
        gcs.ra.to("deg"),
        gcs.dec.to("deg"),
        c="white",
        s=5,
        marker=".",
        edgecolor="k",
        linewidth=0.1,
        zorder=10,
        alpha=0.9,
        transform=axs[0].get_transform("fk5"),
    )
    axs[0].annotate(
        gcs.label,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=16,
        color="red",
    )

    # add the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # add an inset axes for the colorbar
    cax = axs[-1].inset_axes([1.001, 0.0, 0.02, 1.0])  # [x0, y0, width, height]
    # create the colorbar object
    cbar = fig.colorbar(sm, cax=cax, ax=axs[0])
    cbar.minorticks_on()  # add minorticks
    cbar.set_label(f"$\\lambda_{{\\rm eff}}$ - {lambda_map.name}")  # add label

    # Compute the angle corresponding to 10 pc at the distance of the galactic center
    scalebar_length = 100 * u.kpc
    scalebar_angle = (scalebar_length / galaxy_cluster.distance.to("kpc")).to(
        u.deg, equivalencies=u.dimensionless_angles()
    )
    # Add a scale bar
    add_scalebar(axs[-1], scalebar_angle, label="100 kpc", color="white")

    # format all axes
    for j, ax in enumerate(axs):
        ra = ax.coords[0]
        dec = ax.coords[1]
        dec.set_axislabel("Declination (J2000)")
        ra.set_axislabel("Right Ascension (J2000)")

        for obj in [ra, dec]:
            # set the formatting of the axes
            obj.set_major_formatter("dd:mm:ss")
            # display minor ticks
            obj.display_minor_ticks(True)
            obj.set_minor_frequency(10)

    # format the entire figure
    fig.subplots_adjust(
        left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
    )

    fname = "xy_lambdaeff_{:s}.png".format(lambda_map.name)
    fig.savefig(os.path.join(out_path, fname), bbox_inches="tight")
    plt.close()


### Interpolated map of the probability of recovery of the GCs
def figure_interpolated_map_gcs_2d(gcs, lambda_map, out_path):  # , do_dimension = 4):
    # create the interpolated map of the probability of recovery of the GCs for 2d: x and y

    # gcs.create_interpolated_map_probability_recovery_gcs(lambda_map.wcs, do_dimension = 2, do_return_extra_info=True)

    # create the Figure object
    fig, axs = plt.subplots(
        1,
        figsize=(10, 6.5),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": lambda_map.wcs},
    )
    axs = numpy.atleast_1d(axs)
    axs = axs.ravel()

    left = 0.1
    right = 0.87
    top = 0.98
    bottom = 0.1
    hspace = 0.0
    wspace = 0.0

    cmap = mpl.colormaps["inferno"]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    axs[0].annotate(
        gcs.label,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color="white",
        fontsize=16,
    )
    axs[0].imshow(gcs.intp_map.T, cmap=cmap, norm=norm, origin="lower", zorder=0)
    # show the dummy points that we have added to extend the map
    mask = gcs.intp_z == 0
    coords = lambda_map.wcs.all_pix2world(gcs.intp_xy[0][mask], gcs.intp_xy[1][mask], 1)
    print(coords)
    axs[0].scatter(
        coords[0] * u.deg,
        coords[1] * u.deg,
        marker="o",
        c="red",
        s=30,
        zorder=10,
        transform=axs[0].get_transform("fk5"),
    )
    # add the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # add an inset axes for the colorbar
    cax = axs[-1].inset_axes([1.02, 0.0, 0.02, 1.0])  # [x0, y0, width, height]
    # create the colorbar object
    cbar = fig.colorbar(sm, cax=cax, ax=axs[-1])
    cbar.minorticks_on()  # add minorticks
    cbar.set_label("Probability of recovery")  # add label

    # format all axes
    for i, ax in enumerate(axs):
        ra = ax.coords[0]
        dec = ax.coords[1]

        if i % 4 == 0:
            dec.set_axislabel("Declination (J2000)")
        else:
            dec.set_ticks_visible(False)
            dec.set_ticklabel_visible(False)
            dec.set_axislabel("")
        ra.set_axislabel("Right Ascension (J2000)")
        # set the formatting of the axes
        ra.set_major_formatter("hh:mm:ss.s")
        dec.set_major_formatter("dd:mm")
        # display minor ticks
        ra.display_minor_ticks(True)
        dec.display_minor_ticks(True)
        ra.set_minor_frequency(10)
        dec.set_minor_frequency(10)

    # format the entire figure
    fig.subplots_adjust(
        left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
    )

    fname = "xy_interpolated_map_2d_with_dummy_points.png"
    fig.savefig(os.path.join(out_path, fname), bbox_inches="tight")
    plt.close()
    # plt.show()


def figure_interpolated_map_gcs_3d(gcs, lambda_map, out_path):
    # create the interpolated map of the probability of recovery of the GCs for 3d: x, y and F150W

    # gcs.create_interpolated_map_probability_recovery_gcs(lambda_map.wcs, do_dimension = 3, do_return_extra_info=True)

    # create the Figure object
    fig, axs = plt.subplots(
        1,
        5,
        figsize=(20, 6),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": lambda_map.wcs},
    )
    axs = numpy.atleast_1d(axs)
    axs = axs.ravel()

    left = 0.1
    right = 0.87
    top = 0.98
    bottom = 0.1
    hspace = 0.0
    wspace = 0.0

    cmap = mpl.colormaps["inferno"]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    axs[0].annotate(
        "{:s}".format(gcs.label),
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color="white",
        fontsize=16,
    )
    for i, ax in zip(range(len(gcs.intp_bins[2])), axs):
        ax.imshow(
            gcs.intp_map[:, :, i].T, cmap=cmap, norm=norm, origin="lower", zorder=0
        )
        # show the dummy points that we have added to extend the map
        coords = lambda_map.wcs.all_pix2world(gcs.intp_xy[0], gcs.intp_xy[1], 1)
        mask = gcs.intp_z == 0
        # ax.scatter(coords[0][mask] * u.deg, coords[1][mask] * u.deg, marker = "o", c = "C0", s = 30, zorder = 10, transform=ax.get_transform('fk5'))
        # add data points in this bin
        mask = (
            (gcs.intp_xy[2] > gcs.intp_edges[2][i])
            * (gcs.intp_xy[2] < gcs.intp_edges[2][i + 1])
            * (gcs.intp_z > 0)
        )
        ax.scatter(
            coords[0][mask] * u.deg,
            coords[1][mask] * u.deg,
            marker="o",
            c=gcs.intp_z[mask],
            edgecolor="C1",
            s=30,
            zorder=10,
            transform=ax.get_transform("fk5"),
            cmap=cmap,
            norm=norm,
            lw=0.1,
        )
        ax.annotate(
            "F150W = {:.3f}".format(gcs.intp_bins[2][i]),
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            color="white",
            fontsize=16,
        )

    # add the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # add an inset axes for the colorbar
    cax = axs[-1].inset_axes([1.02, 0.0, 0.02, 2.0])  # [x0, y0, width, height]
    # create the colorbar object
    cbar = fig.colorbar(sm, cax=cax, ax=axs[-1])
    cbar.minorticks_on()  # add minorticks
    cbar.set_label("Probability of recovery")  # add label

    # format all axes
    for i, ax in enumerate(axs):
        ra = ax.coords[0]
        dec = ax.coords[1]

        if i % 5 == 0:
            dec.set_axislabel("Declination (J2000)")
        else:
            dec.set_ticks_visible(False)
            dec.set_ticklabel_visible(False)
            dec.set_axislabel("")
        # if i > 4:
        ra.set_axislabel("Right Ascension (J2000)")
        # else:
        #    ra.set_ticks_visible(False)
        #    ra.set_ticklabel_visible(False)
        #    ra.set_axislabel('')

        # set the formatting of the axes
        ra.set_major_formatter("hh:mm:ss.s")
        dec.set_major_formatter("dd:mm")

        # ax.tick_params(bottom = True, left= True, right = True, top = True, axis = "both", which = "both")
        # display minor ticks
        ra.display_minor_ticks(True)
        dec.display_minor_ticks(True)
        ra.set_minor_frequency(10)
        dec.set_minor_frequency(10)

    # format the entire figure
    fig.subplots_adjust(
        left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
    )

    fname = "xy_interpolated_map_3d_with_dummy_points.png"
    fig.savefig(os.path.join(out_path, fname), bbox_inches="tight")
    plt.close()
    # plt.show()


def figure_interpolated_map_gcs_4d(gcs, lambda_map, out_path):
    # create the interpolated map of the probability of recovery of the GCs for 4d: x, y, F150W and sigsky

    # gcs.create_interpolated_map_probability_recovery_gcs(lambda_map.wcs, do_dimension= 4, do_return_extra_info=True)

    for j in range(len(gcs.intp_bins[2])):
        # create the Figure object
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(14, 12),
            sharex=True,
            sharey=True,
            subplot_kw={"projection": lambda_map.wcs},
        )
        axs = numpy.atleast_1d(axs)
        axs = axs.ravel()

        left = 0.1
        right = 0.87
        top = 0.98
        bottom = 0.1
        hspace = 0.0
        wspace = 0.0

        cmap = mpl.colormaps["inferno"]
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        axs[0].annotate(
            "{:s}\nF150W = {:.3f}".format(gcs.label, gcs.intp_bins[2][j]),
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            color="red",
            fontsize=16,
        )
        for i, ax in zip(range(len(gcs.intp_bins[3])), axs):
            ax.imshow(
                gcs.intp_map[:, :, j, i].T,
                cmap=cmap,
                norm=norm,
                origin="lower",
                zorder=0,
            )
            # show the dummy points that we have added to extend the map
            coords = lambda_map.wcs.all_pix2world(gcs.intp_xy[0], gcs.intp_xy[1], 1)
            mask = gcs.intp_z == 0
            ax.scatter(
                coords[0][mask] * u.deg,
                coords[1][mask] * u.deg,
                marker="X",
                c="C0",
                s=30,
                zorder=10,
                transform=ax.get_transform("fk5"),
            )
            ax.annotate(
                r"$\log_{{10}}(\sigma_{{\rm sky}})$ = {:.2f}".format(
                    gcs.intp_bins[3][i]
                ),
                xy=(0.98, 0.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                color="red",
                fontsize=16,
            )

            # add data points in this bin
            mask = (
                (gcs.intp_xy[2] > gcs.intp_edges[2][j])
                * (gcs.intp_xy[2] < gcs.intp_edges[2][j + 1])
                * (gcs.intp_xy[3] > gcs.intp_edges[3][i])
                * (gcs.intp_xy[3] < gcs.intp_edges[3][i + 1])
                * (gcs.intp_z > 0)
            )
            ax.scatter(
                coords[0][mask] * u.deg,
                coords[1][mask] * u.deg,
                marker="o",
                c=gcs.intp_z[mask],
                edgecolor="C1",
                s=30,
                zorder=10,
                transform=ax.get_transform("fk5"),
                cmap=cmap,
                norm=norm,
            )

        # add the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # add an inset axes for the colorbar
        cax = axs[-1].inset_axes([1.02, 0.0, 0.02, 2.0])  # [x0, y0, width, height]
        # create the colorbar object
        cbar = fig.colorbar(sm, cax=cax, ax=axs[-1])
        cbar.minorticks_on()  # add minorticks
        cbar.set_label("Probability of recovery")  # add label

        # format all axes
        for i, ax in enumerate(axs):
            ra = ax.coords[0]
            dec = ax.coords[1]

            if i % 2 == 0:
                dec.set_axislabel("Declination (J2000)")
            else:
                dec.set_ticks_visible(False)
                dec.set_ticklabel_visible(False)
                dec.set_axislabel("")
            # if i > 3:
            ra.set_axislabel("Right Ascension (J2000)")
            # else:
            #    ra.set_ticks_visible(False)
            #    ra.set_ticklabel_visible(False)
            #    ra.set_axislabel('')

            # set the formatting of the axes
            ra.set_major_formatter("hh:mm:ss.s")
            dec.set_major_formatter("dd:mm")

            # display minor ticks
            ra.display_minor_ticks(True)
            dec.display_minor_ticks(True)
            ra.set_minor_frequency(10)
            dec.set_minor_frequency(10)

        # format the entire figure
        fig.subplots_adjust(
            left=left, top=top, bottom=bottom, right=right, hspace=hspace, wspace=wspace
        )

        fname = (
            "xy_interpolated_map_4d_with_dummy_points_f150w{:.3f}".format(
                gcs.intp_bins[2][j]
            ).replace(".", "p")
            + ".png"
        )
        fig.savefig(os.path.join(out_path, fname), bbox_inches="tight")
        plt.close()
        # plt.show()


### general routine to call the right function for the interpolated map
def figure_interpolated_map_gcs(gcs, lambda_map, out_path, do_dimension=4):
    # general routine to show the interpolated map of the probability of recovery of the GCs with the right number of dimensions
    if do_dimension == 2:
        figure_interpolated_map_gcs_2d(gcs, lambda_map, out_path)
    elif do_dimension == 3:
        figure_interpolated_map_gcs_3d(gcs, lambda_map, out_path)
    elif do_dimension == 4:
        figure_interpolated_map_gcs_4d(gcs, lambda_map, out_path)
    else:
        print("Error: do_dimension must be 2, 3 or 4")
        return


def figure_sampled_points_lambda_maps(
    lambda_map1,
    lambda_map2,
    out_path,
    coords,
    lambda_map_xlim_ra=[0, 0],
    lambda_map_ylim_dec=[0, 0],
    gcs_name="Bright GCs",
    do_lambda_map1="uniform",
    do_lambda_map2="uniform",
):
    # lambda maps with spawned datapoints
    fig = plt.figure(figsize=(10, 5))

    for i, map in enumerate([lambda_map1, lambda_map2]):
        ax = fig.add_subplot(1, 2, i + 1, projection=map.wcs)

        img = map.img.T.value
        ax.imshow(
            img,
            origin="lower",
            cmap="inferno",
            transform=ax.get_transform(map.wcs),
            zorder=0,
            norm=mpl.colors.LogNorm(vmin=img.max() / 1e5, vmax=img.max()),
        )
        if i == 0:
            label = "Sampled"
        else:
            label = ""
        ax.set_title(f"{label} $\\Lambda_{{\\rm eff {i+1}}}$ - {map.name}")

        ax.scatter(
            coords[0],
            coords[1],
            transform=ax.get_transform("fk5"),
            c="white",
            s=5,
            marker=".",
            edgecolor="k",
            linewidth=0.1,
            zorder=10,
            alpha=1,
        )
        ax.scatter(
            lambda_map_xlim_ra[0],
            lambda_map_ylim_dec[0],
            transform=ax.get_transform("fk5"),
            c="C0",
            s=30,
            marker="X",
            edgecolor="k",
            linewidth=0.1,
            zorder=10,
            alpha=0.9,
        )
        ax.scatter(
            lambda_map_xlim_ra[0],
            lambda_map_ylim_dec[1],
            transform=ax.get_transform("fk5"),
            c="C1",
            s=30,
            marker="X",
            edgecolor="k",
            linewidth=0.1,
            zorder=10,
            alpha=0.9,
        )
        ax.scatter(
            lambda_map_xlim_ra[1],
            lambda_map_ylim_dec[0],
            transform=ax.get_transform("fk5"),
            c="C2",
            s=30,
            marker="X",
            edgecolor="k",
            linewidth=0.1,
            zorder=10,
            alpha=0.9,
        )
        ax.scatter(
            lambda_map_xlim_ra[1],
            lambda_map_ylim_dec[1],
            transform=ax.get_transform("fk5"),
            c="C3",
            s=30,
            marker="X",
            edgecolor="k",
            linewidth=0.1,
            zorder=10,
            alpha=0.9,
        )
        ra = ax.coords[0]
        dec = ax.coords[1]
        if i == 0:
            dec.set_axislabel("Declination (J2000)")
        else:
            dec.set_ticks_visible(False)
            dec.set_ticklabel_visible(False)
            dec.set_axislabel("")
        ra.set_axislabel("Right Ascension (J2000)")
        for obj in [ra, dec]:
            # set the formatting of the axes
            obj.set_major_formatter("dd:mm:ss")
            # display minor ticks
            obj.display_minor_ticks(True)
            obj.set_minor_frequency(10)
    fname = os.path.join(
        out_path,
        f"imgs_{gcs_name}",
        f"fig_sampled_points_{do_lambda_map1}_{do_lambda_map2}.png",
    ).replace(" ", "_")
    fig.savefig(fname, bbox_inches="tight")
    plt.close()


def figure_lambda_eff_with_spawned_datapoints(
    lambda_map2,
    lambda_eff,
    coords,
    out_path,
    gcs_name="Bright GCs",
    do_lambda_map1="uniform",
    do_lambda_map2="uniform",
):
    # lambda eff with spawned datapoints
    fig = plt.figure(figsize=(10, 5))
    for i, map in enumerate([lambda_map2]):
        ax = fig.add_subplot(1, 1, i + 1, projection=map.wcs)
        cbnorm = mpl.colors.LogNorm(
            vmin=lambda_eff[:, :, 0, 0].min(), vmax=lambda_eff[:, :, 0, 0].max()
        )
        ax.imshow(
            lambda_eff[:, :, 0, 0].T,
            origin="lower",
            cmap="inferno",
            transform=ax.get_transform(map.wcs),
            zorder=0,
            norm=cbnorm,
        )
        ax.scatter(
            coords[0],
            coords[1],
            transform=ax.get_transform("fk5"),
            c="white",
            s=5,
            marker=".",
            edgecolor="k",
            linewidth=0.1,
            zorder=10,
            alpha=0.9,
        )
        # add the colorbar
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=cbnorm)
        # add an inset axes for the colorbar
        cax = ax.inset_axes([1.001, 0.0, 0.02, 1.0])  # [x0, y0, width, height]
        # create the colorbar object
        cbar = fig.colorbar(sm, cax=cax, ax=ax)
        cbar.minorticks_on()  # add minorticks
        cbar.set_label(
            f"$\\Lambda_{{\\rm eff}}$ - {map.name} - slice [0,0]"
        )  # add label
        ra = ax.coords[0]
        dec = ax.coords[1]
        if i == 0:
            dec.set_axislabel("Declination (J2000)")
        else:
            dec.set_ticks_visible(False)
            dec.set_ticklabel_visible(False)
            dec.set_axislabel("")
        ra.set_axislabel("Right Ascension (J2000)")
        for obj in [ra, dec]:
            # set the formatting of the axes
            obj.set_major_formatter("dd:mm:ss")
            # display minor ticks
            obj.display_minor_ticks(True)
            obj.set_minor_frequency(10)

    fname = os.path.join(
        out_path,
        f"imgs_{gcs_name}",
        f"fig_lambda_eff_{do_lambda_map1}_{do_lambda_map2}.png",
    ).replace(" ", "_")
    fig.savefig(fname, bbox_inches="tight")
    plt.close()
