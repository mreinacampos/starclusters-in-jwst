import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Visualization of the cross-maps comparisons

    Using the sample of Bright GCs and the Price24 lambda map
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Decide the type of analysis
    """
    )
    return


@app.cell
def _(os):
    # decide whether to render all the figures or not
    do_figures = True
    # decide how many iteration we'll do
    number_iterations = 1
    # decide whether to be verbose
    do_verbose = False

    # create the output path
    out_path = os.path.join(".", "imgs", "maps_to_maps")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # determine the samples of GCs
    do_bright_gcs = True

    ls_gcs_populations = []
    ls_gcs_labels = []

    if do_bright_gcs:
        ls_gcs_populations.append("Bright GCs")
        ls_gcs_labels.append("F150W$ < 29.5$")

    # determine the lambda maps (predictor maps) to compare against
    ls_lambda_map = [
        "Price24",
    ]
    ls_lambda_type = [
        "lensing map",
    ]

    return (
        do_figures,
        do_verbose,
        ls_gcs_labels,
        ls_gcs_populations,
        ls_lambda_map,
        ls_lambda_type,
        number_iterations,
        out_path,
    )


@app.cell
def _(mo):
    _df = mo.sql(
        f"""

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define the properties of the galaxy cluster

    Needed to re-scale the images from pixels to coordinates
    """
    )
    return


@app.cell
def _(GalaxyCluster, u):
    # 1: create the instance of the Galaxy Cluster class for Abell 2744
    # luminosity distance to Abell2744
    distance_to_a2744 = 1630 * u.Mpc
    # from Harris & Reina-Campos 2023
    arcsec_kpc = (
        2100 * u.kpc / (460 * u.arcsec)
    )  # conversion between arcsec and kpc at the distance of Abell 2744
    # create the instance of the Galaxy Cluster class for Abell 2744
    abell2744 = GalaxyCluster(
        "Abell2744",
        distance=distance_to_a2744,
        redshift=0.308,
        arcsec_to_kpc=arcsec_kpc,
    )
    return (abell2744,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Main program
    """
    )
    return


@app.cell
def _(
    FitsMap,
    GCs,
    abell2744,
    do_figures,
    do_verbose,
    figure_sampled_points_lambda_maps,
    ls_gcs_labels,
    ls_gcs_populations,
    ls_lambda_map,
    ls_lambda_type,
    mfc,
    number_iterations,
    os,
    out_path,
    time,
    u,
):
    # loop over each GC sample - it defines the number of data points to spawn
    for gcs_name, gcs_label in zip(ls_gcs_populations, ls_gcs_labels):
        # 2: create the instance of the GCs class
        bright_gcs = GCs(gcs_name, gcs_label, abell2744)

        # decide how many data points to spawn
        number_gcs = len(bright_gcs.f150w)

        for do_lambda_map2, type_map2 in zip(
            ls_lambda_map, ls_lambda_type
        ):  # map against to compare
            for do_lambda_map1, type_map1 in zip(
                ls_lambda_map, ls_lambda_type
            ):  # map to spawn from
                # skip the cases where the maps are not the same for Blue/Red GCs
                if (
                    "Blue" in gcs_name
                    or "Red" in gcs_name
                    or "High-quality" in gcs_name
                ) and do_lambda_map1 != do_lambda_map2:
                    continue

                # prepare a dictionary to store the results
                print(
                    f"\n*** {gcs_name}: {do_lambda_map1}--{do_lambda_map2} for {number_gcs} GCs"
                )

                # create the instance of the first lambda map - the one from which we'll spawn the datapoints
                lambda_map1 = mfc.create_instance_lambda_map(
                    type_map1, do_lambda_map1, bright_gcs, abell2744
                )

                # create the instance of the second lambda map - the one from which we'll compare against
                lambda_map2 = mfc.create_instance_lambda_map(
                    type_map2, do_lambda_map2, bright_gcs, abell2744
                )

                # read the local sky noise map -- MRC - to be updated
                fname = os.path.join(
                    ".", "data", "GCs_Harris23", "2508_skynoise_grid.fits"
                )
                map_sky_noise = FitsMap(fname)
                print(
                    f"[main] The local sky noise image ranges between {map_sky_noise.img.min()} and {map_sky_noise.img.max()}."
                )
                # find the limits of the lambda1 map in (RA, DEC)
                (
                    lambda_map_xlim_ra,
                    lambda_map_ylim_dec,
                ) = mfc.find_minimum_common_area_between_maps(
                    lambda_map1, lambda_map2, map_sky_noise
                )

                # apply those limits to all maps
                lambda_map1 = mfc.apply_minimum_common_limits_to_image(
                    lambda_map_xlim_ra, lambda_map_ylim_dec, lambda_map1
                )
                lambda_map2 = mfc.apply_minimum_common_limits_to_image(
                    lambda_map_xlim_ra, lambda_map_ylim_dec, lambda_map2
                )
                map_sky_noise = mfc.apply_minimum_common_limits_to_image(
                    lambda_map_xlim_ra, lambda_map_ylim_dec, map_sky_noise
                )

                # rebin the map of the local sky noise into the resolution of lambda map 1
                # - used to get the level of local sky noise at a given location
                (
                    rebin_sky_noise_img_map1,
                    rebin_sky_noise_wcs_map1,
                    rebin_sky_noise_hdr_map1,
                ) = mfc.reduce_and_rebin_image(lambda_map1, map_sky_noise)

                # calculate the map of the probability of recovery given the rebinned local sky noise map
                map_prob_recovery = FitsMap(fname)
                # using a dummy value for the F150W to create a pseudo-map for the probability of recovery given a map of local sky noise
                map_prob_recovery.img = (
                    bright_gcs.probability_of_recovery(
                        29.0 * u.ABmag, rebin_sky_noise_img_map1
                    )
                    * u.dimensionless_unscaled
                )
                map_prob_recovery.wcs = rebin_sky_noise_wcs_map1
                map_prob_recovery.header = rebin_sky_noise_hdr_map1
                # map to spawn datapoints from: a combination of LambdaMap1 and the pseudo-probability of recovery
                wgt_img = lambda_map1.img * map_prob_recovery.img

                # if do_figures:
                #  figure_effective_lambda_map(lambda_map1, map_prob_recovery, wgt_img, out_path, gcs_name, do_lambda_map1)

                # spawn data points from the lambda map
                start = time.time()
                gnr_inds = mfc.spawn_datapoints_from_image(
                    number_gcs * number_iterations, wgt_img.value
                )
                print(
                    f"[main] Spawning datapoints from Lambda1 took {time.time() - start} seconds"
                )
                # assuming a GCLF, spawn number_gcs data points from it
                start = time.time()
                gnr_f150w = mfc.spawn_magnitudes(
                    number_gcs * number_iterations,
                    min_mag=bright_gcs.f150w.min(),
                    max_mag=bright_gcs.f150w.max(),
                    m0=bright_gcs.luminosity_function_mean_mag,
                    sigma=bright_gcs.luminosity_function_sigma_mag,
                    do_verbose=do_verbose,
                )
                print(
                    f"[main] Spawning the magnitudes took {time.time() - start} seconds"
                )
                print(
                    f"[main] The magnitudes range between {bright_gcs.f150w.min()} and {bright_gcs.f150w.max()}."
                )

                # loop over the iterations
                for sample in range(number_iterations):
                    # start counter
                    start_time = time.time()

                    # select number_gcs points from the generated points
                    inds = gnr_inds[sample * number_gcs : (sample + 1) * number_gcs]
                    # determine the coordinates of the (x,y) pixels spawned
                    coords = (
                        lambda_map1.wcs.all_pix2world(
                            inds[:, 0], inds[:, 1], 0, ra_dec_order=True
                        )
                        * u.deg
                    )

                    if do_figures and sample == 0:
                        figure_sampled_points_lambda_maps(
                            lambda_map1,
                            wgt_img,
                            lambda_map2,
                            out_path,
                            coords,
                            do_lambda_map1,
                            do_lambda_map2,
                        )

                    # delete the variables to save memory
                    del coords, inds
                print(f"Cross-testing of {do_lambda_map1}-{do_lambda_map2} is done")

                # delete the variables to save memory
                del (
                    gnr_inds,
                    gnr_f150w,
                )

            del lambda_map1, lambda_map2
        # delete the variables to save memory
        del bright_gcs
    return


@app.cell
def _(mpl, numpy, os, plt):
    def figure_effective_lambda_map(
        lambda_map, map_prob_recovery, wgt_img, out_path, gcs_name, do_lambda_map
    ):
        _fig = plt.figure(figsize=(12, 18.5))
        _axs = []
        _ax = plt.subplot(311, projection=lambda_map.wcs)
        _cb = _ax.imshow(
            lambda_map.img.value.T,
            origin="lower",
            cmap="viridis",
            norm=mpl.colors.LogNorm(
                vmin=1e-4,
                vmax=10,
            ),
        )
        _ax.set_title("Lambda map 1 - {:s}".format(lambda_map.name))
        _axs.append(_ax)
        _ax = plt.subplot(312, projection=map_prob_recovery.wcs)
        _cb = _ax.imshow(
            map_prob_recovery.img.value.T,
            origin="lower",
            cmap="viridis",
            norm=mpl.colors.Normalize(vmin=0, vmax=1),
        )
        _ax.set_title("Probability of recovery map")
        _axs.append(_ax)
        _ax = plt.subplot(313, projection=lambda_map.wcs)
        _cb = _ax.imshow(
            wgt_img.value.T,
            origin="lower",
            cmap="viridis",
            norm=mpl.colors.LogNorm(
                vmin=1e-4,
                vmax=10,
            ),
        )
        _ax.set_title(r"$\lambda_{\rm eff} = \lambda_1 * S$")
        _axs.append(_ax)

        _xlim_ra = numpy.asarray(
            map_prob_recovery.wcs.all_pix2world(
                [0, map_prob_recovery.img.shape[0]], [0, 0], 0
            )
        )[0]
        _ylim_dec = numpy.asarray(
            map_prob_recovery.wcs.all_pix2world(
                [0, 0], [0, map_prob_recovery.img.shape[1]], 0
            )
        )[1]

        for _i, _ax, _wcs, _hdr in zip(
            range(10),
            _axs,
            [
                lambda_map.wcs,
                map_prob_recovery.wcs,
                lambda_map.wcs,
            ],
            [
                lambda_map.header,
                map_prob_recovery.header,
                lambda_map.header,
            ],
        ):
            # for each sample, covert the edges in (RA,DEC) to pixels and apply those limits to the panel
            _lim_pix = _wcs.all_world2pix(_xlim_ra, _ylim_dec, 0)
            _ax.set_xlim(_lim_pix[0])
            _ax.set_ylim(_lim_pix[1])

            ra = _ax.coords[0]
            dec = _ax.coords[1]
            if _i == 0:
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
            # set the aspect ratio to be equal
            _ax.set_aspect("equal")

        _fname = os.path.join(
            out_path,
            f"fig_wgt_img_{do_lambda_map}.png",
        ).replace(" ", "_")
        _fig.savefig(_fname, bbox_inches="tight")
        plt.close()

    return


@app.cell
def _(mpl, numpy, os, plt):
    def figure_sampled_points_lambda_maps(
        lambda_map1,
        wgt_img,
        lambda_map2,
        out_path,
        coords,
        do_lambda_map1="uniform",
        do_lambda_map2="uniform",
    ):
        # lambda maps with spawned datapoints
        fig = plt.figure(figsize=(18, 5.5))
        _axs = []

        _ax = fig.add_subplot(131, projection=lambda_map1.wcs)
        _cb = _ax.imshow(
            wgt_img.value.T,
            origin="lower",
            cmap="inferno",
            norm=mpl.colors.LogNorm(
                vmin=1e-4,
                vmax=10,
            ),
        )
        _ax.set_title(
            r"$\lambda_{\rm eff} (x,y,\theta) = \lambda_1 (x,y) \times S (\theta)$"
        )
        _axs.append(_ax)

        _ax = fig.add_subplot(132, projection=lambda_map2.wcs)
        _ax.scatter(
            coords[0],
            coords[1],
            transform=_ax.get_transform("fk5"),
            c="k",
            s=5,
            marker=".",
            edgecolor="k",
            linewidth=0.1,
            zorder=10,
            alpha=0.8,
        )
        _ax.set_title(r"Sampled datapoints")
        _axs.append(_ax)

        # lambda map 2
        _ax = fig.add_subplot(133, projection=lambda_map2.wcs)
        img = lambda_map2.img.T.value
        _ax.imshow(
            img,
            origin="lower",
            cmap="inferno",
            transform=_ax.get_transform(lambda_map2.wcs),
            zorder=0,
            norm=mpl.colors.LogNorm(vmin=img.max() / 1e5, vmax=img.max()),
        )
        _ax.set_title(r"$\lambda_{2} (x,y)$")
        _axs.append(_ax)

        _xlim_ra = numpy.asarray(
            lambda_map1.wcs.all_pix2world([0, lambda_map1.img.shape[0]], [0, 0], 0)
        )[0]
        _ylim_dec = numpy.asarray(
            lambda_map1.wcs.all_pix2world([0, 0], [0, lambda_map1.img.shape[1]], 0)
        )[1]

        for _i, _ax, _wcs in zip(
            range(10), _axs, [lambda_map1.wcs, lambda_map2.wcs, lambda_map2.wcs]
        ):
            _lim_pix = _wcs.all_world2pix(_xlim_ra, _ylim_dec, 0)

            # for each sample, covert the edges in (RA,DEC) to pixels and apply those limits to the panel
            _ax.set_xlim(_lim_pix[0])
            _ax.set_ylim(_lim_pix[1])

            ra = _ax.coords[0]
            dec = _ax.coords[1]
            dec.set_axislabel("")
            ra.set_axislabel("")
            dec.set_ticklabel_visible(False)
            ra.set_ticklabel_visible(False)
            ra.display_minor_ticks(True)
            dec.display_minor_ticks(True)
            ra.set_minor_frequency(6)
            dec.set_minor_frequency(12)
            ra.tick_params(
                which="major", direction="in", top=True, bottom=True, length=10, width=1
            )
            dec.tick_params(
                which="major", direction="in", right=True, left=True, length=10, width=1
            )
            ra.tick_params(which="minor", length=5)
            dec.tick_params(which="minor", length=5)

            _ax.set_aspect("equal")

        fname = os.path.join(
            out_path,
            f"fig_sampled_points_{do_lambda_map1}_{do_lambda_map2}.pdf",
        ).replace(" ", "_")
        fig.savefig(fname, bbox_inches="tight")
        plt.close()

    return (figure_sampled_points_lambda_maps,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Modules
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    # Import modules
    import numpy, os, time, copy, glob
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from astropy import units as u
    from astropy.table import Table
    from astropy.io import fits
    from astropy import wcs
    from astropy.coordinates import SkyCoord
    from reproject import reproject_interp

    from master_class_lambdamaps import LensingMap, StellarLightMap, XrayMap
    from master_class_fits import FitsMap
    from master_class_galaxy_cluster import GalaxyCluster
    from master_class_gcs import GCs, DataPoints
    import master_validation_figures as mvf
    import master_functions_discrete as mfd
    import master_functions_continuous as mfc
    import master_functions_abell2744 as mfgc
    import master_validation_figures as mvf
    import master_functions_continuous as mfc

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 18.0
    mpl.rcParams["legend.fontsize"] = 16.0
    return FitsMap, GCs, GalaxyCluster, mfc, mo, mpl, numpy, os, plt, time, u


if __name__ == "__main__":
    app.run()
