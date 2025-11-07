import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Model testing: Cross-maps comparisons

    In order to tests the models used, we need to calculate the expected distribution of probability in the ideal case: "For a given model, what if all GCs were observed?"
    The way to test it is, for a given model, to spawn the same number of data points as observed GCs and compute the probability of them having been spawned from the map as usual. By repeating this step many times, we'll obtain the expected distribution of probabibility.

    We can also change the number of data points that we spawn, to test the convergence of the results.

    We can also use this technique to do cross-model validation: e.g. spawn from the noisy/uniform map and compare to any of the convergence maps.
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
    number_iterations = 500  # 200 #500
    # decide whether to be verbose
    do_verbose = False

    # create the output path
    out_path = os.path.join(".", "tables", "maps_to_maps")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # determine the samples of GCs
    do_bright_gcs = True
    do_bright_blue_gcs = True
    do_bright_red_gcs = True
    do_high_quality_gcs = True

    ls_gcs_populations = []
    ls_gcs_labels = []
    if do_bright_gcs:
        ls_gcs_populations.append("Bright GCs")
        ls_gcs_labels.append("F150W$ < 29.5$")
    if do_bright_blue_gcs:
        ls_gcs_populations.append("Bright Blue GCs")
        ls_gcs_labels.append("F150W<29.5\n(F115W-F200W)$_0 < 0$")
    if do_bright_red_gcs:
        ls_gcs_populations.append("Bright Red GCs")
        ls_gcs_labels.append("F150W<29.5\n(F115W-F200W)$_0 \geq 0$")
    if do_high_quality_gcs:
        ls_gcs_populations.append("High-quality GCs")
        ls_gcs_labels.append("F150W<29.5\nZone 1 and 2")

    for key in ls_gcs_populations:
        path = os.path.join(out_path, "imgs_" + key.replace(" ", "_"))
        if not os.path.exists(path):
            os.makedirs(path)

    # determine the lambda maps (predictor maps) to compare against
    ls_lambda_map = [
        "uniform",
        "noisy",
        "Cha24_SL_WL",
        "Cha24_WL",
        "X-ray",
        "Original",
        "BCGless",
        "Price24",
        "Bergamini23",
    ]
    ls_lambda_type = [
        "uniform map",
        "noisy map",
        "lensing map",
        "lensing map",
        "xray map",
        "stellar light",
        "bcgless map",
        "lensing map",
        "lensing map",
    ]

    # ls_lambda_map = ["uniform", "X-ray"]
    # ls_lambda_type = ["uniform map", "xray map"]

    # ls_lambda_map = ["X-ray"]
    # ls_lambda_type = ["xray map"]
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
    Table,
    abell2744,
    do_figures,
    do_verbose,
    ls_gcs_labels,
    ls_gcs_populations,
    ls_lambda_map,
    ls_lambda_type,
    mfc,
    mpl,
    mvf,
    number_iterations,
    numpy,
    os,
    out_path,
    plt,
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
                    "Blue" in gcs_name or "Red" in gcs_name
                ) and do_lambda_map1 != do_lambda_map2:
                    continue

                # prepare a dictionary to store the results
                ls_results = []
                dict_results = {}
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

                # rebin the map of the local sky noise into the resolution of lambda map 2
                # - used to calculate the normalization factor
                (
                    rebin_sky_noise_img_map2,
                    rebin_sky_noise_wcs_map2,
                    rebin_sky_noise_hdr_map2,
                ) = mfc.reduce_and_rebin_image(lambda_map2, map_sky_noise)

                if do_figures:
                    _fig = plt.figure(figsize=(20, 8.5))
                    _axs = []
                    _ax = plt.subplot(131, projection=lambda_map1.wcs)
                    _cb = _ax.imshow(
                        lambda_map1.img.value.T,
                        origin="lower",
                        cmap="viridis",
                        norm=mpl.colors.LogNorm(
                            vmin=lambda_map1.img.value.max() / 1e5,
                            vmax=lambda_map1.img.value.max(),
                        ),
                    )
                    _ax.set_title("Lambda map 1 - {:s}".format(type_map1))
                    # cax = _fig.add_axes([0.15, 0.15, 0.7, 0.02])
                    _cbar = _fig.colorbar(_cb, ax=_ax, orientation="horizontal")
                    _axs.append(_ax)
                    _ax = plt.subplot(132, projection=map_prob_recovery.wcs)
                    _cb = _ax.imshow(
                        map_prob_recovery.img.value.T,
                        origin="lower",
                        cmap="viridis",
                        norm=mpl.colors.Normalize(vmin=0, vmax=1),
                    )
                    _cbar = _fig.colorbar(_cb, ax=_ax, orientation="horizontal")
                    _ax.set_title("Probability of recovery map")
                    _axs.append(_ax)
                    _ax = plt.subplot(133, projection=lambda_map1.wcs)
                    _cb = _ax.imshow(
                        wgt_img.value.T,
                        origin="lower",
                        cmap="viridis",
                        norm=mpl.colors.LogNorm(
                            vmin=wgt_img.value.max() / 1e5, vmax=wgt_img.value.max()
                        ),
                    )
                    _cbar = _fig.colorbar(_cb, ax=_ax, orientation="horizontal")
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
                            lambda_map1.wcs,
                            map_prob_recovery.wcs,
                            lambda_map1.wcs,
                        ],
                        [
                            lambda_map1.header,
                            map_prob_recovery.header,
                            lambda_map1.header,
                        ],
                    ):
                        # for each sample, covert the edges in (RA,DEC) to pixels and apply those limits to the panel
                        _lim_pix = _wcs.all_world2pix(_xlim_ra, _ylim_dec, 0)
                        _ax.set_xlim(_lim_pix[0])
                        _ax.set_ylim(_lim_pix[1])

                        _ax.scatter(
                            _hdr["CRVAL1"],
                            _hdr["CRVAL2"],
                            transform=_ax.get_transform("fk5"),
                            c="red",
                            s=50,
                            marker="o",
                            edgecolor="k",
                            linewidth=0.1,
                            zorder=10,
                            alpha=0.9,
                        )
                        print(
                            "Center of pixel (RA, DEC):", _hdr["CRVAL1"], _hdr["CRVAL2"]
                        )
                        _ax.scatter(
                            lambda_map_xlim_ra[0],
                            lambda_map_ylim_dec[0],
                            transform=_ax.get_transform("fk5"),
                            c="C0",
                            s=30,
                            marker="X",
                            edgecolor="k",
                            linewidth=0.1,
                            zorder=10,
                            alpha=0.9,
                        )
                        _ax.scatter(
                            lambda_map_xlim_ra[0],
                            lambda_map_ylim_dec[1],
                            transform=_ax.get_transform("fk5"),
                            c="C1",
                            s=30,
                            marker="X",
                            edgecolor="k",
                            linewidth=0.1,
                            zorder=10,
                            alpha=0.9,
                        )
                        _ax.scatter(
                            lambda_map_xlim_ra[1],
                            lambda_map_ylim_dec[0],
                            transform=_ax.get_transform("fk5"),
                            c="C2",
                            s=30,
                            marker="X",
                            edgecolor="k",
                            linewidth=0.1,
                            zorder=10,
                            alpha=0.9,
                        )
                        _ax.scatter(
                            lambda_map_xlim_ra[1],
                            lambda_map_ylim_dec[1],
                            transform=_ax.get_transform("fk5"),
                            c="C3",
                            s=30,
                            marker="X",
                            edgecolor="k",
                            linewidth=0.1,
                            zorder=10,
                            alpha=0.9,
                        )

                        _ax.set_aspect("equal")

                    _fname = os.path.join(
                        out_path,
                        f"imgs_{gcs_name}",
                        f"fig_wgt_img_{do_lambda_map1}.png",
                    ).replace(" ", "_")
                    _fig.savefig(_fname, bbox_inches="tight")
                    plt.close()

                if do_figures:
                    _fig = plt.figure(figsize=(16, 8.5))
                    _axs = []
                    _ax = plt.subplot(121)
                    _cb = _ax.imshow(
                        lambda_map1.img.value.T,
                        origin="lower",
                        cmap="viridis",
                        norm=mpl.colors.LogNorm(
                            vmin=lambda_map1.img.value.max() / 1e5,
                            vmax=lambda_map1.img.value.max(),
                        ),
                    )
                    _ax.set_title("Lambda map 1 - {:s}".format(type_map1))
                    # cax = _fig.add_axes([0.15, 0.15, 0.7, 0.02])
                    _cbar = _fig.colorbar(_cb, ax=_ax, orientation="horizontal")
                    _axs.append(_ax)
                    _ax = plt.subplot(122)
                    _cb = _ax.imshow(
                        map_prob_recovery.img.T,
                        origin="lower",
                        cmap="viridis",
                        norm=mpl.colors.LogNorm(
                            vmin=map_prob_recovery.img.max() / 1e5,
                            vmax=map_prob_recovery.img.max(),
                        ),
                    )
                    _cbar = _fig.colorbar(_cb, ax=_ax, orientation="horizontal")
                    _ax.set_title("Probability of recovery map")
                    _axs.append(_ax)

                    for _i, _ax, _wcs, _hdr in zip(
                        range(10),
                        _axs,
                        [lambda_map1.wcs, lambda_map1.wcs],
                        [lambda_map1.header, lambda_map1.header],
                    ):
                        _ax.scatter(
                            _hdr["CRPIX1"],
                            _hdr["CRPIX2"],
                            c="red",
                            s=50,
                            marker="o",
                            edgecolor="k",
                            linewidth=0.1,
                            zorder=10,
                            alpha=0.9,
                        )
                        _ax.set_aspect("equal")

                    _fname = os.path.join(
                        out_path,
                        f"imgs_{gcs_name}",
                        f"fig_wgt_img_{do_lambda_map1}_pixels.png",
                    ).replace(" ", "_")
                    _fig.savefig(_fname, bbox_inches="tight")
                    plt.close()

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
                    # select the magnitudes
                    f150w = (
                        gnr_f150w[sample * number_gcs : (sample + 1) * number_gcs]
                        * u.ABmag
                    )
                    # determine the local sky noise in the (x,y) coordinates spawned -- MRC to be checked
                    log10sigsky = (
                        rebin_sky_noise_img_map1[inds[:, 0], inds[:, 1]]
                        * u.dimensionless_unscaled
                    )
                    # calculate the probability of recovery for the spawned points
                    prob_recovery = bright_gcs.probability_of_recovery(
                        f150w, log10sigsky
                    )

                    # Create a DataPoints object to store the sampled positions
                    datapoints = GCs(
                        "Dummy datapoints",
                        "Dummy datapoints",
                        abell2744,
                        ra=coords[0],
                        dec=coords[1],
                        f150w=f150w,
                        log10sigsky=log10sigsky,
                        prob=prob_recovery,
                    )
                    if do_verbose:
                        print(
                            f"[main] Time to create DataPoints instance - time elapsed = {time.time() - start_time:.2f} s"
                        )

                    if do_figures and sample == 0:
                        lambda_map1.img = wgt_img.copy()
                        mvf.figure_sampled_points_lambda_maps(
                            lambda_map1,
                            lambda_map2,
                            out_path,
                            coords,
                            lambda_map_xlim_ra,
                            lambda_map_ylim_dec,
                            gcs_name,
                            do_lambda_map1,
                            do_lambda_map2,
                        )

                    if do_figures and sample == 0:
                        normalization = mfc.calculate_normalization_poisson_probability(
                            f150w_min=bright_gcs.f150w.min(),
                            f150w_max=bright_gcs.f150w.max(),
                            map_sky_noise=rebin_sky_noise_img_map2,
                            gcs=bright_gcs,
                            lambda_map=lambda_map2,
                        )
                        print(
                            f"[main] Normalization factor for {do_lambda_map2} is {normalization:.4e}"
                        )

                    start = time.time()
                    ### Calculate the Poisson probability of observing the GCs given the lambda map and the selection function
                    ln_prob = mfc.calculate_continuous_spatial_poisson_probability(
                        normalization, lambda_map2, datapoints, do_verbose=do_verbose
                    )
                    ls_results.append(ln_prob)
                    if do_verbose:
                        print(
                            f"[main] Time to calculate the probability - time elapsed = {time.time() - start:.2f} s"
                        )
                    if sample % 100 == 0:
                        print(
                            f"Sample {sample} done - ln_prob = {ln_prob} - time elapsed = {time.time() - start_time:.2f} s"
                        )
                    dict_results[
                        "{:s}-{:s}".format(do_lambda_map1, do_lambda_map2)
                    ] = ls_results

                    # delete the variables to save memory
                    del coords, datapoints, inds, ln_prob

                print(f"Cross-testing of {do_lambda_map1}-{do_lambda_map2} is done")

                # save the results to a table
                table = Table(dict_results)
                tname = os.path.join(
                    out_path,
                    f"table_{gcs_name}_testing_{do_lambda_map1}-{do_lambda_map2}.ecsv".replace(
                        " ", "_"
                    ),
                )
                table.write(tname, overwrite=True)
                print("Table saved to {:s}".format(tname))

                # delete the variables to save memory
                del gnr_inds, gnr_f150w, ls_results, dict_results, table

            del lambda_map1, lambda_map2
        # delete the variables to save memory
        del bright_gcs
    return


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
    return (
        FitsMap,
        GCs,
        GalaxyCluster,
        Table,
        mfc,
        mo,
        mpl,
        mvf,
        numpy,
        os,
        plt,
        time,
        u,
    )


if __name__ == "__main__":
    import multiprocessing

    # prefer 'fork' on mac to avoid extra spawn semaphores (careful with threads/GUI)
    try:
        multiprocessing.set_start_method("fork", force=True)
    except Exception:
        # already set or not allowed; continue
        pass

    try:
        app.run()
    finally:
        # Best-effort cleanup of leftover child processes (terminates and joins)
        for p in multiprocessing.active_children():
            try:
                p.terminate()
                p.join(timeout=2)
            except Exception:
                pass
