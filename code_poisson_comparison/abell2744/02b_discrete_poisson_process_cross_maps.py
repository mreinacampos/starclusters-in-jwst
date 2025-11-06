import marimo

__generated_with = "0.14.6"
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
    mo.md(r"""## Decide the type of analysis""")
    return


@app.cell
def _(os):
    # decide on the dimensionality of the interpolation
    do_dimension = 4
    # decide whether to render all the figures or not
    do_figures = True
    # decide how many iteration we'll do
    number_iterations = 200  # 500
    # decide whether to be verbose
    do_verbose = False

    # create the output path
    out_path = os.path.join(".", "tables_discrete_maps_to_maps")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # determine the samples of GCs
    do_bright_gcs = True
    do_bright_blue_gcs = False
    do_bright_red_gcs = False

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

    return (
        do_dimension,
        do_figures,
        do_verbose,
        ls_gcs_labels,
        ls_gcs_populations,
        ls_lambda_map,
        ls_lambda_type,
        number_iterations,
        out_path,
    )


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
    mo.md(r"""## Main program""")
    return


@app.cell
def _(
    DataPoints,
    GCs,
    SkyCoord,
    Table,
    abell2744,
    copy,
    do_dimension,
    do_figures,
    do_verbose,
    find_minimum_common_area_between_maps,
    glob,
    ls_gcs_labels,
    ls_gcs_populations,
    ls_lambda_map,
    ls_lambda_type,
    mf,
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
        # 2: create the GC catalogue from Harris & Reina-Campos 2024
        # load the GC catalogue from Harris & Reina-Campos 2024
        gc_catalogue = mfgc.load_gc_catalogue()

        # prepare the right mask based on the sample of GCs
        if gcs_name == "Bright GCs":
            mask = gc_catalogue["F150W"].to(u.ABmag) < 29.5 * u.ABmag
        elif gcs_name == "Bright Blue GCs":
            mask = (gc_catalogue["F150W"].to(u.ABmag) < 29.5 * u.ABmag) * (
                gc_catalogue["F115W0"] - gc_catalogue["F200W0"] < 0
            )
        elif gcs_name == "Bright Red GCs":
            mask = (gc_catalogue["F150W"].to(u.ABmag) < 29.5 * u.ABmag) * (
                gc_catalogue["F115W0"] - gc_catalogue["F200W0"] >= 0
            )

        # determine the coordinates of the GCs
        coords_gcs = SkyCoord(
            ra=gc_catalogue[mask]["RA [J2000]"],
            dec=gc_catalogue[mask]["DEC [J2000]"],
            frame="fk5",
            distance=abell2744.distance,
        )

        # create the instance of the GCs class
        bright_gcs = GCs(
            gcs_name,
            gcs_label,
            coords_gcs.ra.to("deg"),
            coords_gcs.dec.to("deg"),
            gc_catalogue[mask]["F150W"].to(u.ABmag),
            numpy.log10(gc_catalogue[mask]["sigsky"].value),
            gc_catalogue[mask]["prob"],
        )

        # decide how many data points to spawn
        number_gcs = len(bright_gcs.f150w)

        for do_lambda_map2, type_map2 in zip(
            ["BCGless", "Price24", "Bergamini23"],
            ["bcgless map", "lensing map", "lensing map"],
        ):  # map against to compare
            for do_lambda_map1, type_map1 in zip(
                ls_lambda_map, ls_lambda_type
            ):  # map to spawn from
                if ("BCGless" in do_lambda_map2) and do_lambda_map1 != "Bergamini23":
                    continue

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

                # create two instances of the GC populations - to load the tailored inteporlated maps and edges
                gcs1 = copy.deepcopy(bright_gcs)
                gcs2 = copy.deepcopy(bright_gcs)

                # create the instance of the first lambda map - the one from which we'll spawn the datapoints
                lambda_map1 = mfd.create_instance_lambda_map(
                    type_map1, do_lambda_map1, gcs1, abell2744
                )

                # create the instance of the second lambda map - the one from which we'll compare against
                lambda_map2 = mfd.create_instance_lambda_map(
                    type_map2, do_lambda_map2, gcs2, abell2744
                )

                # read the interpolated map and all the related information
                out_path_smaps = os.path.join(
                    ".",
                    "tables_discrete_points_to_maps",
                    "maps_selection_function_{:s}".format(gcs1.name.replace(" ", "_")),
                )
                fname = os.path.join(
                    out_path_smaps,
                    f"maps_SBi_{do_lambda_map1}_{gcs_name}".replace(" ", "_"),
                )
                ls_smaps = glob.glob(
                    os.path.join(
                        out_path_smaps, f"maps_SBi_{do_lambda_map1}_{gcs_name}*"
                    ).replace(" ", "_")
                )
                if fname + "_maps.npz" in ls_smaps:
                    mfd.read_information_about_interpolation(fname, gcs1)
                    print("[main] READING the interpolated selection function")
                else:
                    print(
                        f"[main] ERROR - selection function {gcs_name}--{do_lambda_map1} is NOT ready"
                    )

                if do_figures:  # - MRC
                    # intensity histogram of the lambda map - MRC
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.hist(
                        numpy.log10(lambda_map1.img[lambda_map1.img > 0].value),
                        bins=100,
                        density=False,
                        histtype="step",
                        color="C0",
                        label=f"Before - Masked",
                    )

                start = time.time()
                # creates an lambda_map.img_extended with the same dimensions as the interpolated_map_gcs
                if do_dimension == 2:
                    shape = lambda_map1.img.shape
                elif do_dimension == 3:
                    shape = (*lambda_map1.img.shape, len(gcs1.intp_bins[2]))
                else:
                    shape = (
                        *lambda_map1.img.shape,
                        len(gcs1.intp_bins[2]),
                        len(gcs1.intp_bins[3]),
                    )
                # find the limits of the lambda1 map in (RA, DEC)
                (
                    lambda_map_xlim_ra,
                    lambda_map_ylim_dec,
                ) = find_minimum_common_area_between_maps(lambda_map1, lambda_map2)
                lim_pix = lambda_map1.wcs.all_world2pix(
                    lambda_map_xlim_ra, lambda_map_ylim_dec, 0
                )
                yy, xx = numpy.meshgrid(range(shape[1]), range(shape[0]))
                # restrict the range of lambda map1 to avoid spawning datapoints where there's no information in lambda map2
                lambda_map1.img[xx < lim_pix[0][0]] = 0 * lambda_map1.img.unit
                lambda_map1.img[xx > lim_pix[0][1]] = 0 * lambda_map1.img.unit
                lambda_map1.img[yy < lim_pix[1][0]] = 0 * lambda_map1.img.unit
                lambda_map1.img[yy > lim_pix[1][1]] = 0 * lambda_map1.img.unit
                # apply the mask from the interpolated map of the GCs to the lambda map 1
                # -- only spawning datapoints where the GCs are
                # lambda_map1.img[~gcs1.intp_mask] = 0 * lambda_map1.img.unit # 1e-10 * lambda_map1.img.min()
                lambda_map1.extend_dimensions_to_higher_dimensionality(
                    do_dimension=do_dimension, mask=gcs1.intp_mask, shape=shape
                )

                if do_figures:
                    ax.hist(
                        numpy.log10(lambda_map1.img[lambda_map1.img > 0].value),
                        bins=100,
                        density=False,
                        histtype="step",
                        color="C3",
                        label=f"After - Masked",
                    )
                    ax.legend(loc="best")
                    ax.set_xlabel(f"$\\Lambda_1$ - {lambda_map1.name}")
                    ax.set_ylabel("Density")
                    ax.set_yscale("log")
                    fname = os.path.join(
                        out_path,
                        f"imgs_{gcs_name}",
                        f"fig_hist_lambda_{do_lambda_map1}.png",
                    ).replace(" ", "_")
                    fig.savefig(fname, bbox_inches="tight")
                    plt.close()

                start = time.time()
                # create an extended lambda map where each layer has been weighted by the relative abundance of GCs in that layer
                # -- only needed to spawn datapoints
                wgt_img = lambda_map1.img_extended * gcs1.intp_map
                # spawn data points from the lambda map
                gnr_inds = mfd.spawn_datapoints_from_image(
                    number_gcs * number_iterations,
                    wgt_img,
                    do_dimension,
                    name=do_lambda_map1,
                    do_verbose=do_figures,
                )
                print(
                    f"[main] Spawning datapoints from Lambda1 took {time.time() - start} seconds"
                )

                wgt_img = mfd.modify_image_to_mimic_observations(
                    lambda_map1.img_extended, gcs1, do_dimension
                )

                # read the interpolated map and all the related information
                fname = os.path.join(
                    out_path_smaps,
                    f"maps_SBi_{do_lambda_map2}_{gcs_name}".replace(" ", "_"),
                )
                ls_smaps = glob.glob(
                    os.path.join(
                        out_path_smaps, f"maps_SBi_{do_lambda_map2}_{gcs_name}*"
                    ).replace(" ", "_")
                )
                if fname + "_maps.npz" in ls_smaps:
                    mfd.read_information_about_interpolation(fname, gcs2)
                    print("[main] READING the interpolated selection function")
                else:
                    print(
                        f"[main] ERROR - selection function {gcs_name}--{do_lambda_map2} is NOT ready"
                    )

                start = time.time()
                # creates an lambda_map.img_extended with the same dimensions as the interpolated_map_gcs
                if do_dimension == 2:
                    shape = lambda_map2.img.shape
                elif do_dimension == 3:
                    shape = (*lambda_map2.img.shape, len(gcs2.intp_bins[2]))
                else:
                    shape = (
                        *lambda_map2.img.shape,
                        len(gcs2.intp_bins[2]),
                        len(gcs2.intp_bins[3]),
                    )
                lambda_map2.extend_dimensions_to_higher_dimensionality(
                    do_dimension=do_dimension, mask=gcs2.intp_mask, shape=shape
                )
                print(f"[main] Extending Lambda2 took {time.time() - start} seconds")

                # loop over the iterations
                for sample in range(number_iterations):
                    # start counter
                    start_time = time.time()

                    # select number_gcs points from the generated points
                    inds = gnr_inds[sample * number_gcs : (sample + 1) * number_gcs]

                    # Create a DataPoints object to store the sampled positions
                    coords = (
                        lambda_map1.wcs.all_pix2world(
                            inds[:, 0], inds[:, 1], 0, ra_dec_order=True
                        )
                        * u.deg
                    )
                    if do_dimension >= 3:
                        f150w = gcs1.intp_bins[2][inds[:, 2]]
                    else:
                        f150w = numpy.ones(shape=number_gcs)
                    if do_dimension == 4:
                        log10sigsky = gcs1.intp_bins[3][inds[:, 3]]
                    else:
                        log10sigsky = numpy.ones(shape=number_gcs)
                    datapoints = DataPoints(
                        do_lambda_map1,
                        do_lambda_map1,
                        coords[0],
                        coords[1],
                        header=lambda_map1.header,
                    )
                    datapoints = GCs(
                        "Dummy datapoints",
                        "Dummy datapoints",
                        coords[0],
                        coords[1],
                        f150w,
                        log10sigsky,
                        numpy.ones(shape=number_gcs),
                    )
                    # grab the bins and edges for the interpolated map from the observed GCs
                    datapoints.intp_bins = gcs2.intp_bins
                    datapoints.intp_edges = gcs2.intp_edges
                    datapoints.intp_mask = gcs2.intp_mask
                    datapoints.intp_map = gcs2.intp_map

                    if do_verbose:
                        print(
                            f"[main] Time to create DataPoints instance - time elapsed = {time.time() - start_time:.2f} s"
                        )

                    if sample == 0:
                        # 1: renormalize the map to be between [0, 1] to get rid of the units of the effective rate
                        # (i.e. solar masses, fluxes, photon counts, etc.)
                        lambda_eff = mfd.determine_effective_occurrence_rate(
                            datapoints,
                            lambda_map2,
                            do_selection_function=True,
                            do_verbose=do_verbose,
                        )

                        # determine the size of the bins in each dimension: \Delta V_i = \Delta x_i \Delta y_i \Delta \theta_i
                        dV = mfd.determine_pixel_size(datapoints, do_verbose=do_verbose)

                        # calculate the normalization factor of the Poisson probability (first_term)
                        norm = mfd.calculate_normalization_factor(
                            lambda_eff, dV, do_verbose=do_verbose
                        )

                    if do_figures and sample == 0:
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
                        mvf.figure_lambda_eff_with_spawned_datapoints(
                            lambda_map2,
                            lambda_eff,
                            coords,
                            out_path,
                            gcs_name,
                            do_lambda_map1,
                            do_lambda_map2,
                        )

                    # find the pixels in which data points (points or GCs) lie within the high dimensionality space
                    pixels = mfd.find_pixels_within_lambda_map(
                        datapoints, lambda_map2, do_dimension=do_dimension
                    )

                    start = time.time()
                    ### Calculate the Poisson probability of observing the GCs given the lambda map and the selection function
                    ln_prob = mfd.calculate_spatial_poisson_probability(
                        lambda_eff,
                        dV,
                        pixels,
                        norm,
                        do_dimension=do_dimension,
                        do_verbose=do_verbose,
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
                    del coords, datapoints, pixels, inds, ln_prob

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
                del gnr_inds, lambda_eff, dV, norm, ls_results, dict_results, table

            # delete the variables to save memory
            del gcs1, gcs2
            del lambda_map1, lambda_map2
        # delete the variables to save memory
        del bright_gcs, coords_gcs, gc_catalogue

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Functions""")
    return


@app.cell
def _(numpy):
    def find_minimum_common_area_between_maps(map1, map2):
        # find the coordinates of the edges in each map
        map1_xlim_ra = numpy.asarray(
            map1.wcs.all_pix2world([-0.5, map1.img.shape[0] + 0.5], [-0.5, -0.5], 0)
        )[0]
        map1_ylim_dec = numpy.asarray(
            map1.wcs.all_pix2world([-0.5, -0.5], [-0.5, map1.img.shape[1] + 0.5], 0)
        )[1]

        map2_xlim_ra = numpy.asarray(
            map2.wcs.all_pix2world([-0.5, map2.img.shape[0] + 0.5], [-0.5, -0.5], 0)
        )[0]
        map2_ylim_dec = numpy.asarray(
            map2.wcs.all_pix2world([-0.5, -0.5], [-0.5, map2.img.shape[1] + 0.5], 0)
        )[1]

        # find the minimum common area
        map_xlim_ra = [
            numpy.min([map1_xlim_ra[0], map2_xlim_ra[0]]),
            numpy.max([map1_xlim_ra[1], map2_xlim_ra[1]]),
        ]
        map_ylim_dec = [
            numpy.max([map1_ylim_dec[0], map2_ylim_dec[0]]),
            numpy.min([map1_ylim_dec[1], map2_ylim_dec[1]]),
        ]

        print(
            "[find_minimum_common_area_between_maps], BOTH - RA, DEC",
            map_xlim_ra,
            map_ylim_dec,
        )

        return map_xlim_ra, map_ylim_dec

    return (find_minimum_common_area_between_maps,)


@app.cell
def _(mo):
    mo.md(r"""# Modules""")
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
    from astropy.coordinates import SkyCoord
    from master_class_lambdamaps import LensingMap, StellarLightMap, XrayMap
    from master_class_fits import FitsMap
    from master_class_galaxy_cluster import GalaxyCluster
    from master_class_gcs import GCs, DataPoints
    import master_functions_discrete as mfd
    import master_functions_abell2744 as mfgc
    import master_validation_figures as mvf

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 18.0
    mpl.rcParams["legend.fontsize"] = 16.0

    return (
        DataPoints,
        GCs,
        GalaxyCluster,
        SkyCoord,
        Table,
        copy,
        glob,
        mf,
        mo,
        mvf,
        numpy,
        os,
        plt,
        time,
        u,
    )


if __name__ == "__main__":
    app.run()
