import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Calculate the probability of the observed GC population to have been spawned from the lensing map following a inhomogenous Poisson point process

    Assuming GCs are independently distributed following an inhomogeneous Poisson process where the rate is set by the convergence map, what is the probability that we would measure this distribution of GCs given the map?

    * Photometric catalogue of GCs from Harris & Reina-Campos 2023
        * select only GCs in Zone 1 of sky noise
    * Convergence maps:
        * from the UNCOVER team - Furtak+ 2023
        * from Cha+ 2023 -- combining strong and weak lensing constraints
        * from Bergamini + 2023b
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Decide the type of analysis""")
    return


@app.cell
def _(os):
    # decide on the dimensionality of the interpolation
    do_dimension = 4
    # decide whether to render all the figures or not
    do_figures = True
    # decide whether to be verbose or not
    do_verbose = True
    # create the output path for the tables
    out_path = os.path.join(".", "tables_discrete_points_to_maps")
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

    ls_lambda_map = ["uniform"]
    ls_lambda_type = ["uniform map"]
    return (
        do_dimension,
        do_figures,
        do_verbose,
        ls_gcs_labels,
        ls_gcs_populations,
        ls_lambda_map,
        ls_lambda_type,
        out_path,
    )


@app.cell
def _(mo):
    mo.md(r"""## Define properties of the galaxy cluster""")
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
    GCs,
    SkyCoord,
    Table,
    abell2744,
    do_dimension,
    do_figures,
    do_verbose,
    glob,
    ls_gcs_labels,
    ls_gcs_populations,
    ls_lambda_map,
    ls_lambda_type,
    mfd,
    mfgc,
    mvf,
    numpy,
    os,
    out_path,
    time,
    u,
):
    # loop over the different types of maps
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

        # create the dictionary to store the results
        dict_results = {}

        for do_lambda_map, type_map in zip(ls_lambda_map, ls_lambda_type):
            print(f"\n*** {gcs_name}--{do_lambda_map} for {mask.sum()} GCs")
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

            # create the instance of the lensing map class
            lambda_map = mfgc.create_instance_lambda_map(
                type_map, do_lambda_map, bright_gcs, abell2744
            )

            # mask GCs that are outside the edges of the lambda map
            bright_gcs.mask_objects_outside_lambda_map(lambda_map.wcs)

            # create the output path
            out_path_smaps = os.path.join(
                out_path,
                "maps_selection_function_{:s}".format(
                    bright_gcs.name.replace(" ", "_")
                ),
            )
            if not os.path.exists(out_path_smaps):
                os.makedirs(out_path_smaps)

            # check whether the interpolated map exists, and only created if not
            start = time.time()
            fname = os.path.join(
                out_path_smaps, f"maps_SBi_{do_lambda_map}_{gcs_name}".replace(" ", "_")
            )
            ls_smaps = glob.glob(
                os.path.join(
                    out_path_smaps, f"maps_SBi_{do_lambda_map}_{gcs_name}*"
                ).replace(" ", "_")
            )
            if fname + "_maps.npz" in ls_smaps:
                # read the interpolated map and all the related information
                mfd.read_information_about_interpolation(fname, bright_gcs)
                print("[main] READING the interpolated selection function")
            else:
                # create the inteporlated map of the probability of recovery of the GCs
                bright_gcs.create_interpolated_map_probability_recovery_gcs(
                    lambda_map.wcs,
                    do_dimension=do_dimension,
                    do_return_extra_info=True,
                    do_interpolation=True,
                )
                # save the interpolated map and all the related information
                mfd.save_information_about_interpolation(fname, bright_gcs)
                print("[main] CREATING the interpolated selection function")

            print(
                "Time to create/read the interpolated map of the probability of recovery of the GCs: {:.2f} s".format(
                    time.time() - start
                )
            )

            # creates an lambda_map.img_extended with the same dimensions as the interpolated_map_gcs
            lambda_map.extend_dimensions_to_higher_dimensionality(
                do_dimension=do_dimension,
                mask=bright_gcs.intp_mask,
                shape=bright_gcs.intp_map.shape,
            )

            start = time.time()
            # 1: renormalize the map to be between [0, 1] to get rid of the units of the effective rate
            # (i.e. solar masses, fluxes, photon counts, etc.)
            lambda_eff = mfd.determine_effective_occurrence_rate(
                bright_gcs, lambda_map, do_verbose=do_verbose
            )

            # determine the size of the bins in each dimension: \Delta V_i = \Delta x_i \Delta y_i \Delta \theta_i
            dV = mfd.determine_pixel_size(bright_gcs, do_verbose=do_verbose)

            # find the pixels in which data points (points or GCs) lie within the high dimensionality space
            pixels = mfd.find_pixels_within_lambda_map(
                bright_gcs, lambda_map, do_dimension=do_dimension
            )

            # calculate the normalization factor of the Poisson probability (first_term)
            norm = mfd.calculate_normalization_factor(
                lambda_eff, dV, do_verbose=do_verbose
            )

            ### Calculate the Poisson probability of observing the GCs given the lambda map and the selection function
            ln_prob = mfd.calculate_spatial_poisson_probability(
                lambda_eff,
                dV,
                pixels,
                norm,
                do_dimension=len(lambda_map.img_extended.shape),
                do_verbose=do_verbose,
            )
            dict_results[do_lambda_map] = [ln_prob]
            print(
                "Time to calculate the spatial Poisson probability: {:.2f} s".format(
                    time.time() - start
                )
            )

            if do_figures:
                # create the output path
                out_path_images = os.path.join(
                    out_path,
                    "validation_lambda_{:s}".format(bright_gcs.name.replace(" ", "_")),
                )
                if not os.path.exists(out_path_images):
                    os.makedirs(out_path_images)

                # show the lensing model alongside the GCs
                mvf.figure_lensing_model_with_gcs(
                    abell2744, bright_gcs, lambda_map, out_path_images
                )
                # show the effective occurrence rate map
                mvf.figure_effective_occurrence_rate_with_gcs(
                    abell2744, bright_gcs, lambda_map, out_path_images, lambda_eff
                )
                # show the interpolated map of the probability of recovery of the GCs
                mvf.figure_interpolated_map_gcs(
                    bright_gcs, lambda_map, out_path_images, do_dimension=do_dimension
                )

            # create the astropy Table to store it as ecsv
            table = Table(dict_results)
            fname = os.path.join(out_path, f"table_{gcs_name}.ecsv".replace(" ", "_"))
            table.write(fname, overwrite=True)

            # delete the objects to free up memory
            del bright_gcs, lambda_map
            # delete the variables to free up memory
            del lambda_eff, dV, pixels, norm, ln_prob, table
        # delete variables to free up memory
        del gc_catalogue, coords_gcs, dict_results
    print("\n")
    return


@app.cell
def _(mo):
    mo.md(r"""## Functions""")
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""

        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Modules""")
    return


@app.cell
def _():
    # Import modules
    import numpy, os, time, glob, pickle
    import marimo as mo
    from astropy import units as u
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    from master_class_galaxy_cluster import GalaxyCluster
    from master_class_gcs import GCs
    import master_validation_figures as mvf
    import master_functions_discrete as mfd
    import master_functions_abell2744 as mfgc
    import master_validation_figures as mvf

    return (
        GCs,
        GalaxyCluster,
        SkyCoord,
        Table,
        glob,
        mfd,
        mfgc,
        mo,
        mvf,
        numpy,
        os,
        time,
        u,
    )


if __name__ == "__main__":
    app.run()
