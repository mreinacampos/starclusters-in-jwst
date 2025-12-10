import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Point-to-map comparisons via an inhomogeneous Poisson point process

    Notebook to calculate the log-likelihood of a given GC population to have been spawned from a continuous map/image assuming an inhomogenous Poisson point process

    Main assumption: GCs are independently distributed following an inhomogeneous Poisson process where the rate is set by a continuous image
    (e.g., a convergence map from lensing, a mass map from dynamics, etc.).
    Given a set of observed GCs and a map, this notebook calculates the log-likehood of such an event under the continuous assumption
    (i.e. each independent Borel set contains a single GC).

    Inputs:
    * Photometric catalogue of GCs in the galaxy cluster Abell 2744 from Harris & Reina-Campos 2023
    * Intensity/lambda maps from:
      * Projected mass maps (kappa) from lensing models of Abell 2744:
        * from the UNCOVER team - Furtak+ 2023
        * from Cha+ 2023 -- combining strong and weak lensing constraints
        * from Bergamini + 2023b
      * Stellar light from JWST NIRCam imaging of Abell 2744
      * X-ray maps from Chandra observations of Abell 2744

    Outputs:
    * Log-likelihood values for each GC population and each lambda map
    """
    )
    return


@app.cell
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
    # decide whether to be verbose or not
    do_verbose = True
    # create the output path for the tables
    out_path = os.path.join(".", "tables", "points_to_maps")
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
    return (
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
    mo.md(
        r"""
    ## Define properties of the galaxy cluster
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
    mvf,
    os,
    out_path,
    time,
):
    # loop over the different types of maps
    for gcs_name, gcs_label in zip(ls_gcs_populations, ls_gcs_labels):
        print("\n========================================")
        # create the dictionary to store the results
        dict_results = {}

        for do_lambda_map, type_map in zip(ls_lambda_map, ls_lambda_type):
            # create the instance of the GCs class
            bright_gcs = GCs(gcs_name, gcs_label, abell2744)
            print(
                f"\n*** {gcs_name}--{do_lambda_map} for {bright_gcs.mask_catalogue.sum()} GCs"
            )

            # create the instance of the lensing map class
            lambda_map = mfc.create_instance_lambda_map(
                type_map, do_lambda_map, bright_gcs, abell2744
            )

            # mask GCs that are outside the edges of the lambda map
            bright_gcs.mask_objects_outside_lambda_map(lambda_map.wcs)

            # read the local sky noise map -- MRC - to be updated
            fname = os.path.join(".", "data", "GCs_Harris23", "2508_skynoise_grid.fits")
            map_sky_noise = FitsMap(fname)
            print(
                f"[main] The local sky noise image ranges between {map_sky_noise.img.min()} and {map_sky_noise.img.max()}."
            )
            # find the limits of the lambda1 map in (RA, DEC)
            (
                lambda_map_xlim_ra,
                lambda_map_ylim_dec,
            ) = mfc.find_minimum_common_area_between_maps(
                lambda_map, lambda_map, map_sky_noise
            )

            # apply those limits to all maps
            lambda_map = mfc.apply_minimum_common_limits_to_image(
                lambda_map_xlim_ra, lambda_map_ylim_dec, lambda_map
            )
            map_sky_noise = mfc.apply_minimum_common_limits_to_image(
                lambda_map_xlim_ra, lambda_map_ylim_dec, map_sky_noise
            )

            # rebin the map of the local sky noise into the resolution of lambda map 1
            # - used to get the level of local sky noise at a given location
            (
                rebin_sky_noise_img,
                rebin_sky_noise_wcs,
                rebin_sky_noise_hdr,
            ) = mfc.reduce_and_rebin_image(lambda_map, map_sky_noise)

            start = time.time()
            ### Calculate the Poisson probability of observing the GCs given the lambda map and the selection function
            ln_prob = mfc.calculate_continuous_spatial_poisson_probability(
                lambda_map, bright_gcs, do_verbose=do_verbose
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
                # mvf.figure_effective_occurrence_rate_with_gcs(abell2744, bright_gcs, lambda_map,
                #                                              out_path_images, lambda_eff)

            # create the astropy Table to store it as ecsv
            table = Table(dict_results)
            fname = os.path.join(out_path, f"table_{gcs_name}.ecsv".replace(" ", "_"))
            table.write(fname, overwrite=True)

            # delete the objects to free up memory
            del bright_gcs, lambda_map, ln_prob, table
        # delete variables to free up memory
        del dict_results
    print("\n")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Functions
    """
    )
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
    # Import modules
    import numpy, os, time, glob
    import marimo as mo
    from astropy import units as u
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    from master_class_galaxy_cluster import GalaxyCluster
    from master_class_fits import FitsMap
    from master_class_gcs import GCs
    import master_validation_figures as mvf
    import master_functions_discrete as mfd
    import master_functions_continuous as mfc
    import master_functions_abell2744 as mfgc

    return FitsMap, GCs, GalaxyCluster, Table, mfc, mo, mvf, os, time, u


if __name__ == "__main__":
    app.run()
