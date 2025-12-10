import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Pixel-by-pixel correlation between two maps
    """
    )
    return


@app.cell(hide_code=True)
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Load the GC catalogue
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
def _():
    # determine the samples of GCs
    do_bright_gcs = True
    do_bright_blue_gcs = True
    do_bright_red_gcs = True
    do_high_quality_gcs = True

    # decide if you want to smooth the lambda map by the same kernel as the GCs
    do_smooth_lambda_map = False

    # decide whether to do the figures
    do_figures = True

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
        "Bergamini23",
        "Price24",
        "Cha24_WL",
        "Cha24_SL_WL",
        "Original",
        "BCGless",
        "X-ray",
    ]
    ls_lambda_type = [
        "lensing map",
        "lensing map",
        "lensing map",
        "lensing map",
        "stellar light",
        "bcgless map",
        "xray map",
    ]

    return (
        do_figures,
        do_smooth_lambda_map,
        ls_gcs_labels,
        ls_gcs_populations,
        ls_lambda_map,
        ls_lambda_type,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Main program
    """
    )
    return


@app.cell
def _(
    GCs,
    Table,
    abell2744,
    do_figures,
    do_smooth_lambda_map,
    figure_side_by_side_number_density_gcs_lambda_map,
    ls_gcs_labels,
    ls_gcs_populations,
    ls_lambda_map,
    ls_lambda_type,
    mfc,
    numpy,
    os,
    plt,
    scipy,
    u,
):
    # define the size of the smoothing kernel
    sigma_kpc = 20 * u.kpc
    sigma_arcsec = sigma_kpc / abell2744.arcsec_to_kpc

    # create the output path
    out_path = os.path.join(
        ".",
        "imgs",
        "pixel_by_pixel",
        f"smoothed_{sigma_kpc.to_string()}".replace(" ", "").replace(".", "p"),
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dict_results = {"Lambda Map": []}

    print("\n ### Starting the analysis ...")

    for _gcs_name, gcs_label in zip(ls_gcs_populations, ls_gcs_labels):
        _fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
        axs = axs.ravel()

        dict_results[f"{_gcs_name} - $r_{{\\rm S}}$"] = []
        dict_results[f"{_gcs_name} - $p$-value"] = []

        # prepare the dictionary to store results
        for i, _ax, do_lambda_map, _type_map in zip(
            range(10), axs, ls_lambda_map, ls_lambda_type
        ):
            print(f"\n *** [main] Comparing {_gcs_name} -- {do_lambda_map}")
            # create the instance of the GCs class
            bright_gcs = GCs(
                _gcs_name,
                gcs_label,
                abell2744,
            )

            print("[main] Bright GCs info:")
            print(f"  - min RA: {bright_gcs.ra.min()}")
            print(f"  - max RA: {bright_gcs.ra.max()}")
            print(f"  - min DEC: {bright_gcs.dec.min()}")
            print(f"  - max DEC: {bright_gcs.dec.max()}")
            # create the instance of the lensing map class
            lambda_map = mfc.create_instance_lambda_map(
                _type_map, do_lambda_map, bright_gcs, abell2744
            )

            print("[main] Lambda map info:")
            print(f"  - min: {lambda_map.img.min()}")
            print(f"  - max: {lambda_map.img.max()}")

            # convert the convergence maps to projected mass surface densities
            if _type_map == "lensing map":
                lambda_map.convert_to_projected_mass(abell2744)

            # mask GCs that are outside the edges of the lambda map
            # but add the edges to extend the GC map to cover the same FOV as the lambda map
            bright_gcs.mask_objects_outside_lambda_map(
                lambda_map.wcs, do_add_edges=True
            )

            if do_smooth_lambda_map:
                # smooth lambda map by the same kernel as the GC number density
                sigma_px = float(
                    sigma_arcsec
                    / (
                        numpy.mean(
                            numpy.abs(lambda_map.wcs.pixel_scale_matrix.diagonal())
                            * u.deg.to("arcsec")
                        )
                        * u.arcsec
                    )
                )
                # smoothed lambda map
                dummy_img = scipy.ndimage.gaussian_filter(
                    lambda_map.img.value, sigma_px
                )  # + 1e-10
                lambda_map.img = dummy_img * lambda_map.img.unit

            # create the WCS and header for the GCs
            bright_gcs.wcs, bright_gcs.header = bright_gcs.create_wcs_and_header(
                numpy.ones(shape=lambda_map.img.shape)
            )

            # create the number density image of GCs - units: arcsec^-2
            (
                bright_gcs.nrho_img,
                bright_gcs.nrho_smooth_img,
            ) = mfc.create_both_number_density_and_smoothed_maps_gcs(
                bright_gcs, lambda_map.img.shape, sigma_arcsec
            )

            # create the footprint of the GCs - removing first the edges of the lambda map
            bright_gcs.create_footprint_gcs(lambda_map.wcs, do_remove_edges=True)
            print("[main] CREATING the footprint of the GCs from themselves")

            bright_gcs.nrho_img *= bright_gcs.intp_mask.T
            bright_gcs.nrho_smooth_img *= bright_gcs.intp_mask.T
            lambda_map.img *= bright_gcs.intp_mask

            if do_figures:
                figure_side_by_side_number_density_gcs_lambda_map(
                    bright_gcs,
                    lambda_map,
                    label=bright_gcs.label,
                    fname=f"xy_{bright_gcs.name}_{do_lambda_map}.pdf".replace(" ", "_"),
                    out_path=out_path,
                )

            _mask = (bright_gcs.nrho_smooth_img.T.value > 5e-10) & (
                lambda_map.img.value > 1e-10
            )
            _xx = bright_gcs.nrho_smooth_img.T.value[_mask].flatten()
            _yy = lambda_map.img.value[_mask].flatten()
            # print(_xx.min(), _yy.min(), _xx.max(), _yy.max())
            # original data
            _ax.scatter(_xx, _yy, marker=".", color="C0", alpha=0.5)
            # calculate the correlation with a simple linear linear fit
            # _res = stats.linregress(numpy.log10(_xx), numpy.log10(_yy))
            # _xxpp = numpy.logspace(numpy.log10(_xx.min()), numpy.log10(_xx.max()), 100)
            # _ax.plot(_xxpp, numpy.power(10, _res.intercept + _res.slope*numpy.log10(_xxpp)), 'r', linestyle='--', linewidth=3)
            # _ax.annotate(f'$r^2=${_res.rvalue**2:.5f}', xy=(0.98, 0.98), ha='right', va='top', xycoords='axes fraction')
            # calculate the Spearman correlation coefficient
            _spearman_corr, _p_value = scipy.stats.spearmanr(_xx, _yy)
            _ax.annotate(
                f"$r_{{\\rm S}}=${_spearman_corr:.5f}\n $p$-value ={_p_value:.3e}",
                xy=(0.98, 0.98),
                ha="right",
                va="top",
                xycoords="axes fraction",
            )

            # _ax.set_ylabel(f'{lambda_map.name} [{lambda_map.img.unit.to_string()}]')
            # _ax.annotate(f'{lambda_map.name} [{lambda_map.img.unit.to_string()}]',
            #             xy=(0.02, 0.02), ha='left', va='bottom', xycoords='axes fraction')
            _ax.set_ylabel(f"{lambda_map.name} [{lambda_map.img.unit.to_string()}]")

            # store the results for later
            dict_results["Lambda Map"].append(do_lambda_map)
            dict_results[f"{_gcs_name} - $r_{{\\rm S}}$"].append(_spearman_corr)
            dict_results[f"{_gcs_name} - $p$-value"].append(_p_value)

            print("[main] Finished comparison!")

        for _i, _ax in enumerate(axs):
            _ax.set_xscale("log")
            _ax.set_yscale("log")
            if _i > 3:
                _ax.set_xlabel(r"n$_{\rm GCs}$ [arcsec$^{-2}$]")
            # if _i % 4 == 0:
            #    _ax.set_ylabel("LambdaMap")
            if _i == 7:
                _ax.set_axis_off()

        _fig.subplots_adjust(
            left=0.05, top=0.95, bottom=0.1, right=0.95, hspace=0.05, wspace=0.35
        )
        _fig.savefig(
            os.path.join(
                out_path,
                f"correlation_gcs_vs_lambda_maps_{_gcs_name}.png".replace(" ", "_"),
            ),
            bbox_inches="tight",
        )
        plt.close()
        # mo.md(f"""Here's the plot!{mo.as_html(_fig)}""")
        print()
    # save the results
    dict_results["Lambda Map"] = dict_results["Lambda Map"][:7]

    _table = Table(dict_results)
    _tname = os.path.join(out_path, f"table_spearman_rank.latex".replace(" ", "_"))
    _table.write(_tname, overwrite=True, format="ascii.latex")

    _table = Table(dict_results)
    _tname = os.path.join(out_path, f"table_spearman_rank.ecsv".replace(" ", "_"))
    _table.write(_tname, overwrite=True)
    return


@app.cell
def _(
    GCs,
    LogNorm,
    abell2744,
    ls_gcs_labels,
    ls_gcs_populations,
    ls_lambda_map,
    ls_lambda_type,
    mfc,
    numpy,
    os,
    plt,
    u,
):
    def do_ratio():
        # define the size of the smoothing kernel
        sigma_kpc = 20 * u.kpc
        sigma_arcsec = sigma_kpc / abell2744.arcsec_to_kpc

        # create the output path
        out_path = os.path.join(
            ".",
            "imgs",
            "pixel_by_pixel",
            f"smoothed_{sigma_kpc.to_string()}".replace(" ", "").replace(".", "p"),
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for _gcs_name, gcs_label in zip(ls_gcs_populations, ls_gcs_labels):
            _fig = plt.figure(figsize=(20, 10))
            _axs = []

            for i, do_lambda_map, _type_map in zip(
                range(10), ls_lambda_map, ls_lambda_type
            ):
                print(f"\n *** [main] Comparing {_gcs_name} -- {do_lambda_map}")
                # create the instance of the GCs class
                bright_gcs = GCs(_gcs_name, gcs_label, abell2744)

                # create the instance of the lensing map class
                lambda_map = mfc.create_instance_lambda_map(
                    _type_map, do_lambda_map, bright_gcs, abell2744
                )

                # convert the convergence maps to projected mass surface densities
                if _type_map == "lensing map":
                    lambda_map.convert_to_projected_mass(abell2744)

                # mask GCs that are outside the edges of the lambda map
                bright_gcs.mask_objects_outside_lambda_map(
                    lambda_map.wcs, do_add_edges=True
                )

                # create the WCS and header for the GCs
                bright_gcs.wcs, bright_gcs.header = bright_gcs.create_wcs_and_header(
                    numpy.ones(shape=lambda_map.img.shape)
                )

                # create the number density image of GCs - units: arcsec^-2
                (
                    bright_gcs.nrho_img,
                    bright_gcs.nrho_smooth_img,
                ) = mfc.create_both_number_density_and_smoothed_maps_gcs(
                    bright_gcs, lambda_map.img.shape, sigma_arcsec
                )

                # create the footprint of the GCs
                bright_gcs.create_footprint_gcs(lambda_map.wcs, do_remove_edges=True)
                print("[main] CREATING the footprint of the GCs from themselves")

                bright_gcs.nrho_img *= bright_gcs.intp_mask.T
                bright_gcs.nrho_smooth_img *= bright_gcs.intp_mask.T
                lambda_map.img *= bright_gcs.intp_mask

                # renormalize the figures between [0,1]
                lambda_map_img_renorm = renormalize_img(
                    lambda_map.img.value,
                    min=lambda_map.img.value.min(),
                    max=lambda_map.img.value.max(),
                )
                gcs_img_renorm = renormalize_img(
                    bright_gcs.nrho_smooth_img.value,
                    min=bright_gcs.nrho_smooth_img.value.min(),
                    max=bright_gcs.nrho_smooth_img.value.max(),
                )

                print(lambda_map_img_renorm.min(), lambda_map_img_renorm.max())
                print(gcs_img_renorm.min(), gcs_img_renorm.max())

                _ax = _fig.add_subplot(2, 4, i + 1, projection=bright_gcs.wcs)
                _cb = _ax.imshow(
                    gcs_img_renorm / lambda_map_img_renorm.T,
                    origin="lower",
                    cmap="BrBG",
                    norm=LogNorm(vmin=1e-3, vmax=1e3),
                    zorder=0,
                )
                _ax.set_ylabel(f"{lambda_map.name} [{lambda_map.img.unit.to_string()}]")
                _cax = _ax.inset_axes([0.0, 1.02, 1.0, 0.02])  # [x0, y0, width, height]
                _cbar = _fig.colorbar(_cb, cax=_cax, ax=_ax, orientation="horizontal")
                _cbar.ax.xaxis.set_ticks_position("top")
                _cbar.set_label(f"Ratio {bright_gcs.name} / {lambda_map.name}")
                _axs.append(_ax)

            for _i, _ax in enumerate(_axs):
                if _i == 7:
                    _ax.set_axis_off()

                ra = _ax.coords[0]
                dec = _ax.coords[1]
                if _i % 4 == 0:
                    dec.set_axislabel("Declination (J2000)")
                else:
                    dec.set_ticks_visible(True)
                    dec.set_ticklabel_visible(False)
                    dec.set_axislabel("")
                if _i >= 4:
                    ra.set_axislabel("Right Ascension (J2000)")
                else:
                    ra.set_ticks_visible(True)
                    ra.set_ticklabel_visible(False)
                    ra.set_axislabel("")
                # set the formatting of the axes
                ra.set_major_formatter("hh:mm:ss")
                dec.set_major_formatter("dd:mm")
                # display minor ticks
                ra.display_minor_ticks(True)
                dec.display_minor_ticks(True)
                ra.set_minor_frequency(6)
                dec.set_minor_frequency(12)
                ra.tick_params(
                    which="major",
                    direction="in",
                    top=True,
                    bottom=True,
                    length=10,
                    width=1,
                )
                dec.tick_params(
                    which="major",
                    direction="in",
                    right=True,
                    left=True,
                    length=10,
                    width=1,
                )
                ra.tick_params(which="minor", length=5)
                dec.tick_params(which="minor", length=5)

            _fig.subplots_adjust(
                left=0.05, top=0.95, bottom=0.1, right=0.95, hspace=0.05, wspace=0.35
            )
            _fig.savefig(
                os.path.join(
                    out_path,
                    f"ratio_gcs_over_lambda_maps_{_gcs_name}.png".replace(" ", "_"),
                ),
                bbox_inches="tight",
            )
            plt.show()

    do_ratio()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Functions
    """
    )
    return


@app.cell
def _(LogNorm, numpy, os, plt):
    def figure_side_by_side_number_density_gcs_lambda_map(
        sample_gcs, lambda_map, label="", fname="", out_path="./", **kwargs
    ):
        """Side-by-side figure of the smoothed number density of GCs and the LambdaMap"""

        def add_panel_for_gcs_with_contours_and_colorbar(
            fig, ax, sample, cmap, colors, **kwargs
        ):
            _img = sample.nrho_smooth_img.value
            # _img = sample.nrho_img.value + 1e-10
            _cb = ax.imshow(
                _img,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                norm=LogNorm(vmin=0.005, vmax=5),
            )
            # _cb = ax.imshow(_img, aspect='auto', origin='lower', cmap=cmap, norm=LogNorm(vmin=_img.min(), vmax=5))
            levels = [
                numpy.power(10, -1.9),
                numpy.power(10, -1.4),
                numpy.power(10, -1.2),
                numpy.power(10, -0.9),
                numpy.power(10, -0.4),
            ]
            contours = ax.contour(
                _img,
                levels=levels,
                colors=colors,
                transform=ax.get_transform(sample.wcs),
            )
            ax.annotate(
                sample.label,
                xy=(0.98, 0.98),
                ha="right",
                va="top",
                xycoords="axes fraction",
            )
            cax = ax.inset_axes([0.0, 1.0001, 1.0, 0.02])
            cbar = fig.colorbar(
                _cb, cax=cax, ax=ax, orientation="horizontal", location="top"
            )
            cbar.minorticks_on()
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.set_label("Number density [arcsec$^{-2}$]")

        def add_panel_for_lambda_with_colorbar(
            ax, img, cmap, label="", lim_cbar=[1e-10, 1]
        ):
            _cb = ax.imshow(
                img.value,
                origin="lower",
                cmap=cmap,
                norm=LogNorm(vmin=img[img.value > 0].value.min(), vmax=img.value.max()),
                zorder=0,
            )
            levels = [
                numpy.power(10, 8.3),
                numpy.power(10, 8.5),
                numpy.power(10, 8.7),
                numpy.power(10, 8.9),
                numpy.power(10, 9.1),
            ]
            colors = plt.get_cmap("viridis")(numpy.linspace(0.2, 1, 5))
            ax.contour(img.value, levels=levels, colors=colors, alpha=0.8)
            ax.annotate(
                label, xy=(0.98, 0.98), ha="right", va="top", xycoords="axes fraction"
            )
            cax = ax.inset_axes([0.0, 1.0001, 1.0, 0.02])
            cbar = fig.colorbar(
                _cb, cax=cax, ax=ax, orientation="horizontal", location="top"
            )
            cbar.minorticks_on()
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.set_label(f"LambdaMap [{lambda_map.img.unit.to_string()}]")
            return _cb

        dict_cmaps = {
            "Bright GCs": "Greys",
            "Bright Blue GCs": "Blues",
            "Bright Red GCs": "Reds",
            "High-quality GCs": "Greens",
        }
        dict_colors = {
            "Bright GCs": plt.get_cmap("Greys")(numpy.linspace(0.35, 1, 5)),
            "Bright Blue GCs": plt.get_cmap("Blues")(numpy.linspace(0.35, 1, 5)),
            "Bright Red GCs": plt.get_cmap("Reds")(numpy.linspace(0.35, 1, 5)),
            "High-quality GCs": plt.get_cmap("Greens")(numpy.linspace(0.35, 1, 5)),
        }

        fig = plt.figure(figsize=(12, 6))
        axs = []

        # smoothed number density of GCs
        ax = fig.add_subplot(1, 2, 1, projection=sample_gcs.wcs)
        axs.append(ax)
        add_panel_for_gcs_with_contours_and_colorbar(
            fig,
            ax,
            sample_gcs,
            dict_cmaps[sample_gcs.name],
            dict_colors[sample_gcs.name],
            **kwargs,
        )
        # mask = sample_gcs.prob == 0
        # ax.scatter(sample_gcs.ra.to("deg")[mask], sample_gcs.dec.to("deg")[mask],
        #           marker='o', color='C1', s=200, transform=ax.get_transform('fk5'))

        # LambdaMap
        ax = fig.add_subplot(1, 2, 2, projection=lambda_map.wcs)
        axs.append(ax)
        add_panel_for_lambda_with_colorbar(
            ax, lambda_map.img.T, "magma", label=lambda_map.label
        )

        lambda_xlim_ra = numpy.asarray(
            lambda_map.wcs.all_pix2world([0, lambda_map.img.shape[0]], [0, 0], 0)
        )[0]
        lambda_ylim_dec = numpy.asarray(
            lambda_map.wcs.all_pix2world([0, 0], [0, lambda_map.img.shape[1]], 0)
        )[1]
        for j, _ax, _sample in zip(range(10), axs, [sample_gcs, lambda_map]):
            _ax.set_xlabel("")
            ra = _ax.coords[0]
            dec = _ax.coords[1]
            if j % 4 == 0:
                dec.set_axislabel("Declination (J2000)")
            else:
                dec.set_ticks_visible(True)
                dec.set_ticklabel_visible(False)
                dec.set_axislabel("")
            ra.set_axislabel("Right Ascension (J2000)")
            ra.set_major_formatter("hh:mm:ss")
            dec.set_major_formatter("dd:mm")
            ra.display_minor_ticks(True)
            dec.display_minor_ticks(True)
            ra.set_minor_frequency(10)
            dec.set_minor_frequency(10)
            lim_pix = _sample.wcs.all_world2pix(lambda_xlim_ra, lambda_ylim_dec, 0)
            _ax.set_xlim(lim_pix[0])
            _ax.set_ylim(lim_pix[1])
        fig.subplots_adjust(
            left=0.05, top=0.95, bottom=0.1, right=0.95, hspace=0.0, wspace=0.05
        )
        fig.savefig(os.path.join(out_path, fname), bbox_inches="tight")
        plt.close()

    return (figure_side_by_side_number_density_gcs_lambda_map,)


@app.function
def renormalize_img(img, min=0, max=1):
    return (img - min) / (max - min)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Modules
    """
    )
    return


@app.cell
def _():
    # Import modules
    import marimo as mo
    import numpy, os, glob, scipy
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy import stats
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy import units as u
    from astropy import constants as c
    from astropy.table import Table
    from matplotlib.colors import LogNorm, SymLogNorm, Normalize
    from astropy.coordinates import SkyCoord
    import numpy.ma as ma

    # sys.path.append("/Users/mreina/Documents/Science/000-Projects/git-repos/git_shapes_gcs_dm/")
    # from functions_shapes import create_gcs_image
    from master_class_galaxy_cluster import GalaxyCluster
    from master_class_fits import FitsMap
    from master_class_gcs import GCs
    import master_functions_abell2744 as mfgc
    import master_functions_continuous as mfc

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 18.0
    mpl.rcParams["legend.fontsize"] = 16.0
    # mpl.rcParams['savefig.dpi'] = 100
    return (
        GCs,
        GalaxyCluster,
        LogNorm,
        Table,
        mfc,
        mo,
        numpy,
        os,
        plt,
        scipy,
        u,
    )


if __name__ == "__main__":
    app.run()
