import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Code to transform a table into a FITS image
    ## Author: Marta Reina-Campos
    ## Date: August 2025

    The code below transforms any column of a given table into a FITS image with the right header and WCS objects.
    """
    )
    return


@app.cell
def _(SkyCoord, ascii, create_fits_image, fits, numpy, os, plt, u, wcs):
    # define the filename and read the table
    fname = os.path.join(".", "data", "GCs_Harris26", "skymatrix.txt")
    table = ascii.read(
        fname,
        names=["RA", "DEC", "x (px)", "y (px)", "log(sigsky)"],
        guess=False,
        fast_reader=False,
    )

    # redefine the coordinates -- now pixels of the image
    table["img x (px)"] = ((table["x (px)"] - table["x (px)"].min()) / 10).astype(int)
    table["img y (px)"] = ((table["y (px)"] - table["y (px)"].min()) / 10).astype(int)

    # convert the RA , DEC to astropy coordinate objects
    coords = SkyCoord(ra=table["RA"] * u.deg, dec=table["DEC"] * u.deg, frame="icrs")
    table["ra (deg)"] = coords.ra.to("deg")
    table["dec (deg)"] = coords.dec.to("deg")
    table["ra (arcsec)"] = coords.ra.to("arcsec")
    table["dec (arcsec)"] = coords.dec.to("arcsec")

    # create the image by reshaping the array of log(sigsky)
    # set a dummy value of -10
    img = numpy.ones(
        shape=(table["img x (px)"].max() + 1, table["img y (px)"].max() + 1)
    ) * (table["log(sigsky)"].max() * 3)
    img[table["img x (px)"], table["img y (px)"]] = table["log(sigsky)"]

    ### VALIDATION -- plot the image
    print("*** First validation figure", img.shape)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(img.T, origin="lower", vmin=0, vmax=2, cmap="viridis")
    ax.set_ylabel("Image y (px)")
    ax.set_xlabel("Image x (px)")
    plt.show()

    # convert it to a FITS file with the appropriate header
    out_fname = os.path.join(".", "data", "GCs_Harris26", "2511_skynoise_grid.fits")
    create_fits_image(table, img, out_fname)

    ### VALIDATION
    # load the FITS image and WCS object
    with fits.open(out_fname, output_verify="fix") as _fits_table:
        _header = _fits_table[0].header
        _img = _fits_table[
            0
        ].data.T  # transpose the image, it is read as (rows, columns) otherwise
        _wcs = wcs.WCS(_header)
    print("*** Second validation figure")
    fig, ax = plt.subplots(1, figsize=(10, 6), subplot_kw={"projection": _wcs})
    cb = ax.imshow(_img.T, origin="lower", cmap="viridis", vmin=0, vmax=2)
    # add the colorbar
    cax = ax.inset_axes([1.001, 0.0, 0.02, 1.0])  # [x0, y0, width, height]
    # create the colorbar object
    cbar = fig.colorbar(cb, cax=cax, ax=ax)
    cbar.minorticks_on()  # add minorticks
    cbar.set_label(r"$\log_{10}(\sigma_{\rm sky})$")  # add label

    plt.show()
    return (out_fname,)


@app.cell
def _(mo):
    mo.md(r"""### Validation against the Price24 Lambda Map""")
    return


@app.cell
def _(fits, numpy, os, out_fname, plt, u, wcs):
    if True:

        def figure_noise_map_with_lambda_map_contours(
            noise_img, noise_wcs, lensing_img, lensing_wcs, **kwargs
        ):
            def add_panel_noise_map(ax, img, label="", lim_cbar=[0, 1]):
                # cb = ax.imshow(img.T, origin = "lower", cmap = cmap, norm = LogNorm(vmin = lim_cbar[0], vmax = lim_cbar[1]), zorder = 0)
                levels = [-4, 1.8, 2.1, 2.4, 6]
                colors = plt.get_cmap("Greys")(numpy.linspace(0.2, 1, len(levels)))
                ax.contourf(img.T, levels=levels, colors=colors, alpha=0.8)
                ax.annotate(
                    label,
                    xy=(0.98, 0.98),
                    ha="right",
                    va="top",
                    xycoords="axes fraction",
                    color="white",
                )

            def add_panel_stellar_light_map(ax, lmap, cmap, label="", lim_cbar=[0, 1]):
                img = lmap.T / numpy.median(lmap[lmap > 0])
                min_img = numpy.percentile(img[img > 0], 1)
                max_img = numpy.percentile(img[img > 0], 97)
                cb = ax.imshow(
                    img,
                    origin="lower",
                    cmap=cmap,
                    norm=Normalize(vmin=min_img, vmax=max_img),
                    zorder=0,
                )
                ax.annotate(
                    label,
                    xy=(0.98, 0.98),
                    ha="right",
                    va="top",
                    xycoords="axes fraction",
                    color="black",
                )

            def add_lensing_map_contours(
                ax, lensing_img, lensing_wcs, label="", **kwargs
            ):
                levels = [
                    numpy.power(10, 8.3),
                    numpy.power(10, 8.5),
                    numpy.power(10, 8.7),
                    numpy.power(10, 8.9),
                    numpy.power(10, 9.1),
                ]
                colors = plt.get_cmap("magma")(numpy.linspace(0.2, 1, len(levels)))
                contours = ax.contour(
                    lensing_img.T,
                    levels=levels,
                    colors=colors,
                    transform=ax.get_transform(lensing_wcs),
                )
                # ax.annotate(label, xy = (0.98, 0.02), ha = "right", va = "bottom", xycoords = "axes fraction", color = "black")

            fig = plt.figure(figsize=(8, 6.5))
            left = 0.1
            right = 0.87
            top = 0.98
            bottom = 0.1
            hspace = 0.0
            wspace = 0.05
            axs = []

            # lensing map
            ax = fig.add_subplot(111, projection=noise_wcs)
            add_panel_noise_map(
                ax, noise_img, label=r"$\log_{10}(\sigma_{\rm sky})$", lim_cbar=[-4, 6]
            )
            # add_panel_stellar_light_map(ax, noise_img, label = "Stellar light", cmap = "Greys")
            axs.append(ax)

            # add the contours of the lambda map
            add_lensing_map_contours(
                ax, lensing_img, lensing_wcs, label="Price24", **kwargs
            )

            # find the limits of the image for the bright GCs sample in (RA, DEC)
            p24_xlim_ra = numpy.asarray(
                lensing_wcs.all_pix2world([0, lensing_img.shape[0]], [0, 0], 0)
            )[0]
            p24_ylim_dec = numpy.asarray(
                lensing_wcs.all_pix2world([0, 0], [0, lensing_img.shape[1]], 0)
            )[1]

            # format all axes
            for j, ax in zip(range(10), axs):
                ra = ax.coords[0]
                dec = ax.coords[1]
                if j % 3 == 0:
                    dec.set_axislabel("Declination (J2000)")
                else:
                    dec.set_ticks_visible(True)
                    dec.set_ticklabel_visible(False)
                    dec.set_axislabel("")

                ra.set_axislabel("Right Ascension (J2000)")

                # set the formatting of the axes
                ra.set_major_formatter("hh:mm:ss.s")
                dec.set_major_formatter("dd:mm")
                # display minor ticks
                ra.display_minor_ticks(True)
                dec.display_minor_ticks(True)
                # ra.set_minor_frequency(10)
                # dec.set_minor_frequency(12)
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

                # for each sample, covert the edges in (RA,DEC) to pixels and apply those limits to the panel
                lim_pix = noise_wcs.all_world2pix(p24_xlim_ra, p24_ylim_dec, 0)
                ax.set_xlim(lim_pix[0])
                ax.set_ylim(lim_pix[1])

                ax.set_aspect("equal", adjustable="datalim")
            # format the entire figure
            fig.subplots_adjust(
                left=left,
                top=top,
                bottom=bottom,
                right=right,
                hspace=hspace,
                wspace=wspace,
            )
            fig.savefig(
                os.path.join("./", "xy_noisemap_price24.pdf"), bbox_inches="tight"
            )
            plt.show()

        # load the FITS image and WCS object
        with fits.open(out_fname, output_verify="fix") as _fits_table:
            header = _fits_table[0].header
            _img = _fits_table[
                0
            ].data.T  # transpose the image, it is read as (rows, columns) otherwise
            _wcs = wcs.WCS(header)
            print(_img.shape)
            print(_wcs)
            for key in header.keys():
                print(f"{key}: {header[key]}")
        print(_img.min(), _img.max())

        from matplotlib.colors import Normalize
        from master_class_galaxy_cluster import GalaxyCluster
        from master_class_lambdamaps import (
            LensingMap,
            StellarLightMap,
        )

        p24_lambda = LensingMap("Price24", "lensing map")

        # create the instance of the Galaxy Cluster class for Abell 2744
        abell2744 = GalaxyCluster(
            "Abell2744",
            distance=1630 * u.Mpc,
            redshift=0.308,
            arcsec_to_kpc=2100 * u.kpc / (460 * u.arcsec),
        )
        # convert all the convergence maps to projected mass maps
        p24_lambda.convert_to_projected_mass(abell2744)
        figure_noise_map_with_lambda_map_contours(
            _img, _wcs, p24_lambda.img_mass.value, p24_lambda.wcs
        )

        mosaic_avg = StellarLightMap("Original", "stellar light")

        # figure_noise_map_with_lambda_map_contours(mosaic_avg.img.value, mosaic_avg.wcs, p24_lambda.img_mass.value, p24_lambda.wcs)
    return


@app.cell
def _(mo):
    mo.md(r"""## Functions""")
    return


@app.cell
def _(Table, fits, numpy):
    def create_fits_image(table: Table, img: numpy.ndarray, fname: str) -> None:
        """Create a FITS image for a given table and image."""

        # extract the number of pixels in the image
        num_pixels = img.T.shape
        # determine the conversion of degree per pixel
        middec = 0.5 * (
            table["dec (deg)"].to("rad").min().value
            + table["dec (deg)"].to("rad").max().value
        )
        deg_per_pixel = (
            (table["ra (deg)"].to("deg").max() - table["ra (deg)"].to("deg").min())
            * numpy.cos(middec)
            / num_pixels[1],
            (table["dec (deg)"].to("deg").max() - table["dec (deg)"].to("deg").min())
            / num_pixels[0],
        )

        # use the center of the image as the reference point in the header
        mid_pixels = numpy.asarray(
            [int(0.5 * (num_pixels[1] - 1)), int(0.5 * (num_pixels[0] - 1))]
        )
        # convert to pixels in the original image, assuming 10 pixels per unit in the table
        orig_pixels = mid_pixels * 10 + numpy.asarray(
            [table["x (px)"].min(), table["y (px)"].min()]
        )
        mask = (table["x (px)"] == orig_pixels[0]) * (table["y (px)"] == orig_pixels[1])
        coords_ref = (table["ra (deg)"][mask][0], table["dec (deg)"][mask][0])

        print("Reference coordinates: ", coords_ref)

        # create the WCS and header -- using the (0,0) pixel as the reference with max(RA) and min(DEC)
        hdr = fits.Header()
        hdr["NAXIS1"] = num_pixels[0]
        hdr["NAXIS2"] = num_pixels[1]
        hdr["CRPIX1"] = mid_pixels[0]
        hdr["CRPIX2"] = mid_pixels[1]
        hdr["CD1_1"] = -deg_per_pixel[
            0
        ].value  # it includes scaling and rotation information
        hdr["CD2_2"] = deg_per_pixel[1].value
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CDELT1"] = 1.0
        hdr["CDELT2"] = 1.0
        hdr["CUNIT1"] = deg_per_pixel[0].unit.to_string()
        hdr["CUNIT2"] = deg_per_pixel[1].unit.to_string()
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = coords_ref[0]
        hdr["CRVAL2"] = coords_ref[1]
        hdr["LONPOLE"] = 180.0
        hdr["LATPOLE"] = table["dec (deg)"].to("deg").min().value
        hdr["MJDREF"] = 0.0
        hdr["RADESYS"] = "ICRS"
        hdr["EQUINOX"] = 2000.0

        for key in hdr.keys():
            print(f"{key}: {hdr[key]}")

        # create the FITS image and save it
        primary_hdu = fits.PrimaryHDU(data=img.T, header=hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(fname, overwrite=True)
        print("Saved {:s}".format(fname))

    return (create_fits_image,)


@app.cell
def _():
    import marimo as mo

    # Import modules
    import numpy, os, time, copy, glob
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from astropy import wcs
    from astropy import units as u
    from astropy.table import Table
    from astropy.io import ascii, fits
    from astropy.coordinates import SkyCoord
    from matplotlib.colors import LogNorm

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 18.0
    mpl.rcParams["legend.fontsize"] = 16.0
    return SkyCoord, Table, ascii, fits, mo, numpy, os, plt, u, wcs


if __name__ == "__main__":
    app.run()
