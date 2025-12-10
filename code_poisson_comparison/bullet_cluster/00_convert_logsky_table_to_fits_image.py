import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Code to transform a table into a FITS image
    ## Author: Marta Reina-Campos
    ## Date: August 2025

    The code below transforms any column of a given table into a FITS image with the right header and WCS objects.
    """)
    return


@app.cell
def _(
    SkyCoord,
    ascii,
    convert_to_original_image_coords,
    convert_to_table_coordinates,
    create_fits_image,
    fits,
    numpy,
    os,
    plt,
    scipy,
    u,
    wcs,
):
    do_validation_figure = False

    # define the filename and read the table
    fname = os.path.join(".", "data", "GCs_Harris26", "2512_coords_sigmasky_corrected.txt")
    #fname = os.path.join(".", "data", "GCs_Harris26", "old_skymatrix.txt")

    table = ascii.read(
        fname,
        names=["RA", "DEC", "x (px)", "y (px)", "sigsky"],
        #names=["RA", "DEC", "x (px)", "y (px)", "log(sigsky)"],
        guess=False,
        fast_reader=False,
    )
    # calculate the log of sigsky
    table["log(sigsky)"] = numpy.log10(table["sigsky"] + 1e-10)

    # convert to the original image coordinates and compress on the 10-pixel grid
    table["orig img x (px)"], table["orig img y (px)"] = convert_to_original_image_coords(table["x (px)"], table["y (px)"])
    table["compress x (px)"] = (table["orig img x (px)"] / 10).astype(int)
    table["compress y (px)"] = (table["orig img y (px)"] / 10).astype(int)

    # convert back to coordinates in Nick's mosaics
    table["reconv x (px)"], table["reconv y (px)"] = convert_to_table_coordinates(table["compress x (px)"], table["compress y (px)"])
    # re-center in (0,0)
    table["reconv x (px)"] -= table["reconv x (px)"].min()
    table["reconv y (px)"] -= table["reconv y (px)"].min()

    # Create the tilted image 
    #img = numpy.ones(
    #    shape=(int(table["reconv x (px)"].max()) + 1, int(table["reconv y (px)"].max()) + 1)
    #) * (table["log(sigsky)"].max() * 10)
    #img[table["reconv x (px)"], table["reconv y (px)"]] = table["log(sigsky)"]

    # interpolate with the nearest neighbor to fill in the missing gaps
    xmin, xmax = table["reconv x (px)"].min(), table["reconv x (px)"].max()
    ymin, ymax = table["reconv y (px)"].min(), table["reconv y (px)"].max()
    Ygrid, Xgrid = numpy.meshgrid(numpy.linspace(ymin, ymax, int(table["reconv y (px)"].max() + 1)) ,
                                  numpy.linspace(xmin, xmax, int(table["reconv x (px)"].max() + 1)))
    img = scipy.interpolate.griddata(
        points=numpy.column_stack((table["reconv x (px)"], table["reconv y (px)"])),
        values=table["log(sigsky)"],
        xi=(Xgrid, Ygrid),
        method="linear"
    )

    ### VALIDATION -- plot the image
    if do_validation_figure: 
        print("*** First validation figure - without tilting", img.shape)
        fig, ax = plt.subplots(1, figsize=(10, 6))
        cmap = plt.cm.viridis
        cmap.set_over("red")  # set the color for values above the maximum
        cmap.set_under("blue")  # set the color for values above the maximum
        img = numpy.ones(
            shape=(int(table["compress x (px)"].max()) + 1, int(table["compress y (px)"].max()) + 1)
        ) * (table["log(sigsky)"].max() * 10)

        img[table["compress x (px)"], table["compress y (px)"]] = table["log(sigsky)"]
        cb = ax.imshow(img.T, origin="lower", vmin=0.2, vmax=2, cmap=cmap)
        ax.set_ylabel("Image y (px)")
        ax.set_xlabel("Image x (px)")
        # add the colorbar
        cax = ax.inset_axes([0.0, 1.0001, 1.0, 0.02])  # [x0, y0, width, height]
        # create the colorbar object
        cbar = fig.colorbar(cb, cax=cax, ax=ax, orientation="horizontal", location="top", extend = "both")
        cbar.minorticks_on()  # add minorticks
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.set_label(
            r"$\log_{10}(\sigma_{\rm sky})$",
        )  # add label
        plt.show()

    ### VALIDATION -- plot the tilted & interpolated image
    fig, ax = plt.subplots(1, figsize=(10, 6))
    cmap = plt.cm.viridis
    cmap.set_over("red")  # set the color for values above the maximum
    cmap.set_under("blue")  # set the color for values above the maximum
    cb = ax.imshow(img.T, origin="lower", vmin=0.2, vmax=2, cmap=cmap)
    ax.set_ylabel("Image y (px)")
    ax.set_xlabel("Image x (px)")
    # add the colorbar
    cax = ax.inset_axes([0.0, 1.0001, 1.0, 0.02])  # [x0, y0, width, height]
    # create the colorbar object
    cbar = fig.colorbar(cb, cax=cax, ax=ax, orientation="horizontal", location="top", extend = "both")
    cbar.minorticks_on()  # add minorticks
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.set_label(
        r"$\log_{10}(\sigma_{\rm sky})$",
    )  # add label
    plt.show()


    # convert the RA , DEC to astropy coordinate objects
    coords = SkyCoord(ra=table["RA"] * u.deg, dec=table["DEC"] * u.deg, frame="icrs")
    table["ra (deg)"] = coords.ra.to("deg")
    table["dec (deg)"] = coords.dec.to("deg")
    table["ra (arcsec)"] = coords.ra.to("arcsec")
    table["dec (arcsec)"] = coords.dec.to("arcsec")

    # convert it to a FITS file with the appropriate header
    out_fname = os.path.join(".", "data", "GCs_Harris26", "2512_grid_localskynoise.fits")
    create_fits_image(table, img, out_fname)

    ### VALIDATION
    # load the FITS image and WCS object
    with fits.open(out_fname, output_verify="fix") as _fits_table:
        _header = _fits_table[0].header
        _img = _fits_table[0].data.T  # transpose the image, it is read as (rows, columns) otherwise
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

    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Validation against the GC catalogue
    """)
    return


app._unparsable_cell(
    r"""
    ### VALIDATION

    # define the filename and read the table
    _fname = os.path.join(\".\", \"data\", \"GCs_Harris26\", \"2512_bullet_pointsources_corrected.txt\")

    _table = ascii.read(
        _fname,
        names=[\"RA\", \"DEC\", \"x (px)\", \"y (px)\", \"prob\" ,\"F090W\", \"F090W +-\", \"F115W\", \"F115W +-\", \"F150W\", \"F150W +-\", \"F200W\", \"F200W +-\",     \"sh090\", \"sh115\", \"sh150\", \"sh200\", \"lg(skgsky)\"],
        guess=False,
        fast_reader=False,
    )

    # load the FITS image and WCS object
    with fits.open(out_fname, output_verify=\"fix\") as _fits_table:
        _header = _fits_table[0].header
        _img = _fits_table[0].data.T  # transpose the image, it is read as (rows, columns) otherwise
        _wcs = wcs.WCS(_header)
    print(\"*** Second validation figure\")
    _fig, _ax = plt.subplots(1, figsize=(10, 6), subplot_kw={\"projection\": _wcs})
    _cb = _ax.imshow(_img.T, origin=\"lower\", cmap=\"Greys\", vmin=0, vmax=2)
    _coords = SkyCoord(
                ra=_table[\"RA\"] * u.deg,
                dec=_table[\"DEC\"] * u.deg,
                frame=\"fk5\",
            )
    _ax.scatter(_coords.ra, _coords.dec, transform=_ax.get_transform(\"fk5\"), s=1, alpha = _table[\"prob\"])
    # add the colorbar
    _cax = _ax.inset_axes([1.001, 0.0, 0.02, 1.0])  # [x0, y0, width, height]
    # create the colorbar object
    _cbar = _fig.colorbar(_cb, cax=_cax, ax=_ax)
    _cbar.minorticks_on()  # add minorticks
    _cbar.set_label(r\"$\log_{10}(\sigma_{\rm sky})$\")  # add label
    ra = _ax.coords[0]
    dec = _ax.coords[1]
    dec.set_axislabel(\"Declination (J2000)\")4›
    ra.set_axislabel(\"Right Ascension (J2000)\")
    for obj in [ra, dec]:
        # set the formatting of the axes
        obj.set_major_formatter(\"dd:mm:ss\")
        # display minor ticks
        obj.display_minor_ticks(True)
        obj.set_minor_frequency(10)
    plt.show()


    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(r"""
    ## Functions
    """)
    return


@app.cell
def _(numpy):
    # convert the coordinates in Nicks' mosaics into the original image to determine the size of the grid
    # Numbers come from the transformation applied by Bill (units are pixels)
    # rotation of 343.854 and stretch of 1.56441
    # (x', y') are the coordinates in Nick's mosaics (and the ones in the table)
    # (x, y) are the coordinates in the original image
    # x' = 2942.1 + 1.499564 x - 0.4341284 y
    # y' = 3029.4 + 0.4341284 x + 1.499564 y 
    def convert_to_original_image_coords(xp: numpy.ndarray, yp: numpy.ndarray):
        a = 1.499564
        b = -0.4341284
        c = 0.4341284
        d = 1.499564
        x0p = 2942.1
        y0p = 3029.4

        A = numpy.array([[a, b], [c, d]])
        A_inv = numpy.linalg.inv(A)

        original_coords = numpy.dot(A_inv, numpy.array([xp - x0p, yp - y0p]))
        return numpy.round(original_coords[0],0).astype(int), numpy.round(original_coords[1], 0).astype(int)

    def convert_to_table_coordinates(xp: numpy.ndarray, yp: numpy.ndarray):
        a = 1.499564
        b = -0.4341284
        c = 0.4341284
        d = 1.499564
        x0p = 2942.1
        y0p = 3029.4

        x = a * xp + b * yp + x0p
        y = c * xp + d * yp + y0p
        return numpy.round(x,0).astype(int), numpy.round(y, 0).astype(int)
    return convert_to_original_image_coords, convert_to_table_coordinates


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

        # try to use the center of the image as the reference point in the header
        mid_pixels = numpy.asarray(
            [int(0.5 * (num_pixels[1] - 1)), int(0.5 * (num_pixels[0] - 1))]
        )

        def find_ind_nearest(array, value):
            ind = (numpy.abs(array - value)).argmin()
            return array[ind]
        print("[create fits image] Center of image is at", mid_pixels)
        # look for the closest pixel to the center of the image
        inds_x = find_ind_nearest(table["reconv x (px)"], mid_pixels[0])
        inds_y = find_ind_nearest(table["reconv y (px)"][table["reconv x (px)"] == inds_x], mid_pixels[1])
        mid_pixels = numpy.asarray([inds_x, inds_y])
        print("[create fits image] Center of image is at", mid_pixels)

        mask = (table["reconv x (px)"] == mid_pixels[0]) * (table["reconv y (px)"] == mid_pixels[1])
        print(numpy.sum(mask))
        print(numpy.sum((table["reconv x (px)"] == mid_pixels[0])))
        print(numpy.sum((table["reconv y (px)"] == mid_pixels[1])))
        coords_ref = (table["ra (deg)"][mask][0], table["dec (deg)"][mask][0])

        print("[create fits image] Reference coordinates: ", coords_ref)

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
    import numpy, os, time, copy, glob, scipy
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
    return SkyCoord, Table, ascii, fits, mo, numpy, os, plt, scipy, u, wcs


if __name__ == "__main__":
    app.run()
