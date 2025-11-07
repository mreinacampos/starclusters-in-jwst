import marimo

__generated_with = "0.13.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Comparing the GCs in Abell2744 to its lensing map
    
        * Photometric catalogue of GCs from Harris & Reina-Campos 2023
        * Lensing map from the UNCOVER team - Furtak+ 2023
        """
    )
    return


@app.cell
def _():
    # Import modules
    import sys, numpy, os, glob, scipy, pandas
    from scipy import interpolate
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import ScalarFormatter
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy import units as u
    from astropy import constants
    from astropy.cosmology import Planck18, z_at_value
    from astropy.table import Table

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.size"] = 18.0
    mpl.rcParams["legend.fontsize"] = 16.0
    mpl.rcParams["xtick.major.size"] = 10
    mpl.rcParams["xtick.minor.size"] = 5
    mpl.rcParams["ytick.major.size"] = 10
    mpl.rcParams["ytick.minor.size"] = 5
    return Table, WCS, fits, glob, mpl, numpy, os, plt, u


@app.cell
def _(mo):
    mo.md(r"## Load the photometric catalogue of GCs")
    return


@app.cell
def _(Table, glob, numpy, os, u):
    out_path = os.path.join(".", "imgs")
    distance_to_a2744 = 1630 * u.Mpc
    _inpath = os.path.join("..", "A2744_Harris23_GCs")
    _ls_files = glob.glob(
        os.path.join(_inpath, "2404_00_catalogue_GCs_A2744_originalmosaic_psky*")
    )
    gc_catalogue = Table.read(
        _ls_files[0],
        format="ascii",
        names=(
            "RA [J2000]",
            "DEC [J2000]",
            "x[orig px]",
            "y[orig px]",
            "prob",
            "F115W",
            "F150W",
            "F200W",
            "sky",
            "sigsky",
        ),
        units=(u.deg, u.deg, "", "", "", "", "", "", "", ""),
    )
    DM = 41.06
    K_F115 = 0.17
    K_F150 = 0.17
    K_F200 = 0.42
    image_angular_size = 92
    gc_catalogue["x[orig kpc]"] = gc_catalogue["x[orig px]"] * image_angular_size / 1000
    gc_catalogue["y[orig kpc]"] = gc_catalogue["y[orig px]"] * image_angular_size / 1000
    gc_catalogue["F115W0"] = gc_catalogue["F115W"] + K_F115
    gc_catalogue["F150W0"] = gc_catalogue["F150W"] + K_F150
    gc_catalogue["F200W0"] = gc_catalogue["F200W"] + K_F200
    gc_catalogue["M_F115W0"] = gc_catalogue["F115W"] + K_F115 - DM
    gc_catalogue["M_F150W0"] = gc_catalogue["F150W"] + K_F150 - DM
    gc_catalogue["M_F200W0"] = gc_catalogue["F200W"] + K_F200 - DM
    gc_catalogue["Zone"] = 0
    mask = numpy.log10(gc_catalogue["sigsky"]) <= 1.8
    gc_catalogue["Zone"][mask] = 1
    mask = (numpy.log10(gc_catalogue["sigsky"]) > 1.8) * (
        numpy.log10(gc_catalogue["sigsky"]) <= 2.1
    )
    gc_catalogue["Zone"][mask] = 2
    mask = (numpy.log10(gc_catalogue["sigsky"]) > 2.1) * (
        numpy.log10(gc_catalogue["sigsky"]) <= 2.4
    )
    gc_catalogue["Zone"][mask] = 3
    mask = numpy.log10(gc_catalogue["sigsky"]) > 2.4
    gc_catalogue["Zone"][mask] = 4
    cat_zone1 = gc_catalogue[gc_catalogue["Zone"] == 1].copy()
    arcsec_kpc = 2100 * u.kpc / (460 * u.arcsec)
    return gc_catalogue, out_path


@app.cell
def _(mo):
    mo.md(
        r"### Figure: spatial distribution of GCs in Abell2744 colourcoded by the recovery fraction"
    )
    return


@app.cell
def _(gc_catalogue, mpl, numpy, os, out_path, plt):
    _fig, _ax = plt.subplots(1, figsize=(8, 6.5), sharex=True)
    _ax = numpy.atleast_1d(_ax)
    _ax = _ax.ravel()
    _left = 0.1
    _right = 0.87
    _top = 0.98
    _bottom = 0.1
    _hspace = 0.0
    _wspace = 0.0
    _cmap = plt.get_cmap("viridis")
    _norm = mpl.colors.Normalize(0, 1)
    _ax[0].scatter(
        gc_catalogue["RA [J2000]"],
        gc_catalogue["DEC [J2000]"],
        c=gc_catalogue["prob"],
        cmap=_cmap,
        norm=_norm,
        s=5,
        marker=".",
        edgecolor="k",
        linewidth=0.1,
        zorder=10,
    )
    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _cax = _ax[0].inset_axes([1.001, 0.0, 0.02, 1.0])
    _cbar = _fig.colorbar(_sm, cax=_cax, ax=_ax[0])
    _cbar.minorticks_on()
    _cbar.set_label("Probability of recovery")
    for ind in range(len(_ax)):
        _ax[ind].invert_xaxis()
        _ax[ind].set_xlabel("RA [J2000]")
        _ax[ind].set_ylabel("DEC [J2000]")
        _ax[ind].tick_params(
            bottom=True, left=True, right=True, top=True, axis="both", which="both"
        )
        _ax[ind].minorticks_on()
    _fig.subplots_adjust(
        left=_left,
        top=_top,
        bottom=_bottom,
        right=_right,
        hspace=_hspace,
        wspace=_wspace,
    )
    _fname = "xy_allgcs_recovery_fraction.png"
    _fig.savefig(os.path.join(out_path, _fname), bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"## Load the convergence maps")
    return


@app.cell
def _(mo):
    mo.md(r"### Furtak+2023")
    return


@app.cell
def _(WCS, fits, glob, mpl, numpy, os, out_path, plt):
    _inpath = os.path.join(
        "..", "A2744_LensingMap_Furtak23", "Best-model_low-resolution_100mas"
    )
    _ls_files = glob.glob(os.path.join(_inpath, "*-magnif.fits"))
    _ls_files = sorted(
        _ls_files, key=lambda x: int(x.split("/")[-1].split("_")[-1].split("-")[0][1:])
    )
    _left = 0.1
    _right = 0.87
    _top = 0.98
    _bottom = 0.1
    _hspace = 0.0
    _wspace = 0.0
    _cmap = plt.get_cmap("viridis")
    _norm = mpl.colors.LogNorm(vmin=1, vmax=25)
    for _i, _fname in enumerate(_ls_files):
        try:
            _hdul = fits.open(_fname, output_verify="fix")
        except Exception as e:
            print(_fname, e)
            continue
        print(_fname.split("/")[-1], len(_hdul))
        _wcs = WCS(_hdul[0].header)
        sky = _wcs.pixel_to_world(30, 40)
        print(sky)
        if _i == 0:
            _fig, _axs = plt.subplots(
                2,
                4,
                figsize=(24, 10),
                sharex=True,
                sharey=True,
                subplot_kw={"projection": _wcs},
            )
            _axs = numpy.atleast_1d(_axs)
            _axs = _axs.ravel()
        _axs[_i].imshow(_hdul[0].data, norm=_norm, cmap="viridis", origin="lower")
        zs = int(_fname.split("/")[-1].split("_")[-1].split("-")[0][1:])
        _axs[_i].annotate(
            "$z_{{\\rm s}} = {:d}$".format(zs),
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="white",
        )
        _hdul.close()
    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _cax = _axs[-1].inset_axes([1.001, 0.0, 0.02, 2.0])
    _cbar = _fig.colorbar(_sm, cax=_cax, ax=_axs[0])
    _cbar.minorticks_on()
    _cbar.set_label("Magnification factor")
    for _j, _ax in enumerate(_axs):
        _ra = _ax.coords[0]
        _dec = _ax.coords[1]
        if _j % 4 == 0:
            _dec.set_axislabel("Declination (J2000)")
        else:
            _dec.set_ticks_visible(False)
            _dec.set_ticklabel_visible(False)
            _dec.set_axislabel("")
        if _j >= 4:
            _ra.set_axislabel("Right Ascension (J2000)")
        else:
            _ra.set_ticks_visible(False)
            _ra.set_ticklabel_visible(False)
            _ra.set_axislabel("")
        _ra.set_major_formatter("hh:mm:ss.s")
        _dec.set_major_formatter("dd:mm")
        _ra.display_minor_ticks(True)
        _dec.display_minor_ticks(True)
        _ra.set_minor_frequency(10)
        _dec.set_minor_frequency(10)
    _fig.subplots_adjust(
        left=_left,
        top=_top,
        bottom=_bottom,
        right=_right,
        hspace=_hspace,
        wspace=_wspace,
    )
    _fname = "xy_furtak23_lensing_allzs.png"
    _fig.savefig(os.path.join(out_path, _fname), bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"## Convergence map from Furtak+ (2023)")
    return


@app.cell
def _(WCS, fits, glob, mpl, numpy, os, out_path, plt):
    _inpath = os.path.join(
        os.curdir, "A2744_LensingMap_Furtak23", "Best-model_low-resolution_100mas"
    )
    _ls_files = glob.glob(os.path.join(_inpath, "*_kappa.fits"))
    _left = 0.1
    _right = 0.87
    _top = 0.98
    _bottom = 0.1
    _hspace = 0.0
    _wspace = 0.0
    _cmap = plt.get_cmap("viridis")
    _norm = mpl.colors.LogNorm(vmin=0.01, vmax=25)
    for _i, _fname in enumerate(_ls_files):
        try:
            _hdul = fits.open(_fname, output_verify="fix")
        except Exception as e:
            print(_fname, e)
            continue
        _wcs = WCS(_hdul[0].header)
        if _i == 0:
            _fig, _axs = plt.subplots(
                1,
                figsize=(10, 6.5),
                sharex=True,
                sharey=True,
                subplot_kw={"projection": _wcs},
            )
            _axs = numpy.atleast_1d(_axs)
            _axs = _axs.ravel()
        print(
            _fname.split("/")[-1],
            _hdul[0].data.min(),
            _hdul[0].data.max(),
            numpy.sum(numpy.isnan(_hdul[0].data)),
        )
        _axs[_i].imshow(_hdul[0].data, norm=_norm, cmap="viridis", origin="lower")
        _hdul.close()
    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _cax = _axs[-1].inset_axes([1.001, 0.0, 0.02, 1.0])
    _cbar = _fig.colorbar(_sm, cax=_cax, ax=_axs[0])
    _cbar.minorticks_on()
    _cbar.set_label("$\\kappa$")
    for _j, _ax in enumerate(_axs):
        _ra = _ax.coords[0]
        _dec = _ax.coords[1]
        if _j % 4 == 0:
            _dec.set_axislabel("Declination (J2000)")
        else:
            _dec.set_ticks_visible(False)
            _dec.set_ticklabel_visible(False)
            _dec.set_axislabel("")
        _ra.set_axislabel("Right Ascension (J2000)")
        _ra.set_major_formatter("hh:mm:ss.s")
        _dec.set_major_formatter("dd:mm")
        _ra.display_minor_ticks(True)
        _dec.display_minor_ticks(True)
        _ra.set_minor_frequency(10)
        _dec.set_minor_frequency(10)
    _fig.subplots_adjust(
        left=_left,
        top=_top,
        bottom=_bottom,
        right=_right,
        hspace=_hspace,
        wspace=_wspace,
    )
    _fname = "xy_furtak23_kappa.png"
    _fig.savefig(os.path.join(out_path, _fname), bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(WCS, fits, glob, mpl, numpy, os, out_path, plt):
    _inpath = os.path.join(
        os.curdir, "LensingMap_Furtak23", "Best-model_low-resolution_100mas"
    )
    _ls_files = glob.glob(os.path.join(_inpath, "*.fits"))
    ls_files_selected = [
        x
        for x in _ls_files
        if "kappa" in x or "psi" in x or "gamma" in x or ("_magnif" in x)
    ]
    _left = 0.1
    _right = 0.87
    _top = 0.98
    _bottom = 0.1
    _hspace = 0.0
    _wspace = 0.3
    _cmap = plt.get_cmap("viridis")
    _norm = mpl.colors.LogNorm(vmin=0.001, vmax=1)
    for _i, _fname in enumerate(ls_files_selected):
        try:
            _hdul = fits.open(_fname, output_verify="fix")
        except Exception as e:
            print(_fname, e)
            continue
        _wcs = WCS(_hdul[0].header)
        if _i == 0:
            _fig, _axs = plt.subplots(
                1,
                4,
                figsize=(24, 6.5),
                sharex=True,
                sharey=True,
                subplot_kw={"projection": _wcs},
            )
            _axs = numpy.atleast_1d(_axs)
            _axs = _axs.ravel()
        print(
            _fname.split("/")[-1],
            _hdul[0].data.min(),
            _hdul[0].data.max(),
            numpy.sum(numpy.isnan(_hdul[0].data)),
        )
        if numpy.sum(numpy.isnan(_hdul[0].data)):
            _hdul[0].data[numpy.isnan(_hdul[0].data)] = 1e-11
        print(
            _fname.split("/")[-1],
            _hdul[0].data.min(),
            _hdul[0].data.max(),
            numpy.sum(numpy.isnan(_hdul[0].data)),
        )
        _axs[_i].imshow(
            abs(_hdul[0].data / _hdul[0].data.max()),
            norm=_norm,
            cmap="viridis",
            origin="lower",
        )
        _hdul.close()
        label = _fname.split("/")[-1].split("_")[-1].split(".")[0]
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
        _cax = _axs[_i].inset_axes([1.001, 0.0, 0.02, 1.0])
        _cbar = _fig.colorbar(_sm, cax=_cax, ax=_axs[_i])
        _cbar.minorticks_on()
        _cbar.set_label("|{:s}/{:s}$_{{\\rm max}}$|".format(label, label))
    for _j, _ax in enumerate(_axs):
        _ra = _ax.coords[0]
        _dec = _ax.coords[1]
        if _j % 4 == 0:
            _dec.set_axislabel("Declination (J2000)")
        else:
            _dec.set_ticks_visible(False)
            _dec.set_ticklabel_visible(False)
            _dec.set_axislabel("")
        _ra.set_axislabel("Right Ascension (J2000)")
        _ra.set_major_formatter("hh:mm:ss.s")
        _dec.set_major_formatter("dd:mm")
        _ra.display_minor_ticks(True)
        _dec.display_minor_ticks(True)
        _ra.set_minor_frequency(10)
        _dec.set_minor_frequency(10)
    _fig.subplots_adjust(
        left=_left,
        top=_top,
        bottom=_bottom,
        right=_right,
        hspace=_hspace,
        wspace=_wspace,
    )
    _fname = "xy_furtak23_gamma_psi_kappa_magnif.png"
    _fig.savefig(os.path.join(out_path, _fname), bbox_inches="tight")
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
