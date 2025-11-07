### Script containing the basic routines for reading the GC catalogue of Abell 2744
### Author: Marta Reina-Campos
### Date: October 2025

# Import modules
import numpy, os, glob
from astropy import units as u
from astropy.table import Table
from master_class_galaxy_cluster import GalaxyCluster
from astropy.io import fits
from astropy import wcs, constants
import scipy.integrate


class GCLoaders:
    """Mixin providing loader routines for different map types.
    Put any loading / creation routine here so users can edit them in one place.
    """

    def load_gc_luminosity_function_parameters(self) -> tuple[float, float]:
        """Function to load the parameters of the GCLF for Abell 2744 from Harris et al. (2023)"""
        # peak magnitude and dispersion of the GCLF in the F150W filter
        m0 = 31.76 * u.ABmag  # peak magnitude
        sigma = 1.4 * u.ABmag  # dispersion of the GCLF
        return m0, sigma

    def load_gc_catalogue(self) -> Table:
        """Function to load the GC catalogue for Abell 2744, and apply some preliminary colour cuts"""
        # input path of the photometric catalogue
        inpath = os.path.join(".", "data", "GCs_Harris23")
        ls_files = glob.glob(
            os.path.join(inpath, "2404_00_catalogue_GCs_A2744_originalmosaic_psky*")
        )

        # read the photometric catalogue -- frame: fk5
        gc_catalogue = Table.read(
            ls_files[0],
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
            units=(
                u.deg,
                u.deg,
                u.pixel,
                u.pixel,
                u.dimensionless_unscaled,
                u.ABmag,
                u.ABmag,
                u.ABmag,
                "",
                u.dimensionless_unscaled,
            ),
        )
        # distance modulus
        DM = 41.06
        # K-corrections in the 3 bands for a redshift of 0.308
        K_F115 = 0.17
        K_F150 = 0.17
        K_F200 = 0.42
        # coordinates in physical units
        image_angular_size = 92 * u.parsec / u.pixel  # pc per pixel
        gc_catalogue["x[orig kpc]"] = (
            gc_catalogue["x[orig px]"] * image_angular_size
        ).to(u.kpc)
        gc_catalogue["y[orig kpc]"] = (
            gc_catalogue["y[orig px]"] * image_angular_size
        ).to(u.kpc)
        # K-corrected apparent magnitudes
        gc_catalogue["F115W0"] = gc_catalogue["F115W"] + K_F115
        gc_catalogue["F150W0"] = gc_catalogue["F150W"] + K_F150
        gc_catalogue["F200W0"] = gc_catalogue["F200W"] + K_F200

        # K-corrected absolute magnitudes
        gc_catalogue["M_F115W0"] = gc_catalogue["F115W"] + K_F115 - DM
        gc_catalogue["M_F150W0"] = gc_catalogue["F150W"] + K_F150 - DM
        gc_catalogue["M_F200W0"] = gc_catalogue["F200W"] + K_F200 - DM

        # determine the four sky zones
        gc_catalogue["Zone"] = 0
        gc_catalogue["Zone"][numpy.log10(gc_catalogue["sigsky"]) <= 1.8] = 1
        gc_catalogue["Zone"][
            (numpy.log10(gc_catalogue["sigsky"]) > 1.8)
            * (numpy.log10(gc_catalogue["sigsky"]) <= 2.1)
        ] = 2
        gc_catalogue["Zone"][
            (numpy.log10(gc_catalogue["sigsky"]) > 2.1)
            * (numpy.log10(gc_catalogue["sigsky"]) <= 2.4)
        ] = 3
        gc_catalogue["Zone"][(numpy.log10(gc_catalogue["sigsky"]) > 2.4)] = 4

        # log10 of the local sky noise
        gc_catalogue["log10sigsky"] = numpy.log10(gc_catalogue["sigsky"])

        # mask objects with colours outside of the range -1.2 < (F115W0-F150W0) < 1.2
        mask = (gc_catalogue["F115W0"] - gc_catalogue["F150W0"] > -1.2) & (
            gc_catalogue["F115W0"] - gc_catalogue["F150W0"] < 1.2
        )

        return gc_catalogue[mask]

    def create_mask_for_gc_sample(
        self, gcs_name: str, gc_catalogue: Table
    ) -> numpy.ndarray:
        """Create the mask for the GCs catalogue based on the sample of GCs"""
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
        elif gcs_name == "High-quality GCs":
            mask = (gc_catalogue["F150W"].to(u.ABmag) < 29.5 * u.ABmag) * (
                gc_catalogue["Zone"] < 3
            )
        return mask

    def probability_of_recovery(self, f150w: numpy.ndarray, log10_sigma_sky: numpy.ndarray):
        """
        Analytic function describing the probability of recovery based on the magnitude and local sky noise of a given GC.
        Using eq (1) in Harris & Reina-Campos 2024.

        Input: 
        :param f150w : numpy.ndarray
            1D array of apparent magnitudes in the F150W filter (astropy Quantity with units of ABmag).
        :param log10_sigma_sky : numpy.ndarray
            1D array of log10 of the local sky noise (dimensionless).

        Output:
        numpy.ndarray
            Probability of recovery S(f150w, log10_sigma_sky) for each input value.

        Notes:
        b0 is not given in the paper, so I calculated:
        b0 = -numpy.log((1/bright_gcs.prob - 1) * numpy.exp(b1 * bright_gcs.f150w.value + b2 * bright_gcs.log10sigsky))
        """
        b0 = 85.84
        b1 = -2.59
        b2 = -5.37
        g = b0 + b1 * f150w.value + b2 * log10_sigma_sky

        return 1 / (1 + numpy.exp(-g))

    def integrate_probability_of_recovery(self, f150w_min, f150w_max, log10_sigma_sky):
        """
        Analytical integration of the probability_of_recovery function over the specified ranges in F150W. 
        The local sky noise at a given pixel is given by the image of the local sky noise.

        Input:
        :param f150w_min : float
            Lower bound of f150w (ABmag).
        :param f150w_max : float
            Upper bound of f150w (ABmag).
        :param log10_sigma_sky_min : float
            Lower bound of log10_sigma_sky.
        :param log10_sigma_sky_max : float
            Upper bound of log10_sigma_sky.

        Output:
        float
            integral value.
        """
        b0 = 85.84
        b1 = -2.59
        b2 = -5.37
        # evaluate the g function at the integration limits
        g_min = b0 + b1 * f150w_min.value + b2 * log10_sigma_sky
        g_max = b0 + b1 * f150w_max.value + b2 * log10_sigma_sky

        result = (1/b1) * numpy.log(numpy.exp(g_max)+1) - (1/b1) * numpy.log(numpy.exp(g_min)+1)
        return result

class LambdaMapLoaders:
    """Mixin providing loader routines for different map types.
    Put any loading / creation routine here so users can edit them in one place.
    """
    def load_lensing_model(self, name: str):
        # moved from LensingMap.load_lensing_model
        if name == "Cha24_WL":
            fname = os.path.join(
                ".",
                "data",
                "LensingMap_Cha24",
                "A2744_convergence_map_WL.fits",
            )
        elif name == "Cha24_SL_WL":
            fname = os.path.join(
                ".",
                "data",
                "LensingMap_Cha24",
                "A2744_convergence_map_SL_WL.fits",
            )
        elif name == "Price24":
            fname = glob.glob(
                os.path.join(
                    ".",
                    "data",
                    "LensingMap_Price24",
                    "Best-model_low-resolution_100mas",
                    "*_kappa.fits",
                )
            )[0]
        elif name == "Bergamini23":
            fname = os.path.join(
                ".", "data", "LensingMap_Bergamini23", "kappa_B23_DlsDs1.fits"
            )
        else:
            raise ValueError(f"Unknown lensing model name: {name}")

        with fits.open(fname, output_verify="fix") as fits_table:
            header = fits_table[0].header
            img = fits_table[0].data.T

        unit = u.dimensionless_unscaled
        return wcs.WCS(header), header, img * unit

    # converts the convergence map as projected mass surface density and their header and WCS for the chosen lensing model
    def convert_to_projected_mass(self, galaxy_cluster: GalaxyCluster) -> None:
        """Convert the converenge map to a projected mass surface density.
        Sigma is calculated as $$\Sigma(\vec{x}) =  \kappa(\vec{x})\Sigma_{\rm cr} $$
        where $\kappa$ is the convergence map and $\Sigma_{\rm cr}$ is the critical surface density for lensing.
        """

        # determine the critical surface density
        if self.name in ["Furtak23", "Price24", "Bergamini23"]:
            sigma_cr = (constants.c.to("kpc/s") ** 2) / (
                4
                * numpy.pi
                * constants.G.to("kpc3 / (Msun s2)")
                * galaxy_cluster.distance.to("kpc")
            )
        elif (
            "Cha24" in self.name
        ):  # convergence maps from Cha24 - only weak lensing constraints
            sigma_cr = (
                1.777e9 * u.solMass / u.kpc**2
            )  # solar mass per square kpc - critical surface density
        # convert to projected mass surface density
        self.img_mass = self.img * sigma_cr  # units: MSun / kpc^2

    def load_stellar_light_map(self, kind: str):
        if "light" in kind:
            fname = os.path.join(
                ".",
                "data",
                "StellarLight_UNCOVER_CombinedMosaics",
                "abell2744clu-grizli-v7.2-avg-clear_drc_sci_reduced_rebinned.fits",
            )
        elif "less" in kind:
            fname = os.path.join(
                ".",
                "data",
                "StellarLight_UNCOVER_CombinedMosaics",
                "uncover_abell2744clu_avg_bkg_sci_reduced_rebinned.fits",
            )
        else:
            raise ValueError(f"Unknown stellar-light kind: {kind}")

        with fits.open(fname, output_verify="fix") as fits_table:
            header = fits_table[0].header
            img = fits_table[0].data.T

        unit = 10 * u.nanoJansky
        img = img * unit
        img = img.to(unit)
        return wcs.WCS(header), header, img

    def load_xray_map(self):
        fname = os.path.join(
            ".",
            "data",
            "Xray_Chandra_ID8477",
            "primary",
            "acisf08477N003_cntr_img2.fits",
        )
        with fits.open(fname, output_verify="fix") as fits_table:
            header = fits_table[0].header
            img = fits_table[0].data.T + 1e-10

        unit = u.dimensionless_unscaled
        return wcs.WCS(header), header, img * unit
