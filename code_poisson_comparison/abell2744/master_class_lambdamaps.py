### Script containing the definition of the master classes for the different maps
### Author: Marta Reina-Campos
### Date: October 2025

# Import modules
import numpy, os, glob
from astropy.io import fits
from astropy import wcs, constants
from astropy import units as u
from master_class_galaxy_cluster import GalaxyCluster
from master_class_gcs import GCs
from master_functions_abell2744 import MapLoaders

class LambdaMap(MapLoaders):
    """This top-level class is used to define the routines that LambdaMaps need to have.

    :param name: name of map
    :param label: label of map - used in figures
    :param kind: type of map (lensing / ICL / X-ray / noisy distribution / uniform distribution)
    """

    type_class = "lambda map"  # information about the kind of object we're studying

    def __init__(
        self, name: str, kind: str, **kwargs
    ) -> None:  #  gcs : GCs, galaxy_cluster : GalaxyCluster
        self.name = name
        self.label = name
        self.kind = kind

    def extend_dimensions_to_higher_dimensionality(
        self, do_dimension: int, mask: numpy.ndarray, **kwargs
    ) -> None:
        """Extend the lambda map to the high-dimensionality space -- assuming that the lambda map is constant in the extra dimensions"""

        # apply the footprint of the GCs to the image
        self.img *= mask
        # extend the image to the higher dimensionality
        if do_dimension == 2:
            self.img_extended = self.img
        if do_dimension >= 3:
            self.img_extended = numpy.expand_dims(self.img, axis=2)
            self.img_extended = numpy.repeat(
                self.img_extended, kwargs["shape"][2], axis=2
            )
        if do_dimension == 4:
            self.img_extended = numpy.expand_dims(self.img_extended, axis=3)
            self.img_extended = numpy.repeat(
                self.img_extended, kwargs["shape"][3], axis=3
            )


class LensingMap(LambdaMap):
    """This class is used to instantiate the map against we'll compare the GC sample against.

    :param name: name of map
    :param label: label of map - used in figures
    :param kind: type of map (lensing / ICL / X-ray / noisy distribution / uniform distribution)
    """

    def __init__(
        self, name: str, kind: str, **kwargs
    ) -> None:  #  gcs : GCs, galaxy_cluster : GalaxyCluster
        LambdaMap.__init__(self, name, kind)

        if "lensing".lower() in kind:
            self.wcs, self.header, self.img = self.load_lensing_model(self.name)
        elif "uniform".lower() in kind:
            num_pixels = (600, 600)
            self.img = self.create_uniform_distribution(num_pixels)
            self.wcs, self.header = kwargs["gcs"].create_wcs_and_header(self.img)
        elif "noisy".lower() in kind:
            num_pixels = (600, 600)
            sigma_kpc = (20 * u.kpc, 20 * u.kpc)
            self.img = self.create_noisy_distribution(
                num_pixels, sigma_kpc, kwargs["gcs"], kwargs["galaxy_cluster"]
            )
            self.wcs, self.header = kwargs["gcs"].create_wcs_and_header(self.img)
        else:
            print(
                "[LensingMap/__init__] This type of LensingMap wasn't recognized. Please, check the kind of map you're trying to load."
            )


    def create_uniform_distribution(
        self, num_pix: tuple[int, int]
    ) -> numpy.ndarray:  # tuple[WCS, fits.header, numpy.ndarray]:
        """Return a uniform distribution of a given size.
        :type num_pix: tuple with the number of pixels in the (x, y)-axes
        """
        # create a uniform distribution
        print(
            "[create_uniform_distribution] Creating a uniform distribution of size ({:d}, {:d})".format(
                num_pix[0], num_pix[1]
            )
        )
        unit = u.dimensionless_unscaled  # units: dimensionless
        return 0.5 * numpy.ones((num_pix[0], num_pix[1])) * unit  # units: dimensionless

    @u.quantity_input
    def create_noisy_distribution(
        self,
        num_pix: tuple[int, int],
        sigma: tuple[u.kpc, u.kpc],
        gcs: GCs,
        galaxy_cluster: GalaxyCluster,
    ) -> numpy.ndarray:
        """Return a noisy distributio made of a random number of 2D gaussians randomly placed throughout the image of a given size.
        :type num_pix: tuple with the number of pixels in the (x, y)-axes
        :type sigma: tuple with the standard deviation of the 2D Gaussian in kpc in the (x, y)-axes
        """

        def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
            # define a normalized 2D gaussian
            return (
                1.0
                / (2.0 * numpy.pi * sx * sy)
                * numpy.exp(
                    -(
                        (x - mx) ** 2.0 / (2.0 * sx**2.0)
                        + (y - my) ** 2.0 / (2.0 * sy**2.0)
                    )
                )
            )

        # convert the pixel size to kpc at the distance of the galaxy cluster
        pixel_to_kpc = (
            (gcs.ra.to("arcsec").max() - gcs.ra.to("arcsec").min())
            * galaxy_cluster.arcsec_to_kpc
            / num_pix[0],
            (gcs.dec.to("arcsec").max() - gcs.dec.to("arcsec").min())
            * galaxy_cluster.arcsec_to_kpc
            / num_pix[1],
        )
        sigma_pixels = (sigma[0] / pixel_to_kpc[0], sigma[1] / pixel_to_kpc[1])

        # draw from a uniform distribution the number of subhaloes
        number_subhaloes = int(numpy.floor(numpy.random.uniform(10, 50)))
        print(
            "[create_noisy_distribution] Creating a noisy map with {:d} subhaloes of size ({:.2f}, {:.2f}) or ({:.2f}, {:.2f}) pixels".format(
                number_subhaloes, *sigma, *sigma_pixels
            )
        )
        print(
            "[create_noisy_distribution] The pixel size is {:.2f} in the x-axis and {:.2f} in the y-axis".format(
                *pixel_to_kpc
            )
        )
        # draw from uniform distributions to determine the coordinates of the subhaloes
        coords_subhaloes = numpy.array(
            [
                [x, y]
                for x, y in zip(
                    numpy.random.uniform(0, num_pix[0], number_subhaloes),
                    numpy.random.uniform(0, num_pix[1], number_subhaloes),
                )
            ]
        )
        xx, yy = numpy.meshgrid(
            numpy.arange(0, num_pix[0]), numpy.arange(0, num_pix[1]), indexing="ij"
        )
        # create the map of uniform density first
        img = numpy.ones(shape=(num_pix[0], num_pix[1])) * 1e-10
        # add the 2D Gaussians for the subhaloes
        for ind in range(number_subhaloes):
            img += gaussian_2d(
                xx,
                yy,
                mx=coords_subhaloes[ind, 0],
                my=coords_subhaloes[ind, 1],
                sx=sigma_pixels[0],
                sy=sigma_pixels[1],
            )

        unit = u.dimensionless_unscaled  # units: dimensionless
        return img * unit  # units: dimensionless


class StellarLightMap(LambdaMap):
    """This class is used to instantiate the stellar light map against which we'll compare the GC sample against.

    :param name: name of map
    :param label: label of map - used in figures
    :param kind: type of map (lensing / ICL / X-ray / noisy distribution / uniform distribution)
    """

    def __init__(
        self, name: str, kind: str
    ) -> None:  #  gcs : GCs, galaxy_cluster : GalaxyCluster
        # instantiate the LambdaMap
        LambdaMap.__init__(self, name, kind)

        # load the actual mosaic and their header and WCS
        self.wcs, self.header, self.img = self.load_stellar_light_map(kind)



class XrayMap(LambdaMap):
    """This class is used to instantiate the X-ray map against which we'll compare the GC sample against.

    :param name: name of map
    :param label: label of map - used in figures
    :param kind: type of map (lensing / ICL / X-ray / noisy distribution / uniform distribution)
    """

    def __init__(
        self, name: str, kind: str
    ) -> None:  #  gcs : GCs, galaxy_cluster : GalaxyCluster
        # instantiate the LambdaMap
        LambdaMap.__init__(self, name, kind)

        # load the actual mosaic and their header and WCS
        self.wcs, self.header, self.img = self.load_xray_map()

