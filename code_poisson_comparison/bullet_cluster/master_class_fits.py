### Script containing the definition of the master classes for the different maps
### Author: Marta Reina-Campos
### Date: October 2025

# Import modules
import numpy
from astropy.io import fits
from astropy import wcs
from astropy import units as u


class FitsMap:
    """This class is used to instantiate a basic FITS image into an object.

    :param name: name of map
    :param label: label of map - used in figures
    :param kind: type of map (lensing / ICL / X-ray / noisy distribution / uniform distribution)
    """

    def __init__(
        self,
        fname: str,
    ) -> None:
        # load the FITS image and its header and WCS
        self.wcs, self.header, self.img = self.load_map(fname)

    def load_map(self, fname: str) -> tuple[wcs.WCS, fits.Header, numpy.ndarray]:
        """Return the map, its header and WCS."""

        # open the header and the image
        with fits.open(fname, output_verify="fix") as fits_table:
            header = fits_table[0].header
            img = fits_table[
                0
            ].data.T  # transpose the image, it is read as (rows, columns) otherwise

        # assume that the image is dimensionless
        unit = u.dimensionless_unscaled
        img = img * unit
        img = img.to(unit)
        return wcs.WCS(header), header, img
