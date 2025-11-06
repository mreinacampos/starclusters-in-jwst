### Script containing the definition of the DataPoints and GCs classes
### Author: Marta Reina-Campos
### Date: October 2025

# Import modules
import numpy, scipy
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from matplotlib.path import Path
from master_class_galaxy_cluster import GalaxyCluster
from master_functions_abell2744 import MapLoaders
from astropy.coordinates import SkyCoord


class DataPoints(MapLoaders):
    """This class is used to instantiate a sample of data points with RA and DEC.

    :param name: name of the sample
    :param label: label of the sample - used in figures
    :param ra: right ascension of the data points in the J2000 - FK5 reference frame
    :param dec: declination of the data points in the J2000 - FK5 reference frame
    """

    type_class = "data points"  # information about the kind of object we're studying

    def __init__(self, name: str, label: str, ra: u.deg, dec: u.deg, **kwargs) -> None:
        self.name = name
        self.label = label
        self.ra = ra  # right ascension in the J2000 - FK5 reference frame
        self.dec = dec  # declination in the J2000 - FK5 reference frame
        self.kind = "data points"
        # set the header and wcs object if given as kwargs
        if "header" in kwargs.keys():
            self.header = kwargs["header"]
            self.wcs = wcs.WCS(kwargs["header"])

    def define_bins_from_lambda_map(self, map_wcs: wcs.WCS) -> None:
        """Define the bins for the interpolation based on the (x, y) axes of the lambda map."""
        self.intp_bins = (
            numpy.linspace(0, map_wcs.pixel_shape[0] - 1, map_wcs.pixel_shape[0])
            * u.pixel,
            numpy.linspace(0, map_wcs.pixel_shape[1] - 1, map_wcs.pixel_shape[1])
            * u.pixel,
        )
        self.intp_edges = (
            numpy.linspace(
                -0.5, map_wcs.pixel_shape[0] - 0.5, map_wcs.pixel_shape[0] + 1
            )
            * u.pixel,
            numpy.linspace(
                -0.5, map_wcs.pixel_shape[1] - 0.5, map_wcs.pixel_shape[1] + 1
            )
            * u.pixel,
        )


class GCs(MapLoaders):
    """This class is used to instantiate the sample of GCs within the galaxy cluster we're studying.

    :param name: name of the GC sample
    :param label: label of the GC sample - used in figures
    :param galaxy_cluster: GalaxyCluster object corresponding to the galaxy cluster being studied
    :pa
    """

    type_class = (
        "globular clusters"  # information about the kind of object we're studying
    )

    @u.quantity_input
    def __init__(
        self, name: str, label: str, galaxy_cluster: GalaxyCluster, **kwargs
    ) -> None:
        # initialise the relevant parameters
        self.name = name
        self.label = label
        print("[GCs] Initialising the sample of GCs: {:s}".format(self.name))

        if "Dummy" in self.name:
            print("[GCs] Dummy sample selected -- no catalogue will be loaded.")
            self.gc_catalogue = None
            self.mask_catalogue = None
            self.ra = kwargs.get("ra", numpy.array([]) * u.deg)
            self.dec = kwargs.get("dec", numpy.array([]) * u.deg)
            self.f150w = kwargs.get("f150w", numpy.array([]) * u.ABmag)
            self.log10sigsky = kwargs.get(
                "log10sigsky", numpy.array([]) * u.dimensionless_unscaled
            )
            self.prob = kwargs.get("prob", numpy.array([]) * u.dimensionless_unscaled)
            return

        # load the GC catalogue
        self.gc_catalogue = self.load_gc_catalogue()

        # create the mask for the chosen sample of GCs
        self.mask_catalogue = self.create_mask_for_gc_sample(
            self.name, self.gc_catalogue
        )

        # determine the coordinates of the GCs
        coords_gcs = SkyCoord(
            ra=self.gc_catalogue[self.mask_catalogue]["RA [J2000]"].to("deg"),
            dec=self.gc_catalogue[self.mask_catalogue]["DEC [J2000]"].to("deg"),
            frame="fk5",
            distance=galaxy_cluster.distance,
        )
        # initialise the relevant quantities
        self.ra = coords_gcs.ra  # right ascension in the J2000 - FK5 reference frame
        self.dec = coords_gcs.dec  # declination in the J2000 - FK5 reference frame
        self.kind = "globular clusters"
        self.f150w = (
            self.gc_catalogue[self.mask_catalogue]["F150W"] * u.ABmag
        )  # magnitude in the F150W filter
        self.log10sigsky = (
            self.gc_catalogue[self.mask_catalogue]["log10sigsky"]
            * u.dimensionless_unscaled
        )  # log10 of the local sky noise
        self.prob = (
            self.gc_catalogue[self.mask_catalogue]["prob"] * u.dimensionless_unscaled
        )  # probability of recovery for every GC

        # create a header and wcs object for the GCs - dummy pixel size
        self.wcs, self.header = self.create_wcs_and_header(numpy.ones(shape=(400, 900)))

        # initialize the parameters from the luminosity function
        (
            self.luminosity_function_mean_mag,
            self.luminosity_function_sigma_mag,
        ) = self.load_gc_luminosity_function_parameters()

    def create_wcs_and_header(self, img: numpy.ndarray) -> tuple[wcs.WCS, fits.Header]:
        """Return the WCS and header for a given sample of GCs and number of pixels."""

        # extract the number of pixels in the image in the (x, y)-axes
        num_pixels = img.shape  # (dim in x, dim in y)
        # determine the conversion of degree per pixel
        middec = 0.5 * (self.dec.to("rad").min().value + self.dec.to("rad").max().value)
        deg_per_pixel = (
            (self.ra.to("deg").max() - self.ra.to("deg").min())
            * numpy.cos(middec)
            / num_pixels[0],
            (self.dec.to("deg").max() - self.dec.to("deg").min()) / num_pixels[1],
        )
        # create the WCS and header -- using the (0,0) pixel as the reference with max(RA) and min(DEC)
        hdr = fits.Header()
        hdr["NAXIS1"] = num_pixels[0]
        hdr["NAXIS2"] = num_pixels[1]
        hdr["CRPIX1"] = num_pixels[0] / 2 + 0.5
        hdr["CRPIX2"] = num_pixels[1] / 2 + 0.5
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
        hdr["CRVAL1"] = 0.5 * (
            self.ra.to("deg").max().value + self.ra.to("deg").min().value
        )
        hdr["CRVAL2"] = 0.5 * (
            self.dec.to("deg").min().value + self.dec.to("deg").max().value
        )
        hdr["LONPOLE"] = 180.0
        hdr["LATPOLE"] = self.dec.to("deg").min().value
        hdr["MJDREF"] = 0.0
        hdr["RADESYS"] = "FK5"
        hdr["EQUINOX"] = 2000.0
        primary_hdu = fits.PrimaryHDU(data=img.T, header=hdr)
        return wcs.WCS(primary_hdu.header), primary_hdu.header

    def mask_objects_outside_lambda_map(
        self, map_wcs: wcs.WCS, do_add_edges: bool = False
    ) -> None:
        """Mask the GCs that are outside the edges of the lambda map."""
        # convention is such that (0,0) refers to the center of the pixel, not its corner
        coords_edges_lambda_map = (
            map_wcs.all_pix2world(
                [
                    [-0.5, -0.5],
                    [map_wcs.pixel_shape[0] - 0.5, -0.5],
                    [-0.5, map_wcs.pixel_shape[1] - 0.5],
                    [map_wcs.pixel_shape[0] - 0.5, map_wcs.pixel_shape[1] - 0.5],
                ],
                0,
                ra_dec_order=True,
            )
            * u.deg
        )
        ### MASK the GC sample to be within the edges of the lambda map
        # edges of the map in (RA,DEC) space: [[RAmin, RAmax], [DECmin, DECmax]] in deg
        mask = (
            (self.ra.to("deg") >= coords_edges_lambda_map[:, 0].min())
            * (self.ra.to("deg") <= coords_edges_lambda_map[:, 0].max())
            * (
                numpy.abs(self.dec.to("deg"))
                >= numpy.abs(coords_edges_lambda_map[:, 1]).min()
            )
            * (
                numpy.abs(self.dec.to("deg"))
                <= numpy.abs(coords_edges_lambda_map[:, 1]).max()
            )
        )
        if numpy.sum(~mask):
            print(
                "[mask_objects_outside_lambda_map] Coordinates of the missing GC:",
                self.ra.to("arcsec")[~mask],
                self.dec.to("arcsec")[~mask],
                "and edges:",
                coords_edges_lambda_map.to("arcsec"),
            )
        # mask the relevant quantities
        for quantity in ["ra", "dec", "f150w", "log10sigsky", "prob"]:
            if hasattr(self, quantity):
                setattr(self, quantity, getattr(self, quantity)[mask])
        # self.ra = self.ra[mask]; self.dec = self.dec[mask]; self.f150w = self.f150w[mask]; self.log10sigsky = self.log10sigsky[mask]; self.prob = self.prob[mask]
        print(
            "[mask_objects_outside_lambda_map] There are {:d} GCs within the edges of the lambda map and we lost {:d}".format(
                len(self.ra), numpy.sum(~mask)
            )
        )

        if do_add_edges:
            # add the edges of the lambda map to the GC sample
            self.ra = numpy.append(self.ra, coords_edges_lambda_map[:, 0].to("deg"))
            self.dec = numpy.append(self.dec, coords_edges_lambda_map[:, 1].to("deg"))
            self.f150w = numpy.append(
                self.f150w,
                numpy.zeros(shape=coords_edges_lambda_map.shape[0]) * u.ABmag,
            )
            self.log10sigsky = numpy.append(
                self.log10sigsky,
                numpy.zeros(shape=coords_edges_lambda_map.shape[0])
                * u.dimensionless_unscaled,
            )
            self.prob = numpy.append(
                self.prob,
                numpy.zeros(shape=coords_edges_lambda_map.shape[0])
                * u.dimensionless_unscaled,
            )
            print(
                "[mask_objects_outside_lambda_map] Added {:d} edges to the GC sample".format(
                    coords_edges_lambda_map.shape[0]
                )
            )

    def create_footprint_gcs(self, map_wcs: wcs.WCS) -> None:
        """Return a mask corresponding to the footprint of the GCs within the lambda map.
        :param map_wcs: WCS object of the LambdaMap / GCs
        :return: mask with the footprint of the GCs within the lambda map
        """
        # determine the projected coordinates of the GCs in pixels given a WCS object
        # convention determines that (0,0) refers to the center of the first pixel, not its corner (-0.5, -0.5)
        dummy_pixels = map_wcs.all_world2pix(self.ra.to("deg"), self.dec.to("deg"), 0)
        # gather the relevant data based on the dimension of the interpolation -- deal with quantities as if they were numpy arrays
        x = dummy_pixels[0].copy()
        y = dummy_pixels[1].copy()
        number_of_pixels = [map_wcs.pixel_shape[0], map_wcs.pixel_shape[1]]

        # collect the edges of the spatial distribution of the GCs, which we'll use later to mask the interpolated map
        # pad them with +- 1% of the maximum value to make sure we're not missing any GCs
        ls_x = numpy.linspace(x.min(), x.max(), 30)
        dummy_points = numpy.asarray([])
        for i in range(len(ls_x) - 1):
            mask = (x > ls_x[i]) * (x < ls_x[i + 1])
            # use the limits of the x-coordinates on the edges
            if i == 0:
                point = [ls_x[i], y[mask].min() - map_wcs.pixel_shape[1] // 20]
            elif i == len(ls_x) - 2:
                point = [ls_x[i + 1], y[mask].min() - map_wcs.pixel_shape[1] // 20]
            # otherwise, use the minimum y-coordinate of the GCs in the bin
            else:
                point = [x[mask].min(), y[mask].min() - map_wcs.pixel_shape[1] // 20]
            # make sure we don't go below 0
            if point[1] < -0.5:
                point[1] = -0.5
            dummy_points = numpy.append(dummy_points, point)

        for i in range(len(ls_x) - 1, 0, -1):
            mask = (x > ls_x[i - 1]) * (x < ls_x[i])
            # use the limits of the x-coordinates on the edges
            if i == 1:
                point = [ls_x[i - 1], y[mask].max() + map_wcs.pixel_shape[1] // 20]
            elif i == len(ls_x) - 1:
                point = [ls_x[i], y[mask].max() + map_wcs.pixel_shape[1] // 20]
            # otherwise, use the maximum y-coordinate of the GCs in the bin
            else:
                point = [x[mask].max(), y[mask].max() + map_wcs.pixel_shape[1] // 20]
            # make sure we don't go above the maximum value
            if point[1] > map_wcs.pixel_shape[1] - 0.5:
                point[1] = map_wcs.pixel_shape[1] - 0.5
            dummy_points = numpy.append(dummy_points, point)
        dummy_points = dummy_points.reshape(dummy_points.shape[0] // 2, 2)

        # determine the bins in each axis
        bins_x = numpy.linspace(0, number_of_pixels[0] - 1, number_of_pixels[0])
        bins_y = numpy.linspace(0, number_of_pixels[1] - 1, number_of_pixels[1])

        # create the polygon mask
        # find the pixels of the polygon vertices in the grid
        poly_verts = numpy.vstack(
            (
                numpy.searchsorted(bins_x, dummy_points[:, 0], side="left"),
                numpy.searchsorted(bins_y, dummy_points[:, 1], side="left"),
            )
        ).T
        # create an array with the indices of the image
        inds = numpy.indices((number_of_pixels[0], number_of_pixels[1]))
        points = numpy.vstack((inds[0].flatten(), inds[1].flatten())).T
        # create a Path object with the polygon vertices
        path = Path(poly_verts)
        # create a mask with the polygon
        mask_footprint = path.contains_points(points)
        mask_footprint = mask_footprint.reshape(
            number_of_pixels[0], number_of_pixels[1]
        )
        # store the mask
        self.intp_mask = mask_footprint.copy()

    def create_interpolated_map_probability_recovery_gcs(
        self,
        map_wcs: wcs.WCS,
        do_dimension: int = 4,
        do_return_extra_info: bool = True,
        do_figures: bool = False,
        do_interpolation: bool = True,
    ) -> numpy.ndarray:
        """Return the interpolated map of the probability of recovery of the GCs.
        :type map_wcs: WCS object of the LambdaMap / GCs
        :param do_dimension: dimension of the interpolation (2: (RA, DEC), 3: (RA, DEC, F150W), 4: default (RA, DEC, F150W, log10sigsky))
        :param do_return_extra_info: return extra information on the interpolation grid -- needed for debugging and figures
        :param do_figures: create figures of the interpolation grid
        :param do_interpolation: do the interpolation or not
        """

        def find_bins_with_equal_number_points(array, num_pixels, orig_max):
            # given a certain number of bins, find their edges so they have the same number of points
            points_per_bin = len(array) // num_pixels
            inds = numpy.argsort(array)
            edges = numpy.asarray(
                [array[inds[i * points_per_bin]] for i in range(num_pixels)]
            )
            edges[
                0
            ] *= 0.99  # make the first edge slightly smaller than the minimum value
            edges = numpy.append(
                edges, orig_max * 1.01
            )  # make the outer edge slightly larger than the maximum value
            bins = (edges[1:] + edges[:-1]) / 2
            return bins, edges

        # determine the projected coordinates of the GCs in pixels given a WCS object
        # convention determines that (0,0) refers to the center of the first pixel, not its corner (-0.5, -0.5)
        dummy_pixels = map_wcs.all_world2pix(self.ra.to("deg"), self.dec.to("deg"), 0)
        self.pix_x = dummy_pixels[0] * u.pixel
        self.pix_y = dummy_pixels[1] * u.pixel

        # gather the relevant data based on the dimension of the interpolation -- deal with quantities as if they were numpy arrays
        x = self.pix_x.value.copy()
        y = self.pix_y.value.copy()
        z = self.prob.value.copy()
        if do_dimension >= 3:
            f150w = self.f150w.value.copy()
        if do_dimension >= 4:
            log10sigsky = self.log10sigsky.value.copy()

        # Create a (N, D) array of (pix x, pix y, F150W, log10sigsky) pairs and save the original extrema of the input arrays
        if do_dimension == 2:
            xy = numpy.column_stack([x.flat, y.flat])
            orig_min = [x.min(), y.min()]
            orig_max = [x.max(), y.max()]
        elif do_dimension == 3:
            xy = numpy.column_stack([x.flat, y.flat, f150w.flat])
            orig_min = [x.min(), y.min(), f150w.min()]
            orig_max = [x.max(), y.max(), f150w.max()]
        elif do_dimension == 4:
            xy = numpy.column_stack([x.flat, y.flat, f150w.flat, log10sigsky])
            orig_min = [x.min(), y.min(), f150w.min(), log10sigsky.min()]
            orig_max = [x.max(), y.max(), f150w.max(), log10sigsky.max()]

        print(
            "[create_interpolated_map_probability_recovery_gcs] The input data has been collected in a {:d}d space".format(
                do_dimension
            )
        )

        # collect the edges of the spatial distribution of the GCs, which we'll use later to mask the interpolated map
        # pad them with +- 1% of the maximum value to make sure we're not missing any GCs
        ls_x = numpy.linspace(x.min(), x.max(), 30)
        dummy_points = numpy.asarray([])
        for i in range(len(ls_x) - 1):
            mask = (x > ls_x[i]) * (x < ls_x[i + 1])
            # use the limits of the x-coordinates on the edges
            if i == 0:
                point = [ls_x[i], y[mask].min() - map_wcs.pixel_shape[1] // 20]
            elif i == len(ls_x) - 2:
                point = [ls_x[i + 1], y[mask].min() - map_wcs.pixel_shape[1] // 20]
            # otherwise, use the minimum y-coordinate of the GCs in the bin
            else:
                point = [x[mask].min(), y[mask].min() - map_wcs.pixel_shape[1] // 20]
            # make sure we don't go below 0
            if point[1] < -0.5:
                point[1] = -0.5
            dummy_points = numpy.append(dummy_points, point)

        for i in range(len(ls_x) - 1, 0, -1):
            mask = (x > ls_x[i - 1]) * (x < ls_x[i])
            # use the limits of the x-coordinates on the edges
            if i == 1:
                point = [ls_x[i - 1], y[mask].max() + map_wcs.pixel_shape[1] // 20]
            elif i == len(ls_x) - 1:
                point = [ls_x[i], y[mask].max() + map_wcs.pixel_shape[1] // 20]
            # otherwise, use the maximum y-coordinate of the GCs in the bin
            else:
                point = [x[mask].max(), y[mask].max() + map_wcs.pixel_shape[1] // 20]
            # make sure we don't go above the maximum value
            if point[1] > map_wcs.pixel_shape[1] - 0.5:
                point[1] = map_wcs.pixel_shape[1] - 0.5
            dummy_points = numpy.append(dummy_points, point)
        dummy_points = dummy_points.reshape(dummy_points.shape[0] // 2, 2)

        # collect all possible edge points
        edge_points = numpy.array([])
        for xx in [-0.5, map_wcs.pixel_shape[0] - 0.5]:
            for yy in [-0.5, map_wcs.pixel_shape[1] - 0.5]:
                if do_dimension == 2:
                    edge_points = numpy.append(edge_points, [xx, yy])
                elif do_dimension == 3:
                    for ff in [orig_min[2], orig_max[2]]:
                        edge_points = numpy.append(edge_points, [xx, yy, ff])
                elif do_dimension == 4:
                    for ff in [orig_min[2], orig_max[2]]:
                        for ss in [orig_min[3], orig_max[3]]:
                            edge_points = numpy.append(edge_points, [xx, yy, ff, ss])
        edge_points = edge_points.reshape(
            edge_points.shape[0] // do_dimension, do_dimension
        )
        # append those edge points
        x = numpy.append(x, edge_points[:, 0])
        y = numpy.append(y, edge_points[:, 1])
        if do_dimension >= 3:
            f150w = numpy.append(f150w, edge_points[:, 2])
        if do_dimension == 4:
            log10sigsky = numpy.append(log10sigsky, edge_points[:, 3])
        z = numpy.append(z, numpy.zeros(shape=len(edge_points)))

        # 2: decide on the grid for the interpolation
        number_of_pixels = [map_wcs.pixel_shape[0], map_wcs.pixel_shape[1]]
        if do_dimension >= 3:
            number_of_pixels.append(5)
        if do_dimension == 4:
            number_of_pixels.append(4)
        print(
            "[create_interpolated_map_probability_recovery_gcs] Interpolating the probability of observing GCs in a {:d}d space on a grid of".format(
                do_dimension
            ),
            number_of_pixels,
        )

        # determine the bins in each axis
        bins_x = numpy.linspace(0, map_wcs.pixel_shape[0] - 1, number_of_pixels[0])
        bins_y = numpy.linspace(0, map_wcs.pixel_shape[1] - 1, number_of_pixels[1])
        edges_x = numpy.linspace(
            -0.5, map_wcs.pixel_shape[0] - 0.5, number_of_pixels[0] + 1
        )
        edges_y = numpy.linspace(
            -0.5, map_wcs.pixel_shape[1] - 0.5, number_of_pixels[1] + 1
        )
        if do_dimension >= 3:
            # for F150W, determine bins with equal number of points per bin
            bins_f150w, edges_f150w = find_bins_with_equal_number_points(
                f150w, number_of_pixels[2], orig_max[2]
            )
        if do_dimension == 4:
            # for the sky noise, determine bins with equal number of points per bin
            bins_log10sky, edges_log10sky = find_bins_with_equal_number_points(
                log10sigsky, number_of_pixels[3], orig_max[3]
            )

        # normalise the bins in x and y
        norm_bins_x = bins_x / bins_x.max()
        norm_bins_y = bins_y / bins_y.max()

        # generate the grid to interpolate the map on and try nearest neghbour interpolation
        if do_dimension == 2:  # (x, y)
            combined_grid = tuple(
                numpy.meshgrid(norm_bins_x, norm_bins_y, indexing="ij")
            )
            xy = numpy.column_stack([x.flat / x.max(), y.flat / y.max()])
        elif do_dimension == 3:  # (x, y, F150W)
            combined_grid = tuple(
                numpy.meshgrid(norm_bins_x, norm_bins_y, bins_f150w, indexing="ij")
            )
            xy = numpy.column_stack([x.flat / x.max(), y.flat / y.max(), f150w.flat])
        elif do_dimension == 4:  # (x, y, F150W, log10sigsky)
            combined_grid = tuple(
                numpy.meshgrid(
                    norm_bins_x, norm_bins_y, bins_f150w, bins_log10sky, indexing="ij"
                )
            )
            xy = numpy.column_stack(
                [x.flat / x.max(), y.flat / y.max(), f150w.flat, log10sigsky.flat]
            )

        # and interpolate!
        if do_interpolation:
            interpolated_map = scipy.interpolate.griddata(
                xy, z, combined_grid, method="nearest"
            )
            print(
                "[create_interpolated_map_probability_recovery_gcs] The interpolation has run",
                interpolated_map.shape,
            )

        # create the polygon mask
        # find the pixels of the polygon vertices in the grid
        poly_verts = numpy.vstack(
            (
                numpy.searchsorted(bins_x, dummy_points[:, 0], side="left"),
                numpy.searchsorted(bins_y, dummy_points[:, 1], side="left"),
            )
        ).T
        # create an array with the indices of the image
        inds = numpy.indices((number_of_pixels[0], number_of_pixels[1]))
        points = numpy.vstack((inds[0].flatten(), inds[1].flatten())).T
        # create a Path object with the polygon vertices
        path = Path(poly_verts)
        # create a mask with the polygon
        mask = path.contains_points(points)
        mask = mask.reshape(number_of_pixels[0], number_of_pixels[1])
        if do_interpolation:
            # pixels outside of the mask are set to one to avoid influencing the normalisation term
            if do_dimension == 2:
                interpolated_map[~mask] = 0
            if do_dimension == 3:
                for i in range(number_of_pixels[2]):
                    interpolated_map[:, :, i][~mask] = 0
            elif do_dimension == 4:
                for i in range(number_of_pixels[2]):
                    for j in range(number_of_pixels[3]):
                        interpolated_map[:, :, i, j][~mask] = 0

            if do_figures:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
                # ax.imshow(numpy.rot90(ma.masked_invalid(mask), k = -1), origin = "lower")
                ax.imshow(mask.T, origin="lower")
                ax.scatter(poly_verts[:, 0], poly_verts[:, 1], c="C0", marker="o")
                ax.scatter(x, y, c="C1", marker=".", s=1)
                ax.set_ylabel("y [pix]")
                ax.set_xlabel("x [pix]")
                ax.set_title("Mask applied to the interpolated map")
                fig.show()
                plt.close()

        # store the resulting interpolated map and its limits
        if do_return_extra_info:
            self.intp_z = z.copy() * u.dimensionless_unscaled
            # store the mask applied to the interpolated map
            self.intp_mask = mask.copy()

            # renormalise the x and y axes
            if do_dimension == 2:
                self.intp_xy = (
                    xy[:, 0] * x.max() * u.pixel,
                    xy[:, 1] * y.max() * u.pixel,
                )
                self.intp_bins = (bins_x * u.pixel, bins_y * u.pixel)
                self.intp_edges = (edges_x * u.pixel, edges_y * u.pixel)
            elif do_dimension == 3:
                self.intp_xy = (
                    xy[:, 0] * x.max() * u.pixel,
                    xy[:, 1] * y.max() * u.pixel,
                    xy[:, 2] * u.ABmag,
                )
                self.intp_bins = (
                    bins_x * u.pixel,
                    bins_y * u.pixel,
                    bins_f150w * u.ABmag,
                )
                self.intp_edges = (
                    edges_x * u.pixel,
                    edges_y * u.pixel,
                    edges_f150w * u.ABmag,
                )
            elif do_dimension == 4:
                self.intp_xy = (
                    xy[:, 0] * x.max() * u.pixel,
                    xy[:, 1] * y.max() * u.pixel,
                    xy[:, 2] * u.ABmag,
                    xy[:, 3] * u.dimensionless_unscaled,
                )
                self.intp_bins = (
                    bins_x * u.pixel,
                    bins_y * u.pixel,
                    bins_f150w * u.ABmag,
                    bins_log10sky * u.dimensionless_unscaled,
                )
                self.intp_edges = (
                    edges_x * u.pixel,
                    edges_y * u.pixel,
                    edges_f150w * u.ABmag,
                    edges_log10sky * u.dimensionless_unscaled,
                )

        if do_interpolation:
            self.intp_map = interpolated_map.copy()
