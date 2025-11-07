### Script containing the basic routines for calculating the
### inhomogeneous Poisson point process likelihood under the continuous assumption
### Author: Marta Reina-Campos
### Date: October 2025

# Import modules
import numpy, time, scipy
from master_class_gcs import GCs
from master_class_lambdamaps import LambdaMap, LensingMap, StellarLightMap, XrayMap
from master_class_galaxy_cluster import GalaxyCluster
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy import wcs
import skimage

## LOAD THE LAMBDA MAP INSTANCE BASED ON THE TYPE OF MAP REQUESTED


def create_instance_lambda_map(
    type_map: str, do_lambda_map: str, gcs: GCs, galaxy_cluster: GalaxyCluster
):
    """Create the instance of the lambda map based on the type of map requested
    :param type_map: str -- type of map to create
    :param do_lambda_map: str -- path to the map to load
    :param gcs: GCs -- instance of the GCs class
    :param galaxy_cluster: GalaxyCluster -- instance of the GalaxyCluster class
    """
    if type_map in ["lensing map", "uniform map", "noisy map"]:
        if "noisy" in type_map or "uniform" in type_map:
            lambda_map = LensingMap(
                do_lambda_map, type_map, gcs=gcs, galaxy_cluster=galaxy_cluster
            )
        else:
            lambda_map = LensingMap(do_lambda_map, type_map)
    elif type_map in ["stellar light", "bcgless map"]:
        lambda_map = StellarLightMap(do_lambda_map, type_map)
    elif type_map in ["xray map"]:
        lambda_map = XrayMap(do_lambda_map, type_map)
    return lambda_map


### FUNCTIONS TO CREATE THE NUMBER DENSITY MAPS OF GCS ###


def create_number_density_gcs(
    gc_sample: GCs,
    number_of_pixels: numpy.ndarray = numpy.asarray([1024, 1024]),
):
    """Function to calculate the number density maps of GCs given a 2d point distribution.
    :param gc_sample: instance of the GCs class containing the sample of GCs
    :param number_of_pixels: number of pixels in the x and y directions of the desired image
    :return: number density of GCs in the image in weights kpc^-2
    """
    gc_pixels = numpy.asarray(
        gc_sample.wcs.all_world2pix(gc_sample.ra.to("deg"), gc_sample.dec.to("deg"), 0)
    ).T
    # create the bins - convention places (0,0) at the center of the pixel
    bins = [
        numpy.linspace(-0.5, number_of_pixels[0] - 0.5, number_of_pixels[0] + 1),
        numpy.linspace(-0.5, number_of_pixels[1] - 0.5, number_of_pixels[1] + 1),
    ]

    print(
        f"[create_number_density_gcs] Creating number density of GCs in {number_of_pixels[0]} x {number_of_pixels[1]} pixels"
    )

    # create the 2d histogram
    img, xedges, yedges = numpy.histogram2d(
        gc_pixels[:, 0], gc_pixels[:, 1], bins=bins, weights=gc_sample.prob
    )
    img = img.transpose()  # add a small number to avoid division by zero

    # create the number density image of GCs - units: counts arcsec^-2
    area_pixel = gc_sample.wcs.proj_plane_pixel_area().to("arcsec^2")
    nrho = img / area_pixel

    # return the image and its edges
    return nrho, xedges, yedges


def create_smoothed_number_density_gcs(
    img: numpy.ndarray, gc_sample: GCs, sigma_arcsec: 1.0 * u.arcsec
):
    """Function to smooth the number density maps of GCs given a 2d point distribution with a gaussian kernel.
    :param img: 2d numpy array -- number density image of GCs
    :param gc_sample: instance of the GCs class containing the sample of GCs
    :param sigma_arcsec: float -- size of the gaussian kernel in arcseconds
    :return: smoothed number density of GCs in the image in weights arcsec^-2
    """
    # convert the kernel size to pixels
    sigma_px = float(
        sigma_arcsec
        / (
            numpy.mean(
                numpy.abs(gc_sample.wcs.pixel_scale_matrix.diagonal())
                * u.deg.to("arcsec")
            )
            * u.arcsec
        )
    )
    # smoothed GC number density - units: counts/ arcsec^2
    smooth_img_gcs = scipy.ndimage.gaussian_filter(img.value, sigma_px) + 1e-10
    return smooth_img_gcs * img.unit


def create_both_number_density_and_smoothed_maps_gcs(
    gc_sample: GCs,
    number_of_pixels: numpy.ndarray = numpy.asarray([1024, 1024]),
    sigma_arcsec: float = 1.0 * u.arcsec,
):
    """Function to calculate both the number density maps of GCs, and the smoothed version with a gaussian kernel.
    :param gc_sample: instance of the GCs class containing the sample of GCs
    :param number_of_pixels: number of pixels in the x and y directions of the desired image
    :param sigma_arcsec: float -- size of the gaussian kernel in arcseconds
    :return: number density of GCs in the image in weights arcsec^-2, and the smoothed version
    """
    # create the number density image of GCs - units: arcsec^-2
    img, xedges, yedges = create_number_density_gcs(gc_sample, number_of_pixels)
    # smooth the images with a gaussian kernel
    smooth_img = create_smoothed_number_density_gcs(img, gc_sample, sigma_arcsec)

    return img, smooth_img


### FUNCTIONS TO CALCULATE THE POISSON PROBABILITY ###


def renormalize_map(img: numpy.ndarray, do_verbose: bool = False) -> None:
    """Normalize the map such that all values lie between [0, 1] and it is dimensionless.
    :param img: 2d numpy array -- image to renormalize
    :param do_verbose: bool -- whether to print verbose output
    :return: renormalized image
    """
    img = img / numpy.nanmax(img)
    # move the floor in the image to 1e-10 the lowest value, rather than 0
    mask = img > 0
    img[~mask] = 1e-10 * numpy.nanmin(img[mask])
    if do_verbose:
        print(
            "[renormalize_map] The image now extends between {:e} and {:e}".format(
                numpy.nanmin(img), numpy.nanmax(img)
            )
        )
    return img


def find_pixels_within_lambda_map(gcs: GCs, lambda_map: LambdaMap):
    """Given their coordinates, find the pixels wherein the GCs lie within the lambda map.
    :param gcs: instance of the GCs class containing the sample of GCs
    :param lambda_map: instance of the LambdaMap class containing the lambda map
    :return: array with the pixel coordinates of the GCs within the lambda map
    """
    # convert coordinates to pixels
    pix2d = numpy.array(
        lambda_map.wcs.all_world2pix(gcs.ra.to("deg"), gcs.dec.to("deg"), 0)
    ).astype(int)
    pix_gcs = pix2d.T

    # make sure we don't have GCs in the border of the map
    mask = pix_gcs[:, 0] >= lambda_map.img.shape[0]
    if mask.sum():
        print(
            "[find_pixels_within_lambda_map] Corrected pixels in x for {:d} GCs".format(
                mask.sum()
            )
        )
        pix_gcs[mask, 0] = lambda_map.img.shape[0] - 1
    mask = pix_gcs[:, 1] >= lambda_map.img.shape[1]
    if mask.sum():
        print(
            "[find_pixels_within_lambda_map] Corrected pixels in y for {:d} GCs".format(
                mask.sum()
            )
        )
        pix_gcs[mask, 1] = lambda_map.img.shape[1] - 1

    return pix_gcs


def spawn_datapoints_from_image(number: int, image: numpy.ndarray) -> numpy.ndarray:
    """Spawn N data points from an image using the (normalized) pixel values as probabilities.
    :param number: int -- number of data points to spawn
    :param image: 2d numpy array -- image to spawn data points from
    :return: array with the pixel coordinates of the spawned data points
    """
    # Flatten the 2D array and normalize to ensure it sums to 1
    prob_array_flat = image.flatten()
    prob_array_normalized = prob_array_flat / numpy.sum(prob_array_flat)

    # Create a list of indices corresponding to the flattened array
    indices = numpy.arange(prob_array_normalized.size)

    # Sample a random index based on the probabilities
    sampled_index = numpy.random.choice(indices, size=number, p=prob_array_normalized)

    # Convert the sampled flat index back to 2D coordinates
    inds = numpy.asarray(divmod(sampled_index, image.shape[1]))  # returns row, col

    return inds.T


def calculate_normalization_poisson_probability(
    f150w_min: u.ABmag,
    f150w_max: u.ABmag,
    map_sky_noise: numpy.ndarray,
    gcs: GCs,
    lambda_map: LambdaMap,
) -> float:
    """
    Calculate the normalization constant for the Poisson probability distribution.
    :param f150w_min: float -- minimum F150W magnitude
    :param f150w_max: float -- maximum F150W magnitude
    :param map_sky_noise: 2D numpy array -- map of the local sky
    :param gcs: instance of the GCs class containing the sample of GCs
    :param lambda_map: instance of the LambdaMap class containing the lambda map
    :return: float -- normalization constant
    """
    # integrate the probability of recovery over the magnitude range for the observed sky noise map
    img_probability_recovery = gcs.integrate_probability_of_recovery(
        f150w_min, f150w_max, map_sky_noise
    )

    # integrate the effective occurence rate (lambda map * probability of recovery) over the field of view
    normalization_constant = numpy.sum(lambda_map.img.value * img_probability_recovery)

    return normalization_constant


def calculate_continuous_spatial_poisson_probability(
    normalization: float, lambda_map: LambdaMap, gcs: GCs, do_verbose: bool = False
):
    """Calculates the spatial Poisson probability of observing the GCs in the bounded region B
    given the map of the effective rate lambda_bi and the selection function s_xy
    :param lambda_map: instance of the LambdaMap class containing the lambda map
    :param gcs: instance of the GCs class containing the sample of GCs
    :param do_verbose: bool -- whether to print verbose output
    :return: ln_prob -- float -- logarithm of the Poisson probability
    """

    # find the pixels of the GCs within the lambda map
    pixels = find_pixels_within_lambda_map(gcs, lambda_map)

    # renormalize the lambda map to be between [0, 1]
    lambda_map.img = renormalize_map(lambda_map.img)

    start = time.time()
    # choose the values of the rate for the pixels that contain GCs
    ln_effective_rate = numpy.log(
        lambda_map.img[pixels[:, 0], pixels[:, 1]].value * gcs.prob
    )

    # remove pixels that might have NaNs or infs because the GCs are on the edge of the masking region
    mask = numpy.isinf(ln_effective_rate) | numpy.isnan(ln_effective_rate)
    if numpy.sum(mask):
        print("Missing data on {:d} points".format(numpy.sum(mask)))
    # calculate the Poisson probability as ln P = sum_i ln(ln_effective_rate) - normalisation
    ln_prob = -normalization + numpy.sum(ln_effective_rate[~mask])

    if do_verbose:
        print(
            f"[calculate_spatial_poisson_probability] Calculating ln_prob p={ln_prob:.10e} took {time.time() - start:.10e} seconds"
        )
    return ln_prob


def spawn_magnitudes(
    number_gcs: int,
    min_mag: u.ABmag,
    max_mag: u.ABmag,
    m0: u.ABmag,
    sigma: u.ABmag,
    do_verbose: bool = False,
) -> numpy.ndarray:
    """Spawn random magnitudes for the GCs based on a GC luminosity function."""
    """ Based on https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    :param number_gcs: int -- number of GCs to spawn magnitudes for
    :param min_mag: float -- minimum magnitude to spawn
    :param max_mag: float -- maximum magnitude to spawn
    :param m0: float -- mean of the log-normal distribution
    :param sigma: float -- standard deviation of the log-normal distribution
    :param do_verbose: bool -- whether to print verbose output
    :return: array with the spawned magnitudes"""

    # spawn the magnitudes from a truncated normal distribution
    # the normal distribution is defined as:
    # f(x) = (1/(sigma*sqrt(2*pi))) * exp(-((x - m0)^2)/(2*sigma^2))
    # where m0 is the mean of the log of the distribution and sigma is the standard deviation of the log of the distribution
    # we truncate it to be between min_mag and max_mag
    lognormal = scipy.stats.truncnorm(
        loc=m0.value,
        scale=sigma.value,
        a=(min_mag.value - m0.value) / sigma.value,
        b=(max_mag.value - m0.value) / sigma.value,
    )
    samples = lognormal.rvs(size=number_gcs)  # units: ABmag

    if do_verbose:
        fig, ax = plt.subplots(1, figsize=(8, 5.5))
        print(
            f"[spawn_magnitudes] Spawning {number_gcs} magnitudes between {min_mag} and {max_mag}"
        )
        # plot the distribution
        ax.hist(samples, bins=50, density=True, label="Samples")
        x = numpy.linspace(min_mag.value, max_mag.value, 1000)
        ax.plot(x, lognormal.pdf(x), label="PDF")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.legend(loc="upper left")
        ax.set_ylim(1e-5, None)
        plt.show()

    return samples

def find_minimum_common_area_between_maps(map1: LambdaMap, map2: LambdaMap, map3: LambdaMap) -> tuple[list, list]:
    """ Find the coordinates defining the minimum common area between three maps
    :param map1: instance of LabdaMap
    :param map2: instance of LabdaMap
    :param map3: instance of LabdaMap
    :return lists containing the min/max RA and DEC
    """
    # find the coordinates of the edges in each map
    xlim_ra = []
    ylim_dec = []
    for map in [map1, map2, map3]:
        xlim_ra.append(
            numpy.asarray(
                map.wcs.all_pix2world(
                    [-0.5, map.img.shape[0] + 0.5], [-0.5, -0.5], 0
                )
            )[0]
        )
        ylim_dec.append(
            numpy.asarray(
                map.wcs.all_pix2world(
                    [-0.5, -0.5], [-0.5, map.img.shape[1] + 0.5], 0
                )
            )[1]
        )
    xlim_ra = numpy.asarray(xlim_ra)
    ylim_dec = numpy.asarray(ylim_dec)
    # find the minimum common area
    map_xlim_ra = [numpy.min(xlim_ra[:, 0]), numpy.max(xlim_ra[:, 1])]
    map_ylim_dec = [numpy.max(ylim_dec[:, 0]), numpy.min(ylim_dec[:, 1])]
    print(
        "[find_minimum_common_area_between_maps], BOTH - RA, DEC",
        map_xlim_ra,
        map_ylim_dec,
    )
    return map_xlim_ra, map_ylim_dec

def apply_minimum_common_limits_to_image(
    lambda_map_xlim_ra: list, lambda_map_ylim_dec:list, map_to_limit: LambdaMap
) -> LambdaMap:
    """Apply the minimum common area limits to a given map
    Input:
    :param lambda_map_xlim_ra: list of two elements with the min and max RA limits
    :param lambda_map_ylim_dec: list of two elements with the min and max DEC limits
    :param map: instance of a LambdaMap class"""
    _lim_pix = numpy.floor(
        map_to_limit.wcs.all_world2pix(lambda_map_xlim_ra, lambda_map_ylim_dec, 0)
    ).astype(int)
    # yy, xx = numpy.meshgrid(range(lambda_map1.img.shape[1]), range(lambda_map1.img.shape[0]))
    # restrict the range of lambda map1 to avoid spawning datapoints where there's no information in lambda map2
    if _lim_pix[0][0] < 0:
        _lim_pix[0][0] = 0
    if _lim_pix[1][0] < 0:
        _lim_pix[1][0] = 0
    if (
        _lim_pix[0][1] > map_to_limit.img.shape[0] + 1
        or _lim_pix[1][1] > map_to_limit.img.shape[1] + 1
    ):
        print(
            f"[apply_minimum_common_limits_to_image] WARNING: pixel limits ({_lim_pix}) for lambda map 1 exceed image dimensions ({map_to_limit.img.shape})"
        )

    # only keep the pixels within the minimum common area
    tmp_img = map_to_limit.img[
        _lim_pix[0][0] : _lim_pix[0][1], _lim_pix[1][0] : _lim_pix[1][1]
    ]
    map_to_limit.img = tmp_img.copy()
    # update the WCS and header information accordingly
    map_to_limit.wcs.wcs.crpix[0] -= _lim_pix[0][0]
    map_to_limit.wcs.wcs.crpix[1] -= _lim_pix[1][0]
    map_to_limit.header["CRPIX1"] = map_to_limit.wcs.wcs.crpix[0]
    map_to_limit.header["CRPIX2"] = map_to_limit.wcs.wcs.crpix[1]
    map_to_limit.header["NAXIS1"] = map_to_limit.img.shape[0]
    map_to_limit.header["NAXIS2"] = map_to_limit.img.shape[1]

    return map_to_limit

def reduce_and_rebin_image(lambda_map: LambdaMap, map_to_modify: LambdaMap) -> tuple[numpy.ndarray, wcs.WCS, fits.Header]:
    """Given a lambda map and a map of probability of recovery, reduce and rebin the latter so that they can be multiplied together.
    :param lambda_map: instance of LambdaMap to adapt into
    :param map_to_modify: image to rebin
    :return rebinned_img
    :return rebinned_wcs
    :return rebinned_hdr 
    """
    rebinned_hdr = map_to_modify.header.copy()

    factor_resolution_to_change = (
        map_to_modify.img.shape[0] / lambda_map.header["NAXIS1"],
        map_to_modify.img.shape[1] / lambda_map.header["NAXIS2"],
    )
    rebinned_hdr["NAXIS1"] = lambda_map.header["NAXIS1"]
    rebinned_hdr["NAXIS2"] = lambda_map.header["NAXIS2"]
    rebinned_hdr["CRPIX1"] = (
        map_to_modify.header["CRPIX1"]
    ) / factor_resolution_to_change[
        0
    ]  #  - pixels_edges_lambda_map[0,0]
    rebinned_hdr["CRPIX2"] = (
        map_to_modify.header["CRPIX2"]
    ) / factor_resolution_to_change[
        1
    ]  # - pixels_edges_lambda_map[0,1]
    rebinned_hdr["CD1_1"] = (
        map_to_modify.header["CD1_1"] * factor_resolution_to_change[0]
    )
    rebinned_hdr["CD2_2"] = (
        map_to_modify.header["CD2_2"] * factor_resolution_to_change[1]
    )
    rebinned_wcs = wcs.WCS(rebinned_hdr)

    # rebin the image into the arbitrary resolution of the lambda map
    rebinned_img = skimage.transform.resize(
        map_to_modify.img,
        (rebinned_hdr["NAXIS1"], rebinned_hdr["NAXIS2"]),
        anti_aliasing=True,
    )

    return rebinned_img, rebinned_wcs, rebinned_hdr