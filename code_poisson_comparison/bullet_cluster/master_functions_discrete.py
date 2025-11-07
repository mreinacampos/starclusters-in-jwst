### Script containing the basic routines for calculating
# the inhomogeneous Poisson point process likelihood under the discrete assumption
### Author: Marta Reina-Campos
### Date: Dec 4th 2024

# Import modules
import numpy, scipy, time, pickle
from master_class_gcs import DataPoints, GCs
from master_class_lambdamaps import LambdaMap
from typing import Union


def find_pixels_within_lambda_map(
    gcs: GCs, lambda_map: LambdaMap, do_dimension: int = 2
):
    """Find the pixels of the GCs within the high-dimensionality space of the extended lambda map
    :type gcs: DataPoints or GCs instance
    :type lambda_map: LambdaMap instance
    :type do_dimension: int -- number of dimensions of the high-dimensionality space
    :return: pix_gcs -- array of shape (N_gcs, do_dimension) with the pixel indices of each GC in the lambda map
    """

    pix2d = numpy.array(
        lambda_map.wcs.all_world2pix(gcs.ra.to("deg"), gcs.dec.to("deg"), 0)
    ).astype(int)
    if do_dimension == 2:
        pix_gcs = pix2d.T
    elif do_dimension == 3:
        pix_gcs = numpy.vstack(
            (pix2d, numpy.searchsorted(gcs.intp_edges[2], gcs.f150w, side="left") - 1)
        ).T
    elif do_dimension == 4:
        pix_gcs = numpy.vstack(
            (
                pix2d,
                numpy.searchsorted(gcs.intp_edges[2], gcs.f150w, side="left") - 1,
                numpy.searchsorted(gcs.intp_edges[3], gcs.log10sigsky, side="left") - 1,
            )
        ).T

    # make sure we don't have GCs in the border of the map
    mask = pix_gcs[:, 0] >= lambda_map.img_extended.shape[0]
    if mask.sum():
        pix_gcs[mask, 0] = lambda_map.img_extended.shape[0] - 1
        print("Corrected pixels in x for {:d} GCs".format(mask.sum()))
    mask = pix_gcs[:, 1] >= lambda_map.img_extended.shape[1]
    if mask.sum():
        pix_gcs[mask, 1] = lambda_map.img_extended.shape[1] - 1
        print("Corrected pixels in y for {:d} GCs".format(mask.sum()))

    return pix_gcs


def save_information_about_interpolation(fname: str, gcs: GCs):
    """Save the information about the interpolation of the selection function
    :param fname: str -- filename to save the information
    :param gcs: GCs instance -- instance of the GCs class containing the sample
    """
    # save the information on the interpolated map
    numpy.savez_compressed(fname + "_maps", map=gcs.intp_map, mask=gcs.intp_mask)
    # save the edges
    with open(fname + "_edges.pickle", "wb") as f:
        pickle.dump(gcs.intp_edges, f)
    # save the bins
    with open(fname + "_bins.pickle", "wb") as f:
        pickle.dump(gcs.intp_bins, f)
    # save the data
    with open(fname + "_xy.pickle", "wb") as f:
        pickle.dump(gcs.intp_xy, f)
    # save the data
    with open(fname + "_z.pickle", "wb") as f:
        pickle.dump(gcs.intp_z, f)


def read_information_about_interpolation(fname: str, gcs: GCs):
    """Read the information about the interpolation of the selection function
    :param fname: str -- filename to read the information from
    :param gcs: GCs instance -- instance of the GCs class containing the sample
    """
    # load the information
    saved_file = numpy.load(fname + "_maps.npz")
    gcs.intp_map = saved_file["map"]
    gcs.intp_mask = saved_file["mask"]

    with open(fname + "_edges.pickle", "rb") as f:
        gcs.intp_edges = pickle.load(f)
    with open(fname + "_bins.pickle", "rb") as f:
        gcs.intp_bins = pickle.load(f)
    with open(fname + "_xy.pickle", "rb") as f:
        gcs.intp_xy = pickle.load(f)
    with open(fname + "_z.pickle", "rb") as f:
        gcs.intp_z = pickle.load(f)


def spawn_datapoints_from_image(
    number: int,
    image: numpy.ndarray,
    do_dimension: int,
    name: str = "data_points",
    do_verbose: bool = False,
):
    """Spawn N data points from an image using the (normalized) pixel values as probabilities.
    :param number: int -- number of data points to spawn
    :param image: numpy.ndarray -- 2D/3D/4D array with
    :param do_dimension: int -- number of dimensions of the high-dimensionality space
    :param name: str -- name for the output files
    :param do_verbose: bool -- whether to print verbose output
    :return: inds -- array of shape (number, do_dimension) with the indices of
              the spawned data points in each dimension
    """

    # Flatten the 2D array and normalize to ensure it sums to 1
    prob_array_flat = image.flatten()
    prob_array_normalized = prob_array_flat / numpy.sum(prob_array_flat)

    # Create a list of indices corresponding to the flattened array
    indices = numpy.arange(prob_array_normalized.size)

    # Sample a random index based on the probabilities
    sampled_index = numpy.random.choice(indices, size=number, p=prob_array_normalized)

    if do_verbose:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, figsize=(10, 5))
        mask = prob_array_normalized > 0
        ax.hist(
            numpy.log10(prob_array_normalized[mask]),
            bins=100,
            density=False,
            histtype="step",
            color="C0",
        )
        # print(f"[spawn_datapoints_from_image] Probabilities in the image range from {numpy.min(prob_array_normalized[mask])} to {numpy.max(prob_array_normalized[mask])}")
        ax.hist(
            numpy.log10(prob_array_normalized[sampled_index]),
            bins=100,
            density=False,
            histtype="step",
            color="C2",
            label=f"Sampled {number} points",
        )
        # print(f"[spawn_datapoints_from_image] Probabilities CHOSEN in the image range from {numpy.min(prob_array_normalized[sampled_index])} to {numpy.max(prob_array_normalized[sampled_index])}")
        ax.set_yscale("log")
        ax.legend(loc="best")
        ax.set_xlabel(f"Probabilities")
        ax.set_ylabel("Density")
        fig.savefig(f"fig_hist_probabilities_{name}.png", bbox_inches="tight")
        plt.close(fig)

    # safety net
    if len(image.shape) < do_dimension:
        print(
            "[spawn_datapoints_from_image] ERROR: image doesn't have sufficient dimensions"
        )

    # Convert the sampled flat index back to 2D coordinates
    if do_dimension == 2:
        inds = numpy.asarray(divmod(sampled_index, image.shape[1]))  # returns row, col
    elif do_dimension == 3:
        # recover the location of the sampled indices
        d0, tmpidx = divmod(sampled_index, image.shape[1] * image.shape[2])
        d1, d2 = divmod(tmpidx, image.shape[2])

        # check the indices are recovered within the right ranges
        if (
            numpy.sum(d0 > image.shape[0])
            or numpy.sum(d1 > image.shape[1])
            or numpy.sum(d2 > image.shape[2])
            or numpy.sum(
                d0 * image.shape[1] * image.shape[2] + d1 * image.shape[2] + d2
                != sampled_index
            )
        ):
            print(
                "[spawn_datapoints_from_image] WARNING - There is an error in the dimensions of the indices"
            )
        inds = numpy.asarray([d0, d1, d2])

    elif do_dimension == 4:
        # recover the location of the sampled indices
        d0, tmpidx0 = divmod(
            sampled_index, image.shape[1] * image.shape[2] * image.shape[3]
        )
        d1, tmpidx1 = divmod(tmpidx0, image.shape[2] * image.shape[3])
        d2, d3 = divmod(tmpidx1, image.shape[3])

        # check the indices are recovered within the right ranges
        if (
            numpy.sum(d0 > image.shape[0])
            or numpy.sum(d1 > image.shape[1])
            or numpy.sum(d2 > image.shape[2])
            or numpy.sum(d3 > image.shape[3])
        ):
            print(
                f"[spawn_datapoints_from_image] WARNING - There is an error in the dimensions of the indices - {numpy.sum(d0 > image.shape[0])},{numpy.sum(d1 > image.shape[1])},{numpy.sum(d2 > image.shape[2])},{numpy.sum(d3 > image.shape[3])}"
            )
        if numpy.sum(
            d0 * image.shape[1] * image.shape[2] * image.shape[3]
            + d1 * image.shape[2] * image.shape[3]
            + d2 * image.shape[3]
            + d3
            != sampled_index
        ):
            mask = (
                d0 * image.shape[1] * image.shape[2] * image.shape[3]
                + d1 * image.shape[2] * image.shape[3]
                + d2 * image.shape[3]
                + d3
                == sampled_index
            )
            print(
                f"[spawn_datapoints_from_image] WARNING - There is an error in the reconstruction of the sampled_index - {numpy.sum(~mask)} - sampled_index: {sampled_index[~mask]} - d0: {d0[~mask]} - d1: {d1[~mask]} - d2: {d2[~mask]} - d3: {d3[~mask]}"
            )

        inds = numpy.asarray([d0, d1, d2, d3])

    return inds.T


def renormalize_map(img: numpy.ndarray, do_verbose: bool = False) -> None:
    """Normalize the map such that all values lie between [0, 1] and it is dimensionless.
    :param img: numpy.ndarray -- image to renormalize
    :param do_verbose: bool -- whether to print verbose output
    :return: img -- renormalized image
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


def determine_effective_occurrence_rate(
    points: Union[DataPoints, GCs], lambda_map: LambdaMap, do_verbose: bool = False
):
    """Determines the effective occurrence rate of the GCs in the bounded region B
    given the map of the effective rate lambda_bi and the selection function s_xy
    :param points: DataPoints or GCs instance
    :param lambda_map: LambdaMap instance
    :param do_verbose: bool -- whether to print verbose output
    :return: lambda_eff -- numpy.ndarray with the effective occurrence rate map
    """
    # effective rate = lambda map * selection function
    unnorm_lambda_eff = lambda_map.img_extended * points.intp_map  # lambda_bi * s_xy

    start = time.time()
    # 1: renormalize the map to be between [0, 1] to get rid of the units of the effective rate (i.e. solar masses, fluxes, photon counts, etc.)
    lambda_eff = renormalize_map(unnorm_lambda_eff, do_verbose=do_verbose)
    # print("[calculate_spatial_poisson_probability] lambda_eff units and shape", lambda_eff.shape)
    if do_verbose:
        print(
            f"[determine_effective_occurrence_rate] Renormalizing Lambda_eff took {time.time() - start} seconds"
        )

    return lambda_eff


def determine_pixel_size(points: Union[DataPoints, GCs], do_verbose: bool = False):
    """Determine the size of the bins in each dimension: \Delta V_i = \Delta x_i \Delta y_i \Delta \theta_i
    :param points: DataPoints or GCs instance
    :param do_verbose: bool -- whether to print verbose output
    :return: dV -- numpy.ndarray with the volume of each bin
    """
    start = time.time()

    # gather the size of the bins in each dimension
    dbins = [numpy.diff(points.intp_edges[i]) for i in range(len(points.intp_edges))]
    # calculate the volume of each bin
    dV = 1
    ddVV = numpy.meshgrid(*dbins, indexing="ij")
    for x in ddVV:
        dV *= x.value
    if do_verbose:
        print(
            f"[determine_pixel_size] Calculating dV took {time.time() - start} seconds"
        )
    return dV


def calculate_normalization_factor(
    lambda_eff: numpy.ndarray, dV: numpy.ndarray, do_verbose: bool = False
):
    """Calculates the normalization factor for the Poisson probability
    :param lambda_eff: numpy.ndarray -- effective occurrence rate map
    :param dV: numpy.ndarray -- volume of each bin
    :param do_verbose: bool -- whether to print verbose output
    :return: first_term -- normalization factor
    """
    start = time.time()
    # 1: first term, summation over the number of pixels: - \sum_{i=1, k} \Lambda_{\rm eff}(x_i, y_i, \theta_i)
    # calculate it using the Riemann sum -- midpoint rule and the logsumexp() from scipy.special
    first_term = numpy.exp(scipy.special.logsumexp(numpy.log(lambda_eff * dV)))
    if do_verbose:
        print(
            f"[calculate_normalization_factor] Calculating first_term took {time.time() - start} seconds"
        )
    return first_term


def calculate_spatial_poisson_probability(
    lambda_eff: numpy.ndarray,
    dV: numpy.ndarray,
    pixels: numpy.ndarray,
    first_term: float,
    do_dimension: int = 2,
    do_verbose: bool = False,
):
    """Calculates the spatial Poisson probability of observing the GCs in the bounded region B
    given the map of the effective rate lambda_bi and the selection function s_xy
    :param lambda_eff: numpy.ndarray -- effective occurrence rate map
    :param dV: numpy.ndarray -- volume of each bin
    :param pixels: numpy.ndarray -- array of shape (N_gcs, do_dimension)
    :param first_term: float -- normalization factor
    :param do_dimension: int -- number of dimensions of the high-dimensionality space
    :param do_verbose: bool -- whether to print verbose output
    :return: ln_prob -- float -- logarithm of the Poisson probability
    """
    start = time.time()
    # 2: find how many points fall in each pixel
    # initialize a 2D array for the image
    counts_per_pixel = numpy.zeros(lambda_eff.shape, dtype=int)
    # efficiently count how many points are there per pixel
    tuple_pixels = tuple([pixels[:, i] for i in range(do_dimension)])
    numpy.add.at(counts_per_pixel, tuple_pixels, 1)

    # 3: second term, summation over the number of pixels: \sum_{i=1, k} \ln(\Lambda_{\rm eff}(x_i, y_i, \theta_i) dV) n_i
    mask = counts_per_pixel > 0
    second_term = numpy.sum(
        numpy.log(lambda_eff[mask] * dV[mask]) * counts_per_pixel[mask]
    )
    if do_verbose:
        print(
            f"[calculate_spatial_poisson_probability] Calculating second_term took {time.time() - start} seconds"
        )

    start = time.time()
    # 4: third term, summation over the number of pixels: - \sum_{i=1, k} \ln(n_i!)
    # and ln(ni!) is calculated as gammaln(ni+1) = ln(|Gamma(ni+1)|) = ln(ni!)
    mask = counts_per_pixel > 0
    third_term = numpy.sum(scipy.special.gammaln(counts_per_pixel[mask] + 1))
    if do_verbose:
        print(
            f"[calculate_spatial_poisson_probability] Calculating third_term took {time.time() - start} seconds"
        )

    start = time.time()
    # calculate the Poisson probability as
    # ln P = sum_over pixels ln(lambda (pixels))*ni - sum over pixels lambda(pixels) - sum over pixels ln(ni!)
    # where ni (or counts_per_pixel) is the number of GCs per pixel
    ln_prob = -first_term + second_term - third_term

    if do_verbose:
        print(
            "[calculate_spatial_poisson_probability] Terms are {:.10e}, {:.10e}, {:.10e} - ln_prob {:.10e}".format(
                -first_term, second_term, -third_term, ln_prob
            )
        )
        print(
            f"[calculate_spatial_poisson_probability] Calculating ln_prob took {time.time() - start} seconds"
        )
    return ln_prob
