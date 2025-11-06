import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Playground to try a couple of methods to resize/rebin images""")
    return


@app.cell
def _():
    import numpy as np
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    from reproject import reproject_interp, reproject_adaptive

    # Set up initial array with pattern
    input_array = np.zeros((200, 256))
    input_array[::20, :] = 1
    input_array[:, ::20] = 1
    input_array[10::20, 10::20] = 1

    # Define a simple input WCS
    input_wcs = WCS(naxis=2)
    input_wcs.wcs.crpix = 128.5, 128.5
    input_wcs.wcs.cdelt = -0.01, 0.01

    # Define a lower resolution output WCS with rotation
    output_wcs = WCS(naxis=2)
    output_wcs.wcs.crpix = 64.5, 64.5
    output_wcs.wcs.cdelt = input_wcs.wcs.cdelt * (
        (input_wcs.wcs.crpix - 0.5) / (output_wcs.wcs.crpix - 0.5)
    )
    # output_wcs.wcs.cdelt =  -0.02, 0.02
    print(output_wcs.wcs.cdelt)

    # Reproject using interpolation and adaptive resampling
    result_interp, foot_interp = reproject_interp(
        (input_array, input_wcs), output_wcs, shape_out=(128, 128)
    )
    result_hann, foot_hahn = reproject_adaptive(
        (input_array, input_wcs), output_wcs, shape_out=(128, 128), kernel="hann"
    )
    result_gaussian, foot_gaussian = reproject_adaptive(
        (input_array, input_wcs), output_wcs, shape_out=(128, 128), kernel="gaussian"
    )
    print(
        input_array.shape, result_interp.shape, result_hann.shape, result_gaussian.shape
    )
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 4, 1)
    plt.imshow(input_array, origin="lower", vmin=0, vmax=1, interpolation="hanning")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("Input array")
    plt.subplot(2, 4, 2)
    plt.imshow(result_interp, origin="lower", vmin=0, vmax=1)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("reproject_interp")
    plt.subplot(2, 4, 3)
    plt.imshow(result_hann, origin="lower", vmin=0, vmax=0.5)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("reproject_adaptive\nHann kernel")
    plt.subplot(2, 4, 4)
    plt.imshow(result_gaussian, origin="lower", vmin=0, vmax=0.5)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("reproject_adaptive\nGaussian kernel")
    plt.subplot(2, 4, 5)
    plt.imshow(input_array, origin="lower", vmin=0, vmax=1, interpolation="hanning")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("Input array")
    plt.subplot(2, 4, 6)
    plt.imshow(foot_interp, origin="lower", vmin=0, vmax=1, interpolation="hanning")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.subplot(2, 4, 7)
    plt.imshow(foot_hahn, origin="lower", vmin=0, vmax=1, interpolation="hanning")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.subplot(2, 4, 8)
    plt.imshow(foot_gaussian, origin="lower", vmin=0, vmax=1, interpolation="hanning")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()
    return input_array, plt


@app.cell
def _(input_array, plt):
    import skimage

    image_resized = skimage.transform.resize(
        input_array, (128, 128), anti_aliasing=True
    )

    _fig, _axs = plt.subplots(1, 2)
    _axs[0].imshow(input_array, origin="lower", vmin=0, vmax=1)
    _axs[1].imshow(image_resized, origin="lower", vmin=0, vmax=1)
    plt.show()

    return


@app.cell
def _():
    import numpy as np
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    from reproject import reproject_interp, reproject_adaptive

    # Set up initial array with pattern
    input_array = np.zeros((256, 256))
    input_array[::20, :] = 1
    input_array[:, ::20] = 1
    input_array[10::20, 10::20] = 1

    # Define a simple input WCS
    input_wcs = WCS(naxis=2)
    input_wcs.wcs.crpix = 128.5, 128.5
    input_wcs.wcs.cdelt = -0.01, 0.01

    # Define a lower resolution output WCS with rotation
    output_wcs = WCS(naxis=2)
    output_wcs.wcs.crpix = 64.5, 64.5
    output_wcs.wcs.cdelt = -0.02, 0.02

    # Reproject using interpolation and adaptive resampling
    result_interp, _ = reproject_interp(
        (input_array, input_wcs), output_wcs, shape_out=(128, 128)
    )
    result_hann, _ = reproject_adaptive(
        (input_array, input_wcs), output_wcs, shape_out=(128, 128), kernel="hann"
    )
    result_gaussian, _ = reproject_adaptive(
        (input_array, input_wcs), output_wcs, shape_out=(128, 128), kernel="gaussian"
    )
    print(
        input_array.shape, result_interp.shape, result_hann.shape, result_gaussian.shape
    )
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(input_array, origin="lower", vmin=0, vmax=1, interpolation="hanning")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("Input array")
    plt.subplot(1, 4, 2)
    plt.imshow(result_interp, origin="lower", vmin=0, vmax=1)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("reproject_interp")
    plt.subplot(1, 4, 3)
    plt.imshow(result_hann, origin="lower", vmin=0, vmax=0.5)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("reproject_adaptive\nHann kernel")
    plt.subplot(1, 4, 4)
    plt.imshow(result_gaussian, origin="lower", vmin=0, vmax=0.5)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title("reproject_adaptive\nGaussian kernel")
    return input_array, plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
