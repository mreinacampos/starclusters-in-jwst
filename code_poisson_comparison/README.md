### Author: Marta Reina-Campos
### Date: November 2025
---

This is the README file for the code in the code_poisson_comparison directory. The goal of these scripts is to, given a sample of GCs in a galaxy cluster, and a series of images describing the distribution of mass, stellar light, and/or X-ray emitting gas in the cluster, to compute the log-likelihood that an inhomogeneous Poisson point process can describe the observed distribution of GCs given a galactic component. This analysis assesses how similar the observed distribution of GCs is to the different galactic components, i.e. how well it traces them. A similar analysis is done between the maps, spawning datapoint according to a map 1 and the observatioan lprobability of recovery and comparing against a map 2. This calculation ('map-to-map') establishes the values against which to compare the observational 'point-to-map' results.

The directory abell2744 is the sandbox containing all the scripts and directories:
* data -> location of the input maps and GC catalogue
* imgs -> location of the resulting images
* tables -> location of the output tables for the observational results ('point-to-map') and the 'map-to-map' comparisons, as well as their validation images

Key scripts are:
* master_functions_abell2744.py -> contains the loaders for all the maps and GC catalogue, and need to be updated for every galaxy cluster
* 01_spatial_distributions.ipynb -> Jupyter Notebook with visualizations of the different gaalctic components
* 02_poisson_process_points_to_maps.py -> calculation of the log-likelihood between the distribution of GCs and different maps
* 02_poisson_process_maps_to_maps.py -> calculation of the log-likelihood between two sets of maps
* 02d_plot_poisson_process.py -> visualization rotuines of the log-likelihoods between the different galactic components
* 03_pixel_by_pixel_comparisons.py -> pixel-by-pixel comparison between the smoothed number density distribution of an observed sample of GCs and a given galactic component