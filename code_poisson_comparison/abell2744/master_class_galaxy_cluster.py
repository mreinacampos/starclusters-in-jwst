### Script containing the definition of the GalaxyCluster class
### Author: Marta Reina-Campos
### Date: October 2025

# Import modules
from astropy import units as u
from astropy.units import Quantity


class GalaxyCluster:
    """This class is used to instantiate the galaxy cluster we're studying.

    :param name: name of the galaxy cluster
    :param distance: luminosity distance - astropy length unit
    :param redshift: redshift of the galaxy cluster
    :param arsec_to_kpc: conversion factor from arcsec to kpc at the distance of the galaxy cluster - astropy length / angle unit
    """

    type_class = "galaxy cluster"  # information about the kind of object we're studying

    def __init__(
        self,
        name: str,
        distance: Quantity["length"],
        redshift: float,
        arcsec_to_kpc: (u.kpc / u.arcsec),
    ) -> None:
        self.name = name  # name of the galaxy cluster
        self.distance = distance  # luminosity distance to the galaxy cluster, in kpc
        self.redshift = redshift  # redshift of the galaxy cluster
        self.arcsec_to_kpc = arcsec_to_kpc  # conversion between arcsec and kpc at the distance of the galaxy cluster
