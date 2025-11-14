from __future__ import absolute_import, division, print_function

from collections import namedtuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# orthomosaics for which we have labels
ortho_names = ["blackburn1", "blackburn3", "bernard1", "bernard2", "bernard4"]


def get_species(ortho_names):
    # create species dictionary of unique species in an orthomosaic
    species = []
    for name in ortho_names:
        df = gpd.read_file(
            f"/network/scratch/t/tengmeli/lefo-co2/Annotations_geojson/{name}.json"
        )
        classcodes = ann.classcode.unique()
        classvalues = ann.classvalue.unique()
        unique = zip(classcodes, classvalues)
        print(unique)
        species += unique
    return set(species)


# a label and all meta information
Label = namedtuple(
    "Label",
    [
        "name",  # name of the species
        "classvalue",  # original class value in the annotations
        "trainId",  # ID for training
        "category",
    ],
)

labels = [
    #       name                     id    trainId   category
    Label("Other", 0, 0, "Other"),
    Label("Picea glauca", 7173, 1, "Picea"),
    Label("Picea mariana", 7174, 2, "Picea"),
    Label("Pinus banksiana", 7181, 3, "Pinus"),
    Label("Pinus strobus", 7195, 4, "Pinus"),
]

# classvalue to trainID object
classvalue2trainID = {label.classvalue: label.trainId for label in labels}
