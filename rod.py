import cv2.cv2
from importlib_metadata import SelectableGroups
import numpy as np


class Rod:

    def __init__(self):
        self.file = ""
        self.type = ""
        self.orientation = 0
        self.barycenter = []
        self.length = 0
        self.width = 0
        self.width_b = 0
        self.holes = []

    def __repr__(self):
        return f"type: {self.type}  orientation: {self.orientation} " \
               f"num holes: {len(self.holes)} -> \n\t" \
               f"ib: {self.barycenter[0]}, jb: {self.barycenter[1]} \n" \
               f"barycenter width: {self.width_b} \n" \
               f"holes: {self.holes} \n"

    def __str__(self):
        return f"type: {self.type}  orientation: {self.orientation} " \
               f"num holes: {len(self.holes)} -> \n\t" \
               f"ib: {self.barycenter[0]}, jb: {self.barycenter[1]} \n"

    def assign_holes(self, diameter,centroids):
        self.holes.append({"D": diameter,
                               "Cx": centroids[0],
                               "Cy": centroids[1]})