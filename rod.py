import cv2.cv2
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

    def assign_holes(self, num, stats, centroids):
        for i in range(2, num):
            self.holes.append({"D": 2 * np.sqrt(stats[i][cv2.cv2.CC_STAT_AREA] / np.pi),
                               "Cx": centroids[i, 0],
                               "Cy": centroids[i, 1]})
        self.assign_type()

    def assign_type(self):
        if len(self.holes) == 1:
            self.type = "A"
        elif len(self.holes) == 2:
            self.type = "B"
        else:
            self.type = "unknown"

    def __repr__(self):
        return f"type: {self.type}  orientation: {self.orientation} " \
               f"num holes: {len(self.holes)} -> \n\t" \
               f"ib: {self.barycenter[0]}, jb: {self.barycenter[1]} \n" \
               f"barycenter width: {self.width_b} \n"

    def __str__(self):
        return f"type: {self.type}  orientation: {self.orientation} " \
               f"num holes: {len(self.holes)} -> \n\t" \
               f"ib: {self.barycenter[0]}, jb: {self.barycenter[1]} \n"