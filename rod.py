class Rod:

    def __init__(self):
        self.type = ""
        self.orientation = 0
        self.barycenter = []
        self.length = 0
        self.width = 0
        self.width_b = 0
        self.holes = []

    def __repr__(self):
        return f" Rod type: {self.type} \n" \
               f"num holes: {len(self.holes)} \t -> holes: {self.holes} \n" \
               f"ib: {self.barycenter[0]}, jb: {self.barycenter[1]} \n" \
               f"orientation: {self.orientation}" \
               f"barycenter width barycenter: {self.width_b} \n" \
               f"width: {self.width} \t length: {self.length} \n \n"

    def __str__(self):
        return f" Rod type: {self.type} \n" \
               f"num holes: {len(self.holes)} \t -> holes: {self.holes} \n" \
               f"ib: {self.barycenter[0]}, jb: {self.barycenter[1]} \n" \
               f"orientation: {self.orientation}" \
               f"barycenter width barycenter: {self.width_b} \n" \
               f"width: {self.width} \t length: {self.length} \n \n"

    def assign_holes(self, diameter, centroids):
        self.holes.append({"D": diameter,
                           "Cx": centroids[0],
                           "Cy": centroids[1]})
