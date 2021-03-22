class DoiTuong:

    def __init__(self, position):
        self.name = ''
        self.position = position

        c1, c2 = (int(self.position[0]), int(self.position[1])), (int(self.position[2]), int(self.position[3]))

        self.width = c2[0] - c1[0]
        self.height = c2[1] - c1[1]
        cX = int((self.position[0] + self.position[2]) / 2.0)
        cY = int((self.position[1] + self.position[3]) / 2.0)
        self.centroid = (cX,cY)

    def setImage(self, image):
        self.image = image

    
    def toBbox(self):
        c1, c2 = (int(self.position[0]), int(self.position[1])), (int(self.position[2]), int(self.position[3]))

        p1 = c1
        p2 = (c1[0] + self.width, c1[1])
        p3 = c2
        p4 = (c1[0], c1[1] + self.height)

        return  p1, p2, p3, p4

    def __repr__(self):
        return repr((self.name, self.position))
        