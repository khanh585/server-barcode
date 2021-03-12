class DoiTuong:

    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.is_anonymus = name == 'barcode'
        self.id = 'anonymus'
        self.is_dead = False
        self.tracking = -1

        c1, c2 = (int(self.position[0]), int(self.position[1])), (int(self.position[2]), int(self.position[3]))

        self.width = c2[0] - c1[0]
        self.height = c2[1] - c1[1]

    def setImage(self, image):
        if self.is_anonymus:
            self.image = image

    def setID(self, id):
        if id != '':
            self.id = id
            self.is_anonymus = False
    
    def setTracking(self, tracking):
        self.tracking = tracking
    
    def toBbox(self):
        c1, c2 = (int(self.position[0]), int(self.position[1])), (int(self.position[2]), int(self.position[3]))

        p1 = c1
        p2 = (c1[0] + self.width, c1[1])
        p3 = c2
        p4 = (c1[0], c1[1] + self.height)

        return  p1, p2, p3, p4

    def __repr__(self):
        return repr((self.id, self.position))
        