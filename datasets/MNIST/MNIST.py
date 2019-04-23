import os

class MNIST:

    BYTEORDER='big'
    TRAIN_IMAGES_NAME = "train-images.idx3-ubyte"
    TRAIN_LABELS_NAME = "train-labels.idx1-ubyte"
    TEST_IMAGES_NAME = "t10k-images.idx3-ubyte"
    TEST_LABELS_NAME = "t10k-labels.idx1-ubyte"
    IMAGES_MAGIC = 2051
    LABELS_MAGIC = 2049
    COLORS = 1

    def __init__(self):
        self.scripts_dir = os.path.abspath(os.path.dirname(__file__))

        with open(os.path.join(self.scripts_dir,self.TRAIN_IMAGES_NAME), "rb") as f:
            read_magic = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)
            if read_magic != self.IMAGES_MAGIC:
                raise Exception ("Wrong image magic {} in {}".format(read_magic,self.TRAIN_IMAGES_NAME))
            self.number = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)
            self.rows = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)
            self.columns = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)

    def get_shape(self):
        return (self.rows, self.columns, self.COLORS)

    def images_iter(self, name):
        with open(os.path.join(self.scripts_dir,name), "rb") as f:
            read_magic = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)
            if read_magic != self.IMAGES_MAGIC:
                raise Exception ("Wrong image magic {} in {}".format(read_magic,self.TRAIN_IMAGES_NAME))
            number = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)
            rows = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)
            columns = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)

            for _ in range(number):
                image  = []
                for _ in range(rows):
                    row = []
                    for _ in range(columns):
                        row.append(int.from_bytes(f.read(1), byteorder=self.BYTEORDER))
                    image.append(row)
                yield image.copy()

    def labels_iter(self, name):
        with open(os.path.join(self.scripts_dir,name), "rb") as f:
            read_magic = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)
            if read_magic != self.LABELS_MAGIC:
                raise Exception ("Wrong labels magic {} in {}".format(read_magic,self.TRAIN_IMAGES_NAME))
            number = int.from_bytes(f.read(4), byteorder=self.BYTEORDER)

            for _ in range(number):
                yield int.from_bytes(f.read(1), byteorder=self.BYTEORDER)

    def train_set_images(self):
        """
        Generator return list with train image pixels. Shape (self.rows, self.columns, self.COLORS)
        """
        for img in self.images_iter(self.TRAIN_IMAGES_NAME):
            yield img

    def train_set_labels(self):
        """
        Generator returns label for train image.
        """
        for label in self.labels_iter(self.TRAIN_LABELS_NAME):
            yield label

    def test_set_images(self):
        """
        Generator return list with test image pixels. Shape (self.rows, self.columns, self.COLORS)
        """
        for img in self.images_iter(self.TEST_IMAGES_NAME):
            yield img

    def test_set_labels(self):
        """
        Generator returns label for test image.
        """
        for label in self.labels_iter(self.TEST_LABELS_NAME):
            yield label

if __name__ == "__main__":
    # import os
    # from PIL import Image
    import matplotlib.pyplot as plt

    mnist = MNIST()
    for img, label in zip(mnist.test_set_images(), mnist.test_set_labels()):
        plt.figure()
        plt.title(str(label))
        plt.imshow(img)
        plt.show()