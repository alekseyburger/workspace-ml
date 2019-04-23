import numpy as np

class Neuron:
    pass

class Dense_Neuron(Neuron):

    desc_coefficient = 0.001

    def __init__(self, size):
        self.W = np.random.rand(size)*self.desc_coefficient
        self.B = np.random.rand(1)*self.desc_coefficient
        self.gradient = np.zeros((size))

    def run(self, input):
        self.res = np.dot(self.W, input) + self.B
        if self.res <= 0:
            self.res = 0
            self.gradientB = 0
            self.gradient = np.zeros_like(self.gradient)
        else:
            self.gradient = input.copy()
            self.gradientB = 1
        return self.res

    def back_propagation(self, gradient = None):
        assert(gradient.ndim == 0)
        self.gradient *= gradient
        self.W -= self.desc_coefficient*self.gradient
        self.B -= self.desc_coefficient*gradient*self.gradientB

class Layer:

    def __init__(self, number, input_size=0, prev = None):
        self.prev = prev
        self.number = number
        if self.prev:
            input_size = self.prev.number
        self.input_size = input_size
        self.output = np.zeros(shape=(number), dtype="float32")
        self.neurons = []
        for _ in range(number):
            self.neurons.append(Dense_Neuron(input_size))
        self.gradient = np.zeros((input_size))

    def run(self, input = None):
        if self.prev:
            input = self.prev.run(input)

        if input.shape != (self.input_size,):
            raise TypeError("wring input size")
        
        for i, n in enumerate(self.neurons):
            res = n.run(input)
            self.output.itemset(i,res)

        return self.output

    def back_propagation(self, gradient = None):

        self.gradient = np.zeros_like(self.gradient)
        for i, n in enumerate(self.neurons):
            self.gradient += n.W * gradient[i] * n.gradientB
        if self.prev:
            self.prev.back_propagation(self.gradient)

        for i, n in enumerate(self.neurons):
            n.back_propagation(gradient[i])
        

class SoftMax(Layer):

    def run(self, input = None):
        
        input = self.prev.run(input)

        if input.shape != (self.input_size,):
            raise TypeError("wring input size")

        self.output = np.exp(input)

        sum = self.output.sum()
        self.output = self.output/sum

        self.gradient = self.output * (np.ones(self.input_size)-self.output)

        return self.output

    def back_propagation(self, gradient = None):

        self.gradient = self.gradient * gradient
        self.prev.back_propagation(self.gradient)

class Loss:

    def __init__(self, input_size=0, prev = None):
        self.prev = prev
        if self.prev:
            input_size = self.prev.number
        self.input_size = input_size
        #self.output = np.zeros(shape=(number), dtype="float32")
        self.loss = 0.0
        self.gradient = np.zeros((input_size))


    def run(self, input, label):
        if self.prev:
            input = self.prev.run(input)

        if input.shape != (self.input_size,):
            raise TypeError("wring input size")
        # if labels.shape != (self.input_size,):
        #     raise TypeError("wring label size")

        self.gradient = np.zeros_like(self.gradient)
        self.gradient[label] = -1.0/input[label]    # grad log(x) = 1/x
        print(input)
        print(self.gradient)

        self.loss -= np.log(input[label])

        if np.isnan(self.loss):
            print(input)
            raise Exception()
        return self.loss

    def back_propagation(self, gradient = None):

        self.prev.back_propagation(self.gradient)
        self.loss = 0
        

        

if __name__ == "__main__":
    # import os
    # from PIL import Image
    # import matplotlib.pyplot as plt

    # def resize_npimage(original_image, width , height, channels):
    #     original_width, original_height, original_channels = original_image.shape
    #     print(original_width, original_height, original_channels)
    #     resize_image = np.zeros(shape=(width,height,channels), dtype="int32")

    #     for C in range(channels):
    #         for W in range(width):
    #             for H in range(height):
    #                 new_width = int( W * original_width / width )
    #                 new_height = int( H * original_height / height )
    #                 resize_image[W][H][C] = original_image.item((new_width,new_height,C))
    #     return resize_image

    # def load_image( name, width=128, height=128, channels=3) :
    #     img = Image.open( name )
    #     img.load()
    #     data = np.asarray( img, dtype="int32" )
    #     data = resize_npimage(data, width , height, channels)
    #     # plt.figure()
    #     # plt.imshow(data)
    #     # plt.show()
    #     mul = 1
    #     for d in data.shape:
    #         mul *= d
    #     return np.reshape(data,(mul)).astype(dtype="float32")

    # src_path = "/home/alekseyb/workspace-ml-save/datasets/cat_dog_wolf/test_set/cats"
    # newimages = os.listdir(src_path)
    # #for image in newimages:
    # image = os.path.join(src_path, newimages[0])
    # #try:
    # img = load_image(image, 128, 128, 3)
    # inp_size = img.shape[0]
    # print(inp_size)
    # print(type(img[0]))
    # # except:
    # #     print("wrong image format: {}".format(image))
    # #     exit(0)

    import MNIST
    import math
    from functools import reduce

    def mnist_img_label():
        for img,label in zip(mnist.test_set_images(),mnist.test_set_labels()):

            data = np.asarray( img, dtype="float64" )
            data = np.reshape(data,(inp_size))
            data = data/256.0
            yield (data, label)

    mnist = MNIST.MNIST()
    inp_size = reduce(lambda mul,e: mul*e, mnist.get_shape())

    batch_size = 16
    l1 = Layer(140, inp_size)
    l2 = Layer(140, prev=l1)
    model = Layer(10, prev=l2)
    sm = SoftMax(10, prev=model)
    loss = Loss(prev=sm)

    count = batch_size
    for data,label in mnist_img_label():
        out = loss.run(data, label)
        loss.back_propagation()
        print(out)

        #count -= 1
        #if not count: break
        