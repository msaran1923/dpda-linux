import random
import time
import numpy as np

class DataAugmenter:
    def __init__(self, imageLoader, imageSaver):
        self.imageLoader = imageLoader
        self.imageSaver = imageSaver
        self.pipelineDataAugmenter = None

        random.seed(int(time.time()))

    def execute(self, inputDirectory, outputDirectory, imagePath, augmentationCount, scaleFactor, augmentationPercentage):
        pass

    def augmentImage(self, image):
        pass

    def setPipelineDataAugmenter(self, pipelineDataAugmenter):
        self.pipelineDataAugmenter = pipelineDataAugmenter

    def sqr(x):
        return x * x

    @staticmethod
    def cube(x):
        return x * x * x

    @staticmethod
    def randUnity(size=None):
        randomRange = 8192
        return np.random.randint(0, randomRange, size=size) / (randomRange - 1.0)

    @staticmethod
    def randRange(minValue, maxValue):
        randomRange = 8192
        return (random.randint(0, randomRange - 1) / (randomRange - 1.0)) * (maxValue - minValue) + minValue
